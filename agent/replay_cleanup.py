"""Replay-history sanitization shared by EVERY resume surface (messaging gateway, TUI/WebUI gateway).

A turn that died mid-tool-loop (restart command, stale timeout, interrupt before the result was written)
persists a dangling ``assistant(tool_calls)`` or interrupted ``assistant→tool`` tail; on resume the model
re-issues the unanswered call → endless "thinking"/reboot loop. These pure helpers strip those tails."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_result_classification import tool_may_have_side_effect
from agent.turn_context import drop_stale_api_content
from hermes_cli.timefmt import coerce_epoch

logger = logging.getLogger(__name__)

# Orphan-recovery notices: (side-effecting, read-only) for an interrupted block vs a dangling tail.
_INTERRUPTED_NOTICES = (
    "[Orphan recovery: interrupted side-effecting tool may have executed; its effect is UNKNOWN. Inspect state before retrying.]",
    "[Orphan recovery: interrupted read-only tool did not complete.]",
)
_DANGLING_NOTICES = (
    "[Orphan recovery: this tool may have executed before Hermes stopped; its effect is UNKNOWN. Inspect current state before retrying.]",
    "[Orphan recovery: this read-only tool did not complete and had no effect.]",
)


# Every executor ends the killed run's ``output`` with a bracketed marker line: "[Command
# interrupted]" (tools/environments/, exit 130), "[Command interrupted - Modal ...]"
# (managed_modal.py, exit 130), "[execution interrupted ...]" (code_execution_tool.py, exit -1).
_INTERRUPT_MARKER_LINE = re.compile(r"^\[(?:command|execution) interrupted\b[^\n]*\]\s*$", re.IGNORECASE)


def is_interrupted_tool_result(content: Any) -> bool:
    """True only when the result has the executor's interrupt SHAPE: the marker is the last
    line of the output (JSON envelope with a non-zero exit code, or a bare text result). A
    marker quoted inside successful output — a grep hit, a doc example — is ordinary data;
    this runs on every live request, so a false positive rewrites real tool output."""
    if not isinstance(content, str):
        return False
    output = content
    if content.lstrip().startswith("{"):
        try:
            envelope = json.loads(content)
        except ValueError:
            return False
        if not isinstance(envelope, dict) or envelope.get("exit_code") in (0, None):
            return False
        output = envelope.get("output")
        if not isinstance(output, str):
            return False
    last_line = output.rstrip().rsplit("\n", 1)[-1]
    return _INTERRUPT_MARKER_LINE.match(last_line) is not None


def _call_name(call: Dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _call_id(call: Dict[str, Any]) -> str:
    return str(call.get("id") or call.get("call_id") or "")


def _any_side_effecting(calls: List[Dict[str, Any]]) -> bool:
    return any(tool_may_have_side_effect(_call_name(call)) for call in calls)


def _orphan_recovery(name: str, notices: tuple) -> tuple:
    """(effect_disposition, content) for an interrupted/dangling call named ``name``."""
    if tool_may_have_side_effect(name):
        return "unknown", notices[0]
    return "none", notices[1]


def strip_interrupted_tool_tails(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip interrupted assistant→tool blocks anywhere in history (a queued user message may follow one).
    Read-only blocks are dropped; blocks with a side-effecting call are KEPT with the interrupted results
    rewritten as orphan-recovery notices, since the effect may have happened."""
    if not agent_history:
        return agent_history
    cleaned: List[Dict[str, Any]] = []
    i, n = 0, len(agent_history)
    while i < n:
        msg = agent_history[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            j = i + 1
            while j < n and agent_history[j].get("role") == "tool":
                j += 1
            tool_results = agent_history[i + 1:j]
            if any(is_interrupted_tool_result(m.get("content", "")) for m in tool_results):
                calls = msg.get("tool_calls") or []
                if _any_side_effecting(calls):
                    call_names = {_call_id(call): _call_name(call) for call in calls}
                    cleaned.append(msg)
                    for tool_result in tool_results:
                        if is_interrupted_tool_result(tool_result.get("content", "")):
                            name = call_names.get(str(tool_result.get("tool_call_id") or ""), "")
                            disposition, content = _orphan_recovery(name, _INTERRUPTED_NOTICES)
                            tool_result = {**tool_result, "effect_disposition": disposition, "content": content}
                        cleaned.append(tool_result)
                else:
                    logger.debug("Stripping interrupted read-only assistant→tool replay block (indices %d–%d, tool_results=%d)",
                                 i, j - 1, len(tool_results))
                i = j
                continue
        if msg.get("role") == "tool" and is_interrupted_tool_result(msg.get("content", "")):
            logger.debug("Stripping orphan interrupted tool result from replay history")
        else:
            cleaned.append(msg)
        i += 1
    return cleaned


def strip_dangling_tool_call_tail(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Close unanswered replay blocks, including partial blocks and queued-user gaps.

    Keep the historical entry-point name for callers. Completed blocks are verbatim;
    wholly unanswered read-only blocks may still be dropped. Otherwise preserve all
    attempted calls and real results, appending per-call recovery notices inside the
    original block, before any following user message. These are not success receipts.
    """
    from agent.message_sanitization import (
        coalesce_tool_call_id, tool_call_id_variants, tool_result_id_variants,
    )
    from agent.message_metadata import tool_call_occurrence_uids

    changed = False
    if any(a.get("role") == b.get("role") == "assistant"
           for a, b in zip(agent_history, agent_history[1:])):
        # Alternation repair already folds these rows before matching results.
        # Preserve that union/occurrence-UID behavior, but do not prune calls yet
        # or mutate the caller's history while folding.
        from copy import deepcopy
        from agent.agent_runtime_helpers import _merge_consecutive_assistants
        folded, repairs = _merge_consecutive_assistants(deepcopy(agent_history))
        if repairs:
            agent_history = folded
            changed = True
    cleaned: List[Dict[str, Any]] = []
    i = 0
    while i < len(agent_history):
        owner = agent_history[i]
        calls = owner.get("tool_calls") if owner.get("role") == "assistant" else None
        if not calls:
            cleaned.append(owner)
            i += 1
            continue
        j = i + 1
        while j < len(agent_history) and agent_history[j].get("role") == "tool":
            j += 1
        results = agent_history[i + 1:j]
        uids = tool_call_occurrence_uids(owner)
        variants = [tool_call_id_variants(call) for call in calls]
        answered = set()
        for result in results:
            aliases = tool_result_id_variants(result.get("tool_call_id"))
            candidates = [k for k, ids in enumerate(variants) if ids & aliases]
            result_uid = result.get("_tool_call_uid")
            if result_uid:
                answered.update(k for k in candidates if uids[k] == result_uid)
            elif len(candidates) == 1 or (
                candidates and uids[candidates[0]] is not None
                and len({uids[k] for k in candidates}) == 1
            ):
                # Legacy alias-only results can close an unambiguous call (or
                # slots deliberately sharing one native occurrence), not siblings.
                answered.update(candidates)
            elif result.get("effect_disposition") in {"unknown", "none"}:
                # Our identity-less notices are not receipts. Count them one per
                # unidentified slot so conservative legacy recovery is idempotent.
                pending = [k for k in candidates if uids[k] is None and k not in answered]
                if pending:
                    answered.add(pending[0])
        missing = [k for k in range(len(calls)) if k not in answered]
        if not missing:
            cleaned.extend(agent_history[i:j])
        elif not results and not _any_side_effecting(calls):
            changed = True
            logger.debug("Stripping wholly unanswered read-only replay block")
        else:
            changed = True
            cleaned.extend(agent_history[i:j])
            for k in missing:
                call = calls[k]
                call_id = coalesce_tool_call_id(call)
                name = _call_name(call) or "unknown"
                disposition, content = _orphan_recovery(name, _DANGLING_NOTICES)
                notice = make_tool_result_message(
                    name, content, call_id, effect_disposition=disposition)
                uid = uids[k]
                if uid is not None:
                    notice["_tool_call_uid"] = uid
                cleaned.append(notice)
            logger.warning("Recovered unanswered replay call(s) with per-call effect notices")
        i = j
    return cleaned if changed else agent_history


def sanitize_replay_history(agent_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Both strippers in canonical order (interrupted blocks, then dangling tail); same list object when nothing strips."""
    if not agent_history:
        return agent_history
    return strip_dangling_tool_call_tail(strip_interrupted_tool_tails(agent_history))


def canonicalize_replay_history(
    agent_history: List[Dict[str, Any]], *, now: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Apply every destructive replay transform in the shared, fixed order.

    Resume surfaces and the send path must serialize the same history bytes, or a
    resumed request diverges in the middle of the cached prefix.

    The input is never modified. ``now`` is the expiry clock; the send path passes the
    turn's admission time so every request in one turn sees the same bytes.
    """
    if not agent_history:
        return agent_history
    if now is None:
        now = time.time()
    cleaned = strip_interrupted_tool_tails(agent_history)
    cleaned = strip_dangling_tool_call_tail(cleaned)
    return strip_stale_dangerous_confirmations(cleaned, now=now)


# --- Stale dangerous-confirmation text expiry ---

# Short on purpose: a dangerous confirmation must not survive any restart or resume gap.
# ────────────────────────────────────────────────────────────────────── Stale dangerous-confirmation text
# expiry (#59607) ──────────────────────────────────────────────────────────────────────
_DANGEROUS_CONFIRMATION_EXPIRY_SECONDS = 60.0

# Phrases that unlock destructive host actions; case-insensitive substring match so trailing punctuation /
# extra context still matches. Includes i18n variants from the original incident.
_DANGEROUS_CONFIRMATION_PATTERNS: tuple = (
    "confirm forced restart", "confirm forced reboot", "confirm shutdown", "confirm reboot", "confirm power off",
    "yes, delete everything", "confirm wipe", "confirm factory reset",
    "確認強制重開機", "確認強制重開", "確認重啟",
)

# Redacting in place (not deleting the message) preserves strict user/assistant alternation in the replay.
_EXPIRED_CONFIRMATION_SENTINEL = (
    "[A high-risk confirmation previously given here has EXPIRED and must "
    "not be acted on. Ask the user to re-confirm explicitly before "
    "performing any destructive action.]"
)


def is_dangerous_confirmation(content: Any) -> bool:
    """True if user-message text contains a known dangerous confirmation phrase."""
    if not isinstance(content, str):
        return False
    lowered = content.strip().lower()
    return any(pattern in lowered for pattern in _DANGEROUS_CONFIRMATION_PATTERNS)


def strip_stale_dangerous_confirmations(
    agent_history: List[Dict[str, Any]],
    *,
    now: float,
    expiry_seconds: float = _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS,
) -> List[Dict[str, Any]]:
    """Redact IN PLACE dangerous-confirmation text older than ``expiry_seconds`` in user messages: a confirmation
    surviving a restart reads as a fresh re-confirmation minutes later. Untimestamped messages (legacy
    transcripts, test scaffolding) are left untouched.

    See #59607.
    On the next inbound message — possibly a casual "are you there?" from the user minutes later — the LLM
    sees the stale confirmation and may interpret the new turn as a fresh re-confirmation, re-executing the
    destructive action. This is the failure mode reported in #59607.
    """
    if not agent_history:
        return agent_history
    cleaned: List[Dict[str, Any]] = []
    for msg in agent_history:
        ts = msg.get("timestamp") if isinstance(msg, dict) and msg.get("role") == "user" else None
        if ts is None or not is_dangerous_confirmation(msg.get("content", "")):
            cleaned.append(msg)
            continue
        # A present-but-untrustworthy stamp (corrupt, or issued in the future relative to
        # the admission clock) is treated as expired: its age is unknowable, and keeping the
        # text (plus its api_content sidecar) would replay a live confirmation.
        ts_f = coerce_epoch(ts, field="message timestamp")
        age = math.inf if ts_f is None else now - ts_f
        if 0 <= age <= expiry_seconds:
            cleaned.append(msg)
            continue
        logger.debug(
            "Redacting stale dangerous-confirmation text in user message (age=%.1fs, expiry=%.1fs): %r",
            age, expiry_seconds, (msg.get("content") or "")[:80],
        )
        redacted = dict(msg)
        redacted["content"] = _EXPIRED_CONFIRMATION_SENTINEL
        # The api_content sidecar carries the exact bytes sent — the confirmation itself; replaying it would undo the redaction.
        drop_stale_api_content(redacted)
        cleaned.append(redacted)
    return cleaned
