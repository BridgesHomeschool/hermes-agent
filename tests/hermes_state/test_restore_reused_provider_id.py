"""Portable regressions for distinct original writes sharing a provider call ID."""
import copy
import json

import pytest

from agent.replay_cleanup import canonicalize_replay_history
from hermes_state import SessionDB


def call(key):
    return {
        "id": "same-provider-id", "call_id": "same-provider-id", "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"idempotency_key": key, "fingerprint": "fingerprint-" + key}),
        },
    }


@pytest.mark.parametrize("already_folded", [False, True])
@pytest.mark.parametrize("has_real_result", [False, True])
def test_each_unanswered_occurrence_keeps_its_own_uid(tmp_path, has_real_result, already_folded):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", "cli")
        first = {"role": "assistant", "content": "first", "tool_calls": [call("original-first-key")]}
        second = {"role": "assistant", "content": "second", "tool_calls": [call("original-second-key")]}
        rows = [{"role": "user", "content": "start"}, first, second]
        if has_real_result:
            rows.append({"role": "tool", "tool_call_id": "same-provider-id", "content": "real second receipt"})
        db.append_messages_batch("s", rows)
        uids = [row["_tool_call_uids"]["same-provider-id"] for row in (first, second)]
        assert uids[0] != uids[1]
        original_calls = copy.deepcopy(first["tool_calls"] + second["tool_calls"])
        if already_folded:
            # Reproduce the native persisted merged representation, not arbitrary
            # malformed input. SessionDB already assigned both original identities.
            db._conn.execute(
                "UPDATE messages SET tool_calls = ?, tool_call_uids = ? "
                "WHERE session_id = 's' AND role = 'assistant' AND content = 'first'",
                (json.dumps(original_calls), json.dumps({"same-provider-id": uids})),
            )
            db._conn.execute("DELETE FROM messages WHERE session_id = 's' AND role = 'assistant' AND content = 'second'")
            db._conn.commit()
        raw = db.get_messages_as_conversation("s", include_row_ids=True)
        frozen = copy.deepcopy(raw)
        dump = list(db._conn.iterdump())
        expected = uids[:1] if has_real_result else uids
        got = canonicalize_replay_history(raw, now=100)
        assert got[1]["tool_calls"] == original_calls
        assert got[1]["_tool_call_uids"] == {"same-provider-id": uids}
        assert [r.get("_tool_call_uid") for r in got if r.get("effect_disposition") == "unknown"] == expected
        assert canonicalize_replay_history(got, now=100) == got
        for _ in range(2):
            loaded = db.get_messages_as_conversation("s", include_row_ids=True, repair_alternation=True)
            assert loaded[1]["tool_calls"] == original_calls
            assert loaded[1]["_tool_call_uids"] == {"same-provider-id": uids}
            assert [r.get("_tool_call_uid") for r in loaded if r.get("effect_disposition") == "unknown"] == expected
            if has_real_result:
                assert [r for r in loaded if r.get("content") == "real second receipt"] == [raw[-1]]
        assert raw == frozen
        assert list(db._conn.iterdump()) == dump


def test_legacy_sibling_receipt_does_not_prove_distinct_attempts_completed():
    history = [
        {"role": "assistant", "tool_calls": [call("first"), call("second")]},
        {"role": "tool", "tool_call_id": "same-provider-id", "content": "unattributed receipt"},
    ]
    got = canonicalize_replay_history(history, now=100)
    notices = [r for r in got if r.get("effect_disposition") == "unknown"]
    assert len(notices) == 2
    assert all("_tool_call_uid" not in r for r in notices)
    from agent.agent_runtime_helpers import repair_message_sequence

    repaired = copy.deepcopy(got)
    repair_message_sequence(None, repaired)
    assert repaired == got
    assert canonicalize_replay_history(got, now=100) == got


def test_same_response_shared_uid_retains_shared_occurrence_semantics():
    uid = "a" * 32
    history = [
        {"role": "assistant", "tool_calls": [call("first"), call("second")],
         "_tool_call_uids": {"same-provider-id": uid}},
        {"role": "tool", "tool_call_id": "same-provider-id", "_tool_call_uid": uid, "content": "receipt"},
    ]
    assert canonicalize_replay_history(history, now=100) == history


@pytest.mark.parametrize("second_uid", ["b" * 32, None])
@pytest.mark.parametrize("first_already_answered", [False, True])
def test_unattributed_receipt_never_consumes_a_stored_occurrence(second_uid, first_already_answered):
    from agent.agent_runtime_helpers import repair_message_sequence

    first_uid = "a" * 32
    history = [
        {"role": "assistant", "tool_calls": [call("first"), call("second")],
         "_tool_call_uids": {"same-provider-id": [first_uid, second_uid]}},
    ]
    if first_already_answered:
        history.append({"role": "tool", "tool_call_id": "same-provider-id",
                        "_tool_call_uid": first_uid, "content": "identified first receipt"})
    history.append({"role": "tool", "tool_call_id": "same-provider-id",
                    "content": "unattributed receipt"})
    frozen = copy.deepcopy(history)
    canonical = canonicalize_replay_history(history, now=100)
    expected_uids = [second_uid] if first_already_answered else [first_uid, second_uid]
    assert [r.get("_tool_call_uid") for r in canonical
            if r.get("effect_disposition") == "unknown"] == expected_uids
    repaired = copy.deepcopy(canonical)
    repair_message_sequence(None, repaired)
    assert repaired == canonical
    repair_message_sequence(None, repaired)
    assert repaired == canonical
    assert canonicalize_replay_history(repaired, now=100) == canonical
    assert history == frozen
