"""Portable interrupted-block restore regressions using only synthetic state."""
import copy
import json
import sqlite3
import pytest
from hermes_state import SessionDB
from agent.agent_runtime_helpers import repair_message_sequence
from agent.transports.chat_completions import ChatCompletionsTransport


def call(cid, name):
    return {"id": cid, "call_id": cid, "response_item_id": "fc_" + cid,
            "type": "function", "function": {"name": name, "arguments": json.dumps({
                "idempotency_key": "synthetic-" + cid, "source_message_ids": ["synthetic-origin-101"],
                "notify_telegram": False})}}


def assistant(calls):
    return {"role": "assistant", "content": "", "tool_calls": calls}


def result(cid, text="real synthetic read result"):
    return {"role": "tool", "tool_call_id": cid, "content": text}


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("s", "cli")
    yield d
    d.close()


def seed(db, tail):
    db.append_messages_batch("s", [{"role": "user", "content": "Synthetic capture"}] + tail)


def snap(db):
    return list(db._conn.iterdump())


def restore(db):
    before = snap(db)
    raw = db.get_messages_as_conversation("s", include_row_ids=True)
    from agent.replay_cleanup import canonicalize_replay_history
    frozen_raw = copy.deepcopy(raw)
    canonical = canonicalize_replay_history(raw)
    assert raw == frozen_raw, "canonicalization must not mutate its input"
    assert canonicalize_replay_history(canonical) == canonical
    loaded = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)
    assert snap(db) == before, "restore must not mutate any durable SQLite table"
    assert db.get_messages_as_conversation("s", include_row_ids=True) == raw
    assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    frozen = copy.deepcopy(loaded)
    assert repair_message_sequence(None, loaded) == 0, "restore must already be alternation-clean"
    assert loaded == frozen
    return raw, loaded


def wire(messages):
    return json.dumps(ChatCompletionsTransport().convert_messages(copy.deepcopy(messages)), sort_keys=True)


def test_completed_history_cache_prefix_unchanged(db):
    seed(db, [assistant([call("r", "read_file"), call("w", "write_file")]),
              result("r"), result("w", '{"bytes_written":1}'),
              {"role": "assistant", "content": "done"}])
    raw, loaded = restore(db)
    assert loaded == raw
    assert wire(loaded) == wire(raw)


def test_completed_prefix_unchanged_when_write_tail_recovered(db):
    seed(db, [assistant([call("r", "read_file")]), result("r"),
              {"role": "assistant", "content": "done"},
              {"role": "user", "content": "next"}, assistant([call("w", "write_file")])])
    raw, loaded = restore(db)
    assert loaded[:-1] == raw
    assert wire(loaded[:4]) == wire(raw[:4])
    assert loaded[-1]["effect_disposition"] == "unknown"


@pytest.mark.parametrize("name", ["write_file", "mcp__records__create_activity_log", "unknown_plugin"])
def test_side_effect_or_unknown_orphan_preserves_exact_call_and_provenance(db, name):
    seed(db, [assistant([call("w", name)])])
    raw, loaded = restore(db)
    assert loaded[:2] == raw
    assert loaded[1]["tool_calls"] == raw[1]["tool_calls"]
    assert loaded[1]["message_uid"] == raw[1]["message_uid"]
    assert loaded[1]["_tool_call_uids"] == raw[1]["_tool_call_uids"]
    notice = loaded[-1]
    assert notice["tool_call_id"] == "w"
    assert notice["effect_disposition"] == "unknown"
    assert "UNKNOWN" in notice["content"] and "Inspect" in notice["content"]
    assert "activity_ids" not in notice["content"] and "bytes_written" not in notice["content"]
    assert "_row_id" not in notice, "projection-only notice cannot invent durable row identity"


def test_recovered_notice_has_original_tool_uid(db):
    seed(db, [assistant([call("w", "write_file")])])
    raw, loaded = restore(db)
    assert loaded[-1].get("_tool_call_uid") == raw[1]["_tool_call_uids"]["w"]


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("answered_id", ["read-call", "fc_read", "read-call|fc_read"])
def test_provider_alias_partial_block_preserves_write_identity(db, queued, answered_id):
    read = call("read-id", "read_file")
    read.update(call_id="read-call", response_item_id="fc_read")
    write = call("write-id", "write_file")
    write.update(call_id="write-call", response_item_id="fc_write")
    tail = [assistant([read, write]), result(answered_id)]
    if queued:
        tail.append({"role": "user", "content": "queued"})
    seed(db, tail)
    raw, loaded = restore(db)
    assert loaded[1] == raw[1]
    original = raw[1]["tool_calls"][1]
    recovered = loaded[1]["tool_calls"][1]
    assert recovered == original
    assert recovered["function"]["arguments"] == write["function"]["arguments"]
    assert json.loads(recovered["function"]["arguments"])["idempotency_key"] == "synthetic-write-id"
    assert (recovered["id"], recovered["call_id"], recovered["response_item_id"]) == (
        "write-id", "write-call", "fc_write")
    assert loaded[2] == raw[2]
    assert loaded[3]["tool_call_id"] == "write-call"
    assert loaded[3]["_tool_call_uid"] == raw[1]["_tool_call_uids"]["write-call"]
    assert loaded[3]["effect_disposition"] == "unknown"
    if queued:
        assert loaded[4] == raw[3]


def test_read_only_orphan_may_be_discarded(db):
    seed(db, [assistant([call("r", "read_file")])])
    raw, loaded = restore(db)
    assert loaded == raw[:1]


def test_mixed_unanswered_parallel_calls_get_truthful_per_call_status(db):
    seed(db, [assistant([call("r", "read_file"), call("w", "write_file")])])
    raw, loaded = restore(db)
    assert loaded[:2] == raw
    assert [m["tool_call_id"] for m in loaded[2:]] == ["r", "w"]
    assert [m["effect_disposition"] for m in loaded[2:]] == ["none", "unknown"]


@pytest.mark.parametrize("answered,missing", [("read_file", "write_file"), ("write_file", "read_file")])
def test_partially_answered_parallel_block_retains_unanswered_call(db, answered, missing):
    seed(db, [assistant([call("a", answered), call("b", missing)]), result("a")])
    raw, loaded = restore(db)
    assert loaded[1]["tool_calls"] == raw[1]["tool_calls"], "cannot erase an attempted call in a partial parallel block"
    assert loaded[2] == raw[2], "answered result and its identity must remain exact"
    notice = next(m for m in loaded[3:] if m.get("tool_call_id") == "b")
    assert notice["effect_disposition"] == ("unknown" if missing == "write_file" else "none")


def test_side_effect_orphan_followed_by_queued_user_is_not_erased(db):
    seed(db, [assistant([call("w", "write_file")]), {"role": "user", "content": "queued followup"}])
    raw, loaded = restore(db)
    owners = [m for m in loaded if m.get("tool_calls")]
    assert len(owners) == 1, "queued input must not erase the attempted write and retry key"
    assert owners[0]["tool_calls"] == raw[1]["tool_calls"]
    assert any(m.get("tool_call_id") == "w" and m.get("effect_disposition") == "unknown" for m in loaded)


def test_interrupted_write_result_is_unknown_not_success(db):
    seed(db, [assistant([call("w", "write_file")]), result("w", '[Command interrupted]')])
    raw, loaded = restore(db)
    assert loaded[1] == raw[1]
    assert loaded[2]["message_uid"] == raw[2]["message_uid"]
    assert loaded[2]["_tool_call_uid"] == raw[2]["_tool_call_uid"]
    assert loaded[2]["effect_disposition"] == "unknown"
    assert "UNKNOWN" in loaded[2]["content"]


@pytest.mark.parametrize("answered,missing", [
    ("read_file", "write_file"), ("write_file", "read_file"),
    ("read_file", "read_file"), ("write_file", "unknown_plugin"),
])
@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_block_notice_precedes_queued_user_and_pairs_uid(db, answered, missing, interrupted):
    seed(db, [assistant([call("a", answered), call("b", missing)]),
              result("a", "[Command interrupted]" if interrupted else "real result"),
              {"role": "user", "content": "queued followup"}])
    raw, loaded = restore(db)
    # An entirely read-only interrupted block remains intentionally discardable.
    if interrupted and answered == missing == "read_file":
        assert not any(m.get("tool_calls") for m in loaded)
        return
    assert loaded[1] == raw[1]
    if not interrupted:
        assert loaded[2] == raw[2]
    assert loaded[3]["tool_call_id"] == "b"
    assert loaded[3]["_tool_call_uid"] == raw[1]["_tool_call_uids"]["b"]
    assert loaded[3]["effect_disposition"] == ("none" if missing == "read_file" else "unknown")
    assert loaded[4] == raw[3]
    from agent.replay_cleanup import canonicalize_replay_history
    frozen = copy.deepcopy(loaded)
    assert canonicalize_replay_history(loaded) == frozen
    assert loaded == frozen


def test_user_pair_alternation_repair_preserves_both_sources(db):
    seed(db, [{"role": "user", "content": "second"}, {"role": "assistant", "content": "done"}])
    raw, loaded = restore(db)
    assert [m["role"] for m in loaded] == ["user", "assistant"]
    assert loaded[0]["message_uid"] == raw[0]["message_uid"]
    assert raw[1]["message_uid"] in loaded[0]["_absorbed_message_uids"]
    assert "Synthetic capture" in loaded[0]["content"] and "second" in loaded[0]["content"]
