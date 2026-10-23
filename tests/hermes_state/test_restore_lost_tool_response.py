"""Restore-boundary regression for a committed write whose response was lost.

The message shape matches the isolated native CLI SIGKILL proof. This portable
unit fixture uses only synthetic input and never claims a successful receipt.
"""
import json

from hermes_state import SessionDB


def test_unanswered_write_keeps_unknown_effect_and_retry_key(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("lost-response", "cli")
        arguments = {"idempotency_key": "synthetic-lost-response-A", "date": "2026-10-20",
                     "student_ids": ["ada"], "notes": "Ada practised fractions.",
                     "notify_telegram": False}
        call = {"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1",
                "type": "function", "function": {"name": "mcp__records__create_activity_log",
                "arguments": json.dumps(arguments)}}
        db.append_messages_batch("lost-response", [
            {"role": "user", "content": "Synthetic: record fractions and then planting beans."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ])
        raw = db.get_messages_as_conversation("lost-response")
        assert [m["role"] for m in raw] == ["user", "assistant"]
        projected = db.get_messages_as_conversation("lost-response", repair_alternation=True)
        assert [m["role"] for m in projected] == ["user", "assistant", "tool"]
        recovered = projected[1]["tool_calls"][0]
        assert recovered["id"] == "call_1"
        assert json.loads(recovered["function"]["arguments"]) == arguments
        notice = projected[2]
        assert notice["tool_call_id"] == "call_1"
        assert notice["effect_disposition"] == "unknown"
        assert "effect is UNKNOWN" in notice["content"]
        assert "activity_ids" not in notice["content"]
        # Model repair is a projection, not a fabricated durable receipt.
        assert db.get_messages_as_conversation("lost-response") == raw
    finally:
        db.close()
