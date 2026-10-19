import os
import stat

from gateway.telegram_recovery_queue import (
    MAX_ITEM_BYTES,
    MAX_ITEMS,
    TelegramRecoveryQueue,
    TTL_SECONDS,
)


def _enqueue(queue, *, chat="chat", sender="sender", message="message", now=1000.0, text="body"):
    return queue.enqueue(
        chat_id=chat,
        sender_id=sender,
        message_id=message,
        message_type="text",
        text=text,
        now=now,
    )


def test_queue_is_profile_local_private_and_idempotent(tmp_path):
    home = tmp_path / "profile"
    queue = TelegramRecoveryQueue(home)

    assert _enqueue(queue) is True
    assert _enqueue(queue) is False
    entries = queue.read(now=1001.0)

    assert len(entries) == 1
    assert entries[0]["media_paths"] == []
    assert "chat_title" not in entries[0]
    assert "user_name" not in entries[0]
    assert queue.path.parent == home / "state" / "telegram-recovery"
    assert stat.S_IMODE(os.stat(queue.directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(queue.path).st_mode) == 0o600


def test_queue_prunes_expired_entries_on_enqueue_and_read(tmp_path):
    queue = TelegramRecoveryQueue(tmp_path / "profile")
    assert _enqueue(queue, message="old", now=1000.0)
    assert _enqueue(queue, message="fresh", now=1000.0 + TTL_SECONDS + 1)

    entries = queue.read(now=1000.0 + TTL_SECONDS + 1)
    assert [entry["message_id"] for entry in entries] == ["fresh"]


def test_queue_evicts_oldest_entry_to_stay_bounded(tmp_path, monkeypatch):
    queue = TelegramRecoveryQueue(tmp_path / "profile")
    monkeypatch.setattr("gateway.telegram_recovery_queue.MAX_ITEMS", 2)
    monkeypatch.setattr("gateway.telegram_recovery_queue.MAX_TOTAL_BYTES", MAX_ITEM_BYTES * 3)

    assert _enqueue(queue, message="one", now=1000.0)
    assert _enqueue(queue, message="two", now=1001.0)
    assert _enqueue(queue, message="three", now=1002.0)

    assert [entry["message_id"] for entry in queue.read(now=1002.0)] == ["two", "three"]


def test_queue_records_only_existing_local_media_references(tmp_path):
    queue = TelegramRecoveryQueue(tmp_path / "profile")
    cached = tmp_path / "cached.bin"
    cached.write_bytes(b"cache")

    assert queue.enqueue(
        chat_id="chat",
        sender_id="sender",
        message_id="message",
        message_type="document",
        media_paths=[str(cached), "https://example.invalid/file", str(tmp_path / "missing.bin")],
        now=1000.0,
    )
    assert queue.read(now=1000.0)[0]["media_paths"] == [str(cached)]


def test_queue_rejects_oversized_payload_without_creating_a_row(tmp_path):
    queue = TelegramRecoveryQueue(tmp_path / "profile")
    assert _enqueue(queue, text="x" * MAX_ITEM_BYTES) is False
    assert queue.read(now=1000.0) == []
