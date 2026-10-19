"""Private, bounded recovery storage for Telegram messages not admitted to a session.

This queue is deliberately write/read only: it never creates sessions, dispatches an
agent turn, or sends a Telegram reply.  Callers must establish authorization before
calling :meth:`TelegramRecoveryQueue.enqueue`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

# Fixed narrow recovery window.  These are intentionally not gateway settings: enabling
# the queue must not widen normal Telegram admission or require a config change.
TTL_SECONDS = 48 * 60 * 60
MAX_ITEMS = 128
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_ITEM_BYTES = 64 * 1024


class TelegramRecoveryQueue:
    """A per-profile SQLite queue for already-authorized, non-admitted messages."""

    def __init__(self, profile_home: Path | str) -> None:
        self.profile_home = Path(profile_home)
        self.directory = self.profile_home / "state" / "telegram-recovery"
        self.path = self.directory / "queue.sqlite3"

    def _prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # The queue can contain message content, so do not inherit a permissive umask.
        os.chmod(self.directory, 0o700)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS recovery_messages (
                    platform_message_key TEXT PRIMARY KEY,
                    received_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS recovery_messages_received_at ON recovery_messages(received_at)")
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        self._prepare()
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _prune(conn: sqlite3.Connection, now: float) -> None:
        conn.execute("DELETE FROM recovery_messages WHERE received_at < ?", (now - TTL_SECONDS,))

    @staticmethod
    def _bounded_paths(paths: Iterable[object]) -> list[str]:
        # References only.  This module never reads, downloads, or copies media.
        result: list[str] = []
        for path in paths:
            if not isinstance(path, str) or not path or len(path) > 4096:
                continue
            candidate = Path(path)
            # A recovery row may point at an existing local cache item, but must
            # never turn a Telegram file identifier/URL into a new download.
            if candidate.is_absolute() and candidate.is_file():
                result.append(str(candidate))
        return result

    def enqueue(
        self,
        *,
        chat_id: str,
        sender_id: str,
        message_id: str,
        message_type: str,
        text: str = "",
        platform_update_id: int | None = None,
        media_paths: Iterable[object] = (),
        now: float | None = None,
    ) -> bool:
        """Atomically retain one message, returning False for duplicate or over-size input.

        ``chat_id`` and ``sender_id`` are identifiers, not labels.  Deliberately absent:
        chat title, username, display name, and any other raw chat label.
        """
        received_at = time.time() if now is None else float(now)
        chat_id, sender_id, message_id = str(chat_id), str(sender_id), str(message_id)
        if not chat_id or not sender_id or not message_id:
            return False
        payload: dict[str, Any] = {
            "platform": "telegram",
            "chat_id": chat_id,
            "sender_id": sender_id,
            "message_id": message_id,
            "message_type": str(message_type),
            "received_at": received_at,
            "text": str(text or ""),
            "media_paths": self._bounded_paths(media_paths),
        }
        if platform_update_id is not None:
            payload["platform_update_id"] = int(platform_update_id)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        size = len(encoded.encode("utf-8"))
        if size > MAX_ITEM_BYTES:
            return False
        # Telegram message IDs are only unique inside a chat, so key both values.
        key = f"{chat_id}:{message_id}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, received_at)
                if conn.execute("SELECT 1 FROM recovery_messages WHERE platform_message_key = ?", (key,)).fetchone():
                    conn.execute("ROLLBACK")
                    return False
                while True:
                    count, total = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM recovery_messages"
                    ).fetchone()
                    if count < MAX_ITEMS and total + size <= MAX_TOTAL_BYTES:
                        break
                    oldest = conn.execute(
                        "SELECT platform_message_key FROM recovery_messages ORDER BY received_at ASC, rowid ASC LIMIT 1"
                    ).fetchone()
                    if oldest is None:
                        conn.execute("ROLLBACK")
                        return False
                    conn.execute("DELETE FROM recovery_messages WHERE platform_message_key = ?", (oldest[0],))
                conn.execute(
                    "INSERT INTO recovery_messages(platform_message_key, received_at, payload_json, payload_bytes) VALUES (?, ?, ?, ?)",
                    (key, received_at, encoded, size),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def read(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Return retained entries after TTL pruning; this never dispatches or acknowledges them."""
        current = time.time() if now is None else float(now)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, current)
                rows = conn.execute(
                    "SELECT payload_json FROM recovery_messages ORDER BY received_at ASC, rowid ASC"
                ).fetchall()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return [json.loads(row[0]) for row in rows]
