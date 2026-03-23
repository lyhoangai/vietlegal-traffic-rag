"""SQLite-backed short-term memory for chat sessions."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "chat_memory.db"


def _db_path() -> str:
    return os.getenv("MEMORY_DB_PATH", str(DEFAULT_DB_PATH))


def init_db() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
            ON chat_messages (session_id, created_at)
            """
        )


def save_message(session_id: str, role: str, content: str) -> None:
    if not session_id or not content.strip():
        return
    init_db()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content.strip(), time.time()),
        )


def remember_turn(session_id: str, query: str, answer: str) -> None:
    if not session_id:
        return
    save_message(session_id, "user", query)
    save_message(session_id, "assistant", answer)


def get_session_history(session_id: str, limit: int = 12) -> list[dict]:
    if not session_id:
        return []
    init_db()
    with sqlite3.connect(_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    rows.reverse()
    return [
        {"role": role, "content": content, "created_at": created_at}
        for role, content, created_at in rows
    ]


def get_session_messages(session_id: str, limit: int = 12) -> list[dict]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in get_session_history(session_id, limit=limit)
    ]


def get_recent_sessions(limit: int = 20) -> list[dict]:
    init_db()
    with sqlite3.connect(_db_path()) as conn:
        rows = conn.execute(
            """
            WITH session_updates AS (
                SELECT session_id, MAX(created_at) AS updated_at
                FROM chat_messages
                GROUP BY session_id
                ORDER BY updated_at DESC
                LIMIT ?
            )
            SELECT
                session_updates.session_id,
                session_updates.updated_at,
                COALESCE(
                    (
                        SELECT content
                        FROM chat_messages
                        WHERE session_id = session_updates.session_id
                          AND role = 'user'
                        ORDER BY created_at ASC, id ASC
                        LIMIT 1
                    ),
                    (
                        SELECT content
                        FROM chat_messages
                        WHERE session_id = session_updates.session_id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ),
                    ''
                ) AS title,
                (
                    SELECT content
                    FROM chat_messages
                    WHERE session_id = session_updates.session_id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                ) AS last_message,
                (
                    SELECT COUNT(*)
                    FROM chat_messages
                    WHERE session_id = session_updates.session_id
                ) AS message_count
            FROM session_updates
            ORDER BY session_updates.updated_at DESC
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "session_id": session_id,
            "title": title or "Cuoc tro chuyen",
            "last_message": last_message or "",
            "updated_at": updated_at,
            "message_count": message_count,
        }
        for session_id, updated_at, title, last_message, message_count in rows
    ]


def delete_session_messages(session_id: str) -> None:
    if not session_id:
        return
    init_db()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))


def clear_all_messages() -> None:
    init_db()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("DELETE FROM chat_messages")
