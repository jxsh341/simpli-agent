"""Embedded SQLite memory provider with FTS5-backed search when available."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class SQLiteMemory:
    """Persist agent messages in SQLite and expose simple full-text search."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        self.fts_enabled = self._init_db()

    def _init_db(self) -> bool:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            try:
                self.connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS history_fts
                    USING fts5(role, content, content='history', content_rowid='id')
                    """
                )
            except sqlite3.OperationalError:
                return False
        return True

    def add_message(self, role: str, content: str) -> int:
        """Store a message and return its row id."""
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO history (role, content) VALUES (?, ?)", (role, content)
            )
            row_id = int(cursor.lastrowid)
            if self.fts_enabled:
                self.connection.execute(
                    "INSERT INTO history_fts(rowid, role, content) VALUES (?, ?, ?)",
                    (row_id, role, content),
                )
        return row_id

    def list_messages(self, limit: Optional[int] = None) -> list[dict[str, str | int]]:
        """Return persisted messages in insertion order."""
        query = "SELECT id, role, content FROM history ORDER BY id"
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = self.connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[dict[str, str | int]]:
        """Search message history using FTS5, falling back to LIKE."""
        if self.fts_enabled:
            rows = self.connection.execute(
                """
                SELECT h.id, h.role, h.content
                FROM history_fts f
                JOIN history h ON h.id = f.rowid
                WHERE history_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT id, role, content FROM history
                WHERE content LIKE ? OR role LIKE ?
                ORDER BY id
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.connection.close()
