"""Semantic memory with vector search using embeddings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False


class VectorMemory:
    """Embedded vector memory for semantic search.

    Uses sqlite-vec if available, otherwise falls back to in-memory numpy search.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_dim: int = 1536,
        embed_func: Optional[callable] = None,
    ) -> None:
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        self.embedding_dim = embedding_dim
        self.embed_func = embed_func or self._default_embed
        self._use_sqlite_vec = False
        self._init_db()

    def _default_embed(self, text: str) -> list[float]:
        """Default hash-based embedding (deterministic, no API needed)."""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert to pseudo-embedding
        digest = hash_obj.digest()
        if NUMPY_AVAILABLE:
            vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            vec = np.resize(vec, self.embedding_dim)
            vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
            return vec.tolist()
        else:
            # Pure Python fallback
            vec = [float(b) / 255.0 for b in digest]
            vec = (vec * (self.embedding_dim // len(vec) + 1))[:self.embedding_dim]
            norm = sum(v * v for v in vec) ** 0.5
            return [v / norm for v in vec] if norm > 0 else vec

    def _init_db(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB
                )
                """
            )

            if SQLITE_VEC_AVAILABLE:
                try:
                    self.connection.enable_load_extension(True)
                    sqlite_vec.load(self.connection)
                    self.connection.enable_load_extension(False)
                    self.connection.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec
                        USING vec0(embedding float[{self.embedding_dim}])
                        """
                    )
                    self._use_sqlite_vec = True
                except Exception:
                    self._use_sqlite_vec = False

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Add content with its embedding to the vector store."""
        embedding = self.embed_func(content)
        embedding_bytes = json.dumps(embedding).encode()

        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO vectors (content, metadata, embedding) VALUES (?, ?, ?)",
                (content, json.dumps(metadata or {}), embedding_bytes),
            )
            row_id = cursor.lastrowid

            if self._use_sqlite_vec:
                self.connection.execute(
                    "INSERT INTO vectors_vec(rowid, embedding) VALUES (?, ?)",
                    (row_id, json.dumps(embedding)),
                )

        return row_id

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for similar content using vector similarity."""
        query_embedding = self.embed_func(query)

        if self._use_sqlite_vec:
            return self._search_sqlite_vec(query_embedding, limit, threshold)
        else:
            return self._search_numpy(query_embedding, limit, threshold)

    def _search_sqlite_vec(
        self,
        query_embedding: list[float],
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search using sqlite-vec extension."""
        rows = self.connection.execute(
            """
            SELECT v.id, v.content, v.metadata, v.embedding,
                   vec_distance_cosine(v_vec.embedding, ?) as distance
            FROM vectors v
            JOIN vectors_vec v_vec ON v.id = v_vec.rowid
            WHERE vec_distance_cosine(v_vec.embedding, ?) < ?
            ORDER BY distance
            LIMIT ?
            """,
            (json.dumps(query_embedding), json.dumps(query_embedding), 1.0 - threshold, limit),
        ).fetchall()

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "score": 1.0 - row["distance"],
            }
            for row in rows
        ]

    def _search_numpy(
        self,
        query_embedding: list[float],
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search using in-memory numpy cosine similarity."""
        rows = self.connection.execute(
            "SELECT id, content, metadata, embedding FROM vectors"
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        results = []
        for row in rows:
            emb = json.loads(row["embedding"].decode())
            vec = np.array(emb, dtype=np.float32)
            vec_norm = np.linalg.norm(vec)

            if query_norm > 0 and vec_norm > 0:
                score = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
            else:
                score = 0.0

            if score >= threshold:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()


class SemanticMemory:
    """High-level semantic memory combining keyword and vector search."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embed_func: Optional[callable] = None,
    ) -> None:
        from .memory import SQLiteMemory
        self.keyword_memory = SQLiteMemory(db_path)
        self.vector_memory = VectorMemory(db_path, embed_func=embed_func)

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a message to both keyword and vector memory."""
        msg_id = self.keyword_memory.add_message(role, content)
        meta = {"role": role, "message_id": msg_id}
        if metadata:
            meta.update(metadata)
        self.vector_memory.add(content, meta)
        return msg_id

    def search(
        self,
        query: str,
        limit: int = 10,
        semantic: bool = True,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search memory using keyword and/or semantic search."""
        keyword_results = self.keyword_memory.search(query, limit=limit)

        if not semantic:
            return keyword_results

        vector_results = self.vector_memory.search(query, limit=limit, threshold=threshold)

        # Merge and deduplicate by content
        seen = set()
        merged = []

        for r in keyword_results + vector_results:
            key = (r.get("role"), r["content"][:100])
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged[:limit]

    def close(self) -> None:
        """Close both memory stores."""
        self.keyword_memory.close()
        self.vector_memory.close()


__all__ = ["VectorMemory", "SemanticMemory", "SQLITE_VEC_AVAILABLE", "NUMPY_AVAILABLE"]