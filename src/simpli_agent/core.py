"""High-level Agent facade tying tools, memory, and backends together."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional

from .backends import Backend, HermesBackend
from .decorators import generate_tool_schema
from .memory import SQLiteMemory

_BACKENDS: dict[str, type[Backend]] = {"hermes": HermesBackend}


class Agent:
    """Programmatic agent runtime with embedded SQLite state and tool schemas."""

    def __init__(
        self,
        model: str = "gpt-4o",
        backend: str | Backend = "hermes",
        db_path: str | Path = ":memory:",
    ) -> None:
        self.model = model
        self.backend_type = backend if isinstance(backend, str) else backend.__class__.__name__
        self.tools: dict[str, Callable[..., Any]] = {}
        self.schemas: list[dict[str, Any]] = []
        self.memory = SQLiteMemory(db_path)
        self.backend = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str | Backend) -> Backend:
        if isinstance(backend, Backend):
            return backend
        try:
            return _BACKENDS[backend]()
        except KeyError as exc:
            available = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"Unknown backend '{backend}'. Available backends: {available}") from exc

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register ``func`` as an agent tool and return it unchanged."""
        schema = generate_tool_schema(func)
        self.schemas.append(schema)
        self.tools[func.__name__] = func
        return func

    def run(self, prompt: str) -> str:
        """Execute a task, persist user/assistant messages, and return the response."""
        self.memory.add_message("user", prompt)
        response = self.backend.run(
            model=self.model,
            prompt=prompt,
            tools=self.tools,
            schemas=self.schemas,
            history=self.memory.list_messages(),
        )
        self.memory.add_message("assistant", response)
        return response

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream a response as whitespace-delimited chunks."""
        response = self.run(prompt)
        for chunk in response.split():
            yield chunk

    def history(self, limit: Optional[int] = None) -> list[dict[str, str | int]]:
        """Return persisted conversation messages."""
        return self.memory.list_messages(limit=limit)

    def search_memory(self, query: str, limit: int = 10) -> list[dict[str, str | int]]:
        """Search persisted conversation messages."""
        return self.memory.search(query, limit=limit)

    def close(self) -> None:
        """Close resources owned by the agent."""
        self.memory.close()
