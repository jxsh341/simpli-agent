"""Backend abstraction for agent execution providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence


class Backend(ABC):
    """Abstract execution backend used by :class:`simpli_agent.Agent`."""

    @abstractmethod
    def run(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        """Execute a prompt and return the final assistant response."""
