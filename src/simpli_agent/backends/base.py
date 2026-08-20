"""Backend abstraction for agent execution providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Callable, Union


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

    def run_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run with pre-built messages, returning raw response with tool_calls.
        
        Optional: backends that support tool calling loops should override this.
        """
        raise NotImplementedError("run_with_tools not implemented")

    async def run_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        """Async version of run. Default implementation delegates to sync."""
        return self.run(
            model=model,
            prompt=prompt,
            tools=tools,
            schemas=schemas,
            history=history,
        )

    async def run_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Async version of run_with_tools."""
        raise NotImplementedError("run_with_tools_async not implemented")

    def stream_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> Iterator[Union[str, Any]]:
        """Stream response with tool progress. Optional override."""
        raise NotImplementedError("stream_with_tools not implemented")

    async def stream_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[Union[str, Any]]:
        """Async stream response with tool progress. Optional override."""
        raise NotImplementedError("stream_with_tools_async not implemented")

    async def stream_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream response as async chunks. Default yields full response."""
        response = await self.run_async(
            model=model,
            prompt=prompt,
            tools=tools,
            schemas=schemas,
            history=history,
        )
        for chunk in response.split():
            yield chunk + " "
