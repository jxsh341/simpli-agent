"""Native Hermes backend integration placeholder."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Mapping, Sequence, Union

from ..types import ToolProgress
from .base import Backend


class HermesBackend(Backend):
    """Deterministic embedded backend used until a native Hermes kernel is attached."""

    def run(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        tool_summary = f" with {len(schemas)} tool(s)" if schemas else ""
        return f"[Simpli-Agent Response using {model}{tool_summary}] Completed task: {prompt}"

    def run_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        tool_summary = f" with {len(schemas)} tool(s)" if schemas else ""
        return {"content": f"[Simpli-Agent Response using {model}{tool_summary}] Completed task"}

    async def run_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        return self.run(model=model, prompt=prompt, tools=tools, schemas=schemas, history=history)

    async def run_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.run_with_tools(model=model, messages=messages, tools=tools, schemas=schemas)

    async def stream_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ):
        response = self.run(model=model, prompt=prompt, tools=tools, schemas=schemas, history=history)
        for chunk in response.split():
            yield chunk + " "

    def stream_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> Iterator[Union[str, ToolProgress]]:
        """Stream response with tool progress (Hermes placeholder)."""
        tool_summary = f" with {len(schemas)} tool(s)" if schemas else ""
        response = f"[Simpli-Agent Response using {model}{tool_summary}] Completed task"
        for chunk in response.split():
            yield chunk + " "

    async def stream_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[Union[str, ToolProgress]]:
        """Async stream response with tool progress (Hermes placeholder)."""
        tool_summary = f" with {len(schemas)} tool(s)" if schemas else ""
        response = f"[Simpli-Agent Response using {model}{tool_summary}] Completed task"
        for chunk in response.split():
            yield chunk + " "
