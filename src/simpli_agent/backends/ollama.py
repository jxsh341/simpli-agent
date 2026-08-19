"""Ollama API backend implementation for local models."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, Sequence

from .base import Backend


class OllamaBackend(Backend):
    """Ollama API backend for running local models with tool calling support."""

    def __init__(
        self,
        base_url: str | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx package required. Install with: pip install httpx") from exc

        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = httpx.Client(base_url=self.base_url, timeout=60.0)
        self.async_client = None  # lazy init

    def _convert_tools(self, schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool schemas to Ollama format."""
        tools = []
        for schema in schemas:
            if schema.get("type") == "function":
                func = schema["function"]
                tools.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func["parameters"],
                    },
                })
        return tools

    def run(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        ollama_tools = self._convert_tools(schemas)

        while True:
            response = self.client.post("/api/chat", json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools if ollama_tools else None,
                "stream": False,
            }).json()

            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            messages.append(message)

            if not tool_calls:
                return content

            for tool_call in tool_calls:
                func_name = tool_call.get("function", {}).get("name")
                if not func_name or func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                args = tool_call.get("function", {}).get("arguments", {})
                result = tools[func_name](**args)

                messages.append({
                    "role": "tool",
                    "content": str(result),
                })

    def run_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        ollama_tools = self._convert_tools(schemas)

        response = self.client.post("/api/chat", json={
            "model": model,
            "messages": list(messages),
            "tools": ollama_tools if ollama_tools else None,
            "stream": False,
        }).json()

        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", {}),
                }
                for i, tc in enumerate(tool_calls)
            ]
        return result

    async def run_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ) -> str:
        import httpx

        if self.async_client is None:
            self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        ollama_tools = self._convert_tools(schemas)

        while True:
            response = await self.async_client.post("/api/chat", json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools if ollama_tools else None,
                "stream": False,
            })
            response.raise_for_status()
            data = response.json()

            message = data.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            messages.append(message)

            if not tool_calls:
                return content

            for tool_call in tool_calls:
                func_name = tool_call.get("function", {}).get("name")
                if not func_name or func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                args = tool_call.get("function", {}).get("arguments", {})
                result = tools[func_name](**args)

                messages.append({
                    "role": "tool",
                    "content": str(result),
                })

    async def run_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        import httpx

        if self.async_client is None:
            self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        ollama_tools = self._convert_tools(schemas)

        response = await self.async_client.post("/api/chat", json={
            "model": model,
            "messages": list(messages),
            "tools": ollama_tools if ollama_tools else None,
            "stream": False,
        })
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", {}),
                }
                for i, tc in enumerate(tool_calls)
            ]
        return result

    async def stream_async(
        self,
        *,
        model: str,
        prompt: str,
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
        history: Sequence[dict[str, Any]],
    ):
        import httpx

        if self.async_client is None:
            self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        ollama_tools = self._convert_tools(schemas)

        async with self.async_client.stream("POST", "/api/chat", json={
            "model": model,
            "messages": messages,
            "tools": ollama_tools if ollama_tools else None,
            "stream": True,
        }) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]