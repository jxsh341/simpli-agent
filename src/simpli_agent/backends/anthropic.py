"""Anthropic API backend implementation."""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

from .base import Backend


class AnthropicBackend(Backend):
    """Anthropic API backend with tool calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )

    def _convert_tools(self, schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool schemas to Anthropic format."""
        tools = []
        for schema in schemas:
            if schema.get("type") == "function":
                func = schema["function"]
                tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func["parameters"],
                })
        return tools

    def _convert_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic handles system separately
                continue
            elif role == "tool":
                # Tool results go as user messages with tool_result blocks
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": str(content),
                    }],
                })
            elif role == "assistant":
                # Assistant messages may have tool calls
                if "tool_calls" in msg:
                    blocks = []
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "input": tc["arguments"],
                        })
                    converted.append({"role": "assistant", "content": blocks})
                else:
                    converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": role, "content": content})
        return converted

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

        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(schemas)

        while True:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_msg,
                messages=anth_messages,
                tools=anth_tools if anth_tools else None,
            )

            # Convert response to our format
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            anth_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                return content

            for tool_call in tool_calls:
                func_name = tool_call["name"]
                if func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                result = tools[func_name](**tool_call["arguments"])

                anth_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": str(result),
                    }],
                })

    def run_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(schemas)

        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=anth_messages,
            tools=anth_tools if anth_tools else None,
        )

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
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
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(schemas)

        while True:
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_msg,
                messages=anth_messages,
                tools=anth_tools if anth_tools else None,
            )

            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            anth_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                return content

            for tool_call in tool_calls:
                func_name = tool_call["name"]
                if func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                result = tools[func_name](**tool_call["arguments"])

                anth_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": str(result),
                    }],
                })

    async def run_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(schemas)

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=anth_messages,
            tools=anth_tools if anth_tools else None,
        )

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        result = {"content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
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
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(schemas)

        stream = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=anth_messages,
            tools=anth_tools if anth_tools else None,
            stream=True,
        )

        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text