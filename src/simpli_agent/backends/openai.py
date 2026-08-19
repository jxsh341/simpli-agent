"""OpenAI API backend implementation."""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

from .base import Backend


class OpenAIBackend(Backend):
    """OpenAI API backend with tool calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
        )

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

        while True:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=schemas if schemas else None,
                tool_choice="auto" if schemas else None,
            )

            message = response.choices[0].message
            messages.append(message.model_dump())

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                if func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                import json
                args = json.loads(tool_call.function.arguments)
                result = tools[func_name](**args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
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
        """Run with pre-built messages, returning raw response with tool_calls."""
        response = self.client.chat.completions.create(
            model=model,
            messages=list(messages),
            tools=schemas if schemas else None,
            tool_choice="auto" if schemas else None,
        )

        message = response.choices[0].message
        result = {"content": message.content or ""}
        if message.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in message.tool_calls
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
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        while True:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=schemas if schemas else None,
                tool_choice="auto" if schemas else None,
            )

            message = response.choices[0].message
            messages.append(message.model_dump())

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                if func_name not in tools:
                    raise ValueError(f"Unknown tool: {func_name}")

                import json
                args = json.loads(tool_call.function.arguments)
                result = tools[func_name](**args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
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
        """Async run with pre-built messages, returning raw response with tool_calls."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = await client.chat.completions.create(
            model=model,
            messages=list(messages),
            tools=schemas if schemas else None,
            tool_choice="auto" if schemas else None,
        )

        message = response.choices[0].message
        result = {"content": message.content or ""}
        if message.tool_calls:
            import json
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in message.tool_calls
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
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas if schemas else None,
            tool_choice="auto" if schemas else None,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content