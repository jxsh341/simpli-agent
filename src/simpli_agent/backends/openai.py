"""OpenAI API backend implementation."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Mapping, Sequence, Union

from ..types import ToolProgress
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

    def stream_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> Iterator[Union[str, ToolProgress]]:
        """Stream response with tool progress (OpenAI streaming with tools)."""
        stream = self.client.chat.completions.create(
            model=model,
            messages=list(messages),
            tools=schemas if schemas else None,
            tool_choice="auto" if schemas else None,
            stream=True,
        )

        tool_calls_buffer: dict[int, dict] = {}
        content_buffer = ""

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_buffer += delta.content
                yield delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    if index not in tool_calls_buffer:
                        tool_calls_buffer[index] = {"id": "", "name": "", "arguments": ""}
                    
                    if tc.id:
                        tool_calls_buffer[index]["id"] = tc.id
                    if tc.function.name:
                        tool_calls_buffer[index]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_buffer[index]["arguments"] += tc.function.arguments

        if tool_calls_buffer:
            for index, tc_data in tool_calls_buffer.items():
                func_name = tc_data["name"]
                if func_name not in tools:
                    yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=f"Unknown tool: {func_name}", is_final=True, error=f"Unknown tool: {func_name}")
                    continue

                yield ToolProgress(call_id=tc_data["id"], name=func_name, progress="Starting...", is_final=False)

                try:
                    args = json.loads(tc_data["arguments"])
                    func = tools[func_name]
                    result = func(**args)
                    
                    if inspect.isgenerator(result) or (hasattr(result, '__iter__') and not isinstance(result, (str, dict, list, tuple, bytes))):
                        final_result = None
                        for item in result:
                            final_result = item
                            yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=item, is_final=False)
                        yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=final_result, is_final=True)
                    else:
                        yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=result, is_final=True)
                except Exception as e:
                    yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=str(e), is_final=True, error=str(e))

    async def stream_with_tools_async(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Mapping[str, Callable[..., Any]],
        schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[Union[str, ToolProgress]]:
        """Async stream response with tool progress (OpenAI streaming with tools)."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        stream = await client.chat.completions.create(
            model=model,
            messages=list(messages),
            tools=schemas if schemas else None,
            tool_choice="auto" if schemas else None,
            stream=True,
        )

        tool_calls_buffer: dict[int, dict] = {}
        content_buffer = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_buffer += delta.content
                yield delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    if index not in tool_calls_buffer:
                        tool_calls_buffer[index] = {"id": "", "name": "", "arguments": ""}
                    
                    if tc.id:
                        tool_calls_buffer[index]["id"] = tc.id
                    if tc.function.name:
                        tool_calls_buffer[index]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_buffer[index]["arguments"] += tc.function.arguments

        if tool_calls_buffer:
            for index, tc_data in tool_calls_buffer.items():
                func_name = tc_data["name"]
                if func_name not in tools:
                    yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=f"Unknown tool: {func_name}", is_final=True, error=f"Unknown tool: {func_name}")
                    continue

                yield ToolProgress(call_id=tc_data["id"], name=func_name, progress="Starting...", is_final=False)

                try:
                    args = json.loads(tc_data["arguments"])
                    func = tools[func_name]
                    
                    if inspect.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                    
                    if inspect.isasyncgen(result):
                        final_result = None
                        async for item in result:
                            final_result = item
                            yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=item, is_final=False)
                        yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=final_result, is_final=True)
                    elif inspect.isgenerator(result):
                        final_result = None
                        for item in result:
                            final_result = item
                            yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=item, is_final=False)
                        yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=final_result, is_final=True)
                    else:
                        yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=result, is_final=True)
                except Exception as e:
                    yield ToolProgress(call_id=tc_data["id"], name=func_name, progress=str(e), is_final=True, error=str(e))