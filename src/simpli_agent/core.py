"""High-level Agent facade tying tools, memory, and backends together."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .backends import (
    Backend,
    HermesBackend,
    OpenAIBackend,
    AnthropicBackend,
    OllamaBackend,
)
from .decorators import generate_tool_schema
from .memory import SQLiteMemory
from .semantic_memory import SemanticMemory
from .structured import PYDANTIC_AVAILABLE, parse_structured_output
from .tracing import CallbackHandler, Tracer, get_tracer

T = TypeVar("T")

_BACKENDS: dict[str, type[Backend]] = {"hermes": HermesBackend}
if OpenAIBackend:
    _BACKENDS["openai"] = OpenAIBackend
if AnthropicBackend:
    _BACKENDS["anthropic"] = AnthropicBackend
if OllamaBackend:
    _BACKENDS["ollama"] = OllamaBackend


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolResult:
    """Represents a tool execution result."""
    call_id: str | None
    name: str
    result: Any
    error: str | None = None


class Agent(AbstractContextManager):
    """Programmatic agent runtime with embedded SQLite state and tool schemas."""

    def __init__(
        self,
        model: str = "gpt-4o",
        backend: str | Backend = "hermes",
        db_path: str | Path = ":memory:",
        system_prompt: str | None = None,
        parallel_tools: bool = True,
        max_turns: int = 10,
        semantic_memory: bool = False,
        embed_func: Optional[callable] = None,
        tracer: Tracer | CallbackHandler | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.parallel_tools = parallel_tools
        self.max_turns = max_turns
        self.backend_type = backend if isinstance(backend, str) else backend.__class__.__name__
        self.tools: dict[str, Callable[..., Any]] = {}
        self.tool_schemas: list[dict[str, Any]] = []
        self.tool_output_models: dict[str, type] = {}
        self.tool_confirm: dict[str, bool] = {}
        self.memory = SQLiteMemory(db_path)
        self.semantic_memory = SemanticMemory(db_path, embed_func=embed_func) if semantic_memory else None
        self.backend = self._resolve_backend(backend)
        self._turn_count = 0
        self._checkpoints: list[dict[str, Any]] = []
        self._tracer = tracer or get_tracer()
        self._callbacks = CallbackHandler() if not isinstance(tracer, CallbackHandler) else tracer

    def _resolve_backend(self, backend: str | Backend) -> Backend:
        if isinstance(backend, Backend):
            return backend
        try:
            return _BACKENDS[backend]()
        except KeyError as exc:
            available = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"Unknown backend '{backend}'. Available backends: {available}") from exc

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        confirm: bool = False,
        output_model: type | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register ``func`` as an agent tool.

        Args:
            func: The function to register
            confirm: If True, require human confirmation before executing
            output_model: Optional Pydantic model to validate/parse the output
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            schema = generate_tool_schema(f)
            self.tool_schemas.append(schema)
            self.tools[f.__name__] = f
            if output_model:
                self.tool_output_models[f.__name__] = output_model
            if confirm:
                self.tool_confirm[f.__name__] = True
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self.memory.list_messages():
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        self._callbacks.emit("on_tool_start", call.name, call.arguments)
        start_time = time.time()

        if call.name not in self.tools:
            result = ToolResult(call_id=call.call_id, name=call.name, result=None, error=f"Unknown tool: {call.name}")
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result

        if self.tool_confirm.get(call.name, False):
            confirm = input(f"Confirm tool '{call.name}' with args {call.arguments}? [y/N] ")
            if confirm.lower() != "y":
                result = ToolResult(call_id=call.call_id, name=call.name, result=None, error="User cancelled")
                self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
                return result

        try:
            func = self.tools[call.name]
            result_value = func(**call.arguments)

            if call.name in self.tool_output_models and PYDANTIC_AVAILABLE:
                result_value = parse_structured_output(self.tool_output_models[call.name], str(result_value))

            result = ToolResult(call_id=call.call_id, name=call.name, result=result_value)
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result
        except Exception as e:
            result = ToolResult(call_id=call.call_id, name=call.name, result=None, error=str(e))
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result

    async def _execute_tool_async(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call asynchronously."""
        self._callbacks.emit("on_tool_start", call.name, call.arguments)
        start_time = time.time()

        if call.name not in self.tools:
            result = ToolResult(call_id=call.call_id, name=call.name, result=None, error=f"Unknown tool: {call.name}")
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result

        if self.tool_confirm.get(call.name, False):
            confirm = input(f"Confirm tool '{call.name}' with args {call.arguments}? [y/N] ")
            if confirm.lower() != "y":
                result = ToolResult(call_id=call.call_id, name=call.name, result=None, error="User cancelled")
                self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
                return result

        try:
            func = self.tools[call.name]
            if inspect.iscoroutinefunction(func):
                result_value = await func(**call.arguments)
            else:
                result_value = func(**call.arguments)

            if call.name in self.tool_output_models and PYDANTIC_AVAILABLE:
                result_value = parse_structured_output(self.tool_output_models[call.name], str(result_value))

            result = ToolResult(call_id=call.call_id, name=call.name, result=result_value)
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result
        except Exception as e:
            result = ToolResult(call_id=call.call_id, name=call.name, result=None, error=str(e))
            self._callbacks.emit("on_tool_end", call.name, result, time.time() - start_time)
            return result

    def _execute_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls, optionally in parallel."""
        if self.parallel_tools and len(calls) > 1:
            return asyncio.run(self._execute_tools_async(calls))
        return [self._execute_tool(call) for call in calls]

    async def _execute_tools_async(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel."""
        tasks = [self._execute_tool_async(call) for call in calls]
        return await asyncio.gather(*tasks)

    def _run_loop(
        self,
        prompt: str,
        *,
        output_model: type[T] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> str | T:
        """Main agent loop with tool calling."""
        if messages is None:
            messages = self._build_messages(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        self._turn_count = 0

        while self._turn_count < self.max_turns:
            self._turn_count += 1

            if hasattr(self.backend, "run_with_tools"):
                response = self.backend.run_with_tools(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    schemas=self.tool_schemas,
                )
            else:
                response = self.backend.run(
                    model=self.model,
                    prompt=prompt,
                    tools=self.tools,
                    schemas=self.tool_schemas,
                    history=messages,
                )
                messages.append({"role": "assistant", "content": response})
                break

            if response.get("tool_calls"):
                tool_calls = [
                    ToolCall(name=tc["name"], arguments=tc["arguments"], call_id=tc.get("id"))
                    for tc in response["tool_calls"]
                ]
                results = self._execute_tools(tool_calls)

                for result in results:
                    if result.error:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": f"Error: {result.error}",
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": str(result.result),
                        })
                continue

            final_content = response.get("content", "")
            if output_model and PYDANTIC_AVAILABLE:
                return parse_structured_output(output_model, final_content)
            return final_content

        raise RuntimeError(f"Max turns ({self.max_turns}) exceeded")

    async def _run_loop_async(
        self,
        prompt: str,
        *,
        output_model: type[T] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> str | T:
        """Async main agent loop with tool calling."""
        if messages is None:
            messages = self._build_messages(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        self._turn_count = 0

        while self._turn_count < self.max_turns:
            self._turn_count += 1

            if hasattr(self.backend, "run_with_tools_async"):
                response = await self.backend.run_with_tools_async(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    schemas=self.tool_schemas,
                )
            else:
                response = await self.backend.run_async(
                    model=self.model,
                    prompt=prompt,
                    tools=self.tools,
                    schemas=self.tool_schemas,
                    history=messages,
                )
                messages.append({"role": "assistant", "content": response})
                break

            if response.get("tool_calls"):
                tool_calls = [
                    ToolCall(name=tc["name"], arguments=tc["arguments"], call_id=tc.get("id"))
                    for tc in response["tool_calls"]
                ]
                results = await self._execute_tools_async(tool_calls)

                for result in results:
                    if result.error:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": f"Error: {result.error}",
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": str(result.result),
                        })
                continue

            final_content = response.get("content", "")
            if output_model and PYDANTIC_AVAILABLE:
                return parse_structured_output(output_model, final_content)
            return final_content

        raise RuntimeError(f"Max turns ({self.max_turns}) exceeded")

    def run(
        self,
        prompt: str,
        *,
        output_model: type[T] | None = None,
    ) -> str | T:
        """Execute a task, persist messages, and return response or structured output."""
        run_id = str(time.time())
        self._callbacks.emit("on_run_start", run_id, {"prompt": prompt, "model": self.model})

        start_time = time.time()
        try:
            self.memory.add_message("user", prompt)
            if self.semantic_memory:
                self.semantic_memory.add_message("user", prompt)
            messages = self._build_messages(prompt)
            result = self._run_loop(prompt, output_model=output_model, messages=messages)

            if isinstance(result, str):
                self.memory.add_message("assistant", result)
                if self.semantic_memory:
                    self.semantic_memory.add_message("assistant", result)
            else:
                self.memory.add_message("assistant", str(result))
                if self.semantic_memory:
                    self.semantic_memory.add_message("assistant", str(result))

            self._callbacks.emit("on_run_end", run_id, result, time.time() - start_time)
            return result
        except Exception as e:
            self._callbacks.emit("on_error", e)
            raise

    async def run_async(
        self,
        prompt: str,
        *,
        output_model: type[T] | None = None,
    ) -> str | T:
        """Async version of run with structured output support."""
        self.memory.add_message("user", prompt)
        if self.semantic_memory:
            self.semantic_memory.add_message("user", prompt)
        messages = self._build_messages(prompt)
        result = await self._run_loop_async(prompt, output_model=output_model, messages=messages)

        if isinstance(result, str):
            self.memory.add_message("assistant", result)
            if self.semantic_memory:
                self.semantic_memory.add_message("assistant", result)
        else:
            self.memory.add_message("assistant", str(result))
            if self.semantic_memory:
                self.semantic_memory.add_message("assistant", str(result))
        return result

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream a response as whitespace-delimited chunks."""
        response = self.run(prompt)
        for chunk in response.split():
            yield chunk

    async def stream_async(self, prompt: str) -> AsyncIterator[str]:
        """Async stream a response."""
        async for chunk in self.backend.stream_async(
            model=self.model,
            prompt=prompt,
            tools=self.tools,
            schemas=self.tool_schemas,
            history=self._build_messages(prompt),
        ):
            yield chunk

    def history(self, limit: Optional[int] = None) -> list[dict[str, str | int]]:
        """Return persisted conversation messages."""
        return self.memory.list_messages(limit=limit)

    def search_memory(
        self,
        query: str,
        limit: int = 10,
        semantic: bool = True,
        threshold: float = 0.3,
    ) -> list[dict[str, str | int]]:
        """Search persisted conversation messages.

        Args:
            query: Search query
            limit: Maximum results
            semantic: Use semantic/vector search if available
            threshold: Minimum similarity score for semantic results (0-1)
        """
        if self.semantic_memory and semantic:
            return self.semantic_memory.search(query, limit=limit, semantic=True, threshold=threshold)
        return self.memory.search(query, limit=limit)

    def checkpoint(self) -> int:
        """Create a checkpoint of current conversation state. Returns checkpoint ID."""
        checkpoint = {
            "messages": self.memory.list_messages(),
            "turn_count": self._turn_count,
        }
        self._checkpoints.append(checkpoint)
        return len(self._checkpoints) - 1

    def rollback(self, checkpoint_id: int) -> None:
        """Rollback to a previous checkpoint."""
        if 0 <= checkpoint_id < len(self._checkpoints):
            cp = self._checkpoints[checkpoint_id]
            self.memory.connection.execute("DELETE FROM history")
            self.memory.connection.execute("DELETE FROM history_fts")
            for msg in cp["messages"]:
                self.memory.add_message(msg["role"], msg["content"])
            self._turn_count = cp["turn_count"]
            self._checkpoints = self._checkpoints[:checkpoint_id + 1]

    def fork(self) -> "Agent":
        """Create a new agent with the same conversation history."""
        new_agent = Agent(
            model=self.model,
            backend=self.backend,
            db_path=":memory:",
            system_prompt=self.system_prompt,
            parallel_tools=self.parallel_tools,
            max_turns=self.max_turns,
            semantic_memory=self.semantic_memory is not None,
        )
        for msg in self.memory.list_messages():
            new_agent.memory.add_message(msg["role"], msg["content"])
            if new_agent.semantic_memory:
                new_agent.semantic_memory.add_message(msg["role"], msg["content"])
        new_agent.tools = self.tools.copy()
        new_agent.tool_schemas = self.tool_schemas.copy()
        new_agent.tool_output_models = self.tool_output_models.copy()
        new_agent.tool_confirm = self.tool_confirm.copy()
        return new_agent

    def close(self) -> None:
        """Close resources owned by the agent."""
        self.memory.close()
        if self.semantic_memory:
            self.semantic_memory.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
