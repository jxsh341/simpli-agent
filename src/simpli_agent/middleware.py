"""Built-in middleware for simpli-agent."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .core import Middleware, ToolCall, ToolResult


def logging_middleware(
    *,
    log_args: bool = True,
    log_result: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> Middleware:
    """Log tool calls and results."""
    log = logger or print

    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        if log_args:
            log(f"[tool] {call.name}({call.arguments})")
        result = next(call)
        if log_result:
            if result.error:
                log(f"[tool] {call.name} -> ERROR: {result.error}")
            else:
                log(f"[tool] {call.name} -> {result.result}")
        return result

    return middleware


def timing_middleware(
    *,
    logger: Optional[Callable[[str], None]] = None,
    slow_threshold: float = 1.0,
) -> Middleware:
    """Log tool execution time."""
    log = logger or print

    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        start = time.perf_counter()
        result = next(call)
        elapsed = time.perf_counter() - start
        if elapsed > slow_threshold:
            log(f"[timing] {call.name} took {elapsed:.3f}s (SLOW)")
        else:
            log(f"[timing] {call.name} took {elapsed:.3f}s")
        return result

    return middleware


def retry_middleware(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: Optional[Callable[[Exception], bool]] = None,
) -> Middleware:
    """Retry failed tool calls with exponential backoff."""
    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        delay = base_delay
        last_error = None

        for attempt in range(max_retries + 1):
            result = next(call)
            if not result.error:
                return result

            last_error = Exception(result.error)
            if retry_on and not retry_on(last_error):
                return result

            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * exponential_base, max_delay)

        # All retries exhausted
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            result=None,
            error=f"Retries exhausted: {last_error}",
        )

    return middleware


def cache_middleware(
    *,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[[ToolCall], str]] = None,
) -> Middleware:
    """Cache tool results by arguments."""
    cache: dict[str, tuple[ToolResult, float]] = {}

    def default_key(call: ToolCall) -> str:
        import json
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"

    key = key_func or default_key

    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        cache_key = key(call)
        now = time.time()

        if cache_key in cache:
            result, timestamp = cache[cache_key]
            if ttl is None or (now - timestamp) < ttl:
                # Return cached result (create new object to avoid mutation)
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    result=result.result,
                    error=result.error,
                )

        result = next(call)
        if not result.error:
            cache[cache_key] = (result, now)
        return result

    return middleware


def rate_limit_middleware(
    max_calls: int,
    window: float = 60.0,
) -> Middleware:
    """Rate limit tool calls (per tool name)."""
    from collections import deque
    calls: dict[str, deque[float]] = {}

    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        now = time.time()
        if call.name not in calls:
            calls[call.name] = deque()

        # Remove old calls outside window
        while calls[call.name] and calls[call.name][0] < now - window:
            calls[call.name].popleft()

        if len(calls[call.name]) >= max_calls:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=None,
                error=f"Rate limit exceeded for {call.name}",
            )

        calls[call.name].append(now)
        return next(call)

    return middleware


def transform_args_middleware(
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> Middleware:
    """Transform tool arguments before execution."""
    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        new_args = transform(call.arguments)
        new_call = ToolCall(name=call.name, arguments=new_args, call_id=call.call_id)
        return next(new_call)
    return middleware


def transform_result_middleware(
    transform: Callable[[ToolResult], ToolResult],
) -> Middleware:
    """Transform tool result after execution."""
    def middleware(call: ToolCall, next: Callable[[ToolCall], ToolResult]) -> ToolResult:
        result = next(call)
        return transform(result)
    return middleware


# Async versions
from .core import AsyncMiddleware


async def async_retry_middleware(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: Optional[Callable[[Exception], bool]] = None,
) -> AsyncMiddleware:
    """Async retry with exponential backoff."""
    async def middleware(call: ToolCall, next: Callable[[ToolCall], Any]) -> ToolResult:
        delay = base_delay
        last_error = None

        for attempt in range(max_retries + 1):
            result = await next(call)
            if not result.error:
                return result

            last_error = Exception(result.error)
            if retry_on and not retry_on(last_error):
                return result

            if attempt < max_retries:
                import asyncio
                await asyncio.sleep(delay)
                delay = min(delay * exponential_base, max_delay)

        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            result=None,
            error=f"Retries exhausted: {last_error}",
        )

    return middleware


async def async_cache_middleware(
    *,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[[ToolCall], str]] = None,
) -> AsyncMiddleware:
    """Async cache tool results."""
    cache: dict[str, tuple[ToolResult, float]] = {}

    def default_key(call: ToolCall) -> str:
        import json
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"

    key = key_func or default_key

    async def middleware(call: ToolCall, next: Callable[[ToolCall], Any]) -> ToolResult:
        cache_key = key(call)
        now = time.time()

        if cache_key in cache:
            result, timestamp = cache[cache_key]
            if ttl is None or (now - timestamp) < ttl:
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    result=result.result,
                    error=result.error,
                )

        result = await next(call)
        if not result.error:
            cache[cache_key] = (result, now)
        return result

    return middleware


__all__ = [
    "logging_middleware",
    "timing_middleware",
    "retry_middleware",
    "cache_middleware",
    "rate_limit_middleware",
    "transform_args_middleware",
    "transform_result_middleware",
    "async_retry_middleware",
    "async_cache_middleware",
]