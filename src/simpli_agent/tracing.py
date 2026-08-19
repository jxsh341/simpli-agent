"""Observability and tracing hooks for agent runs."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import Iterator

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore


@dataclass
class Span:
    """A single trace span."""
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def finish(self) -> None:
        self.end_time = time.time()

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class Tracer:
    """Simple tracer for agent operations."""

    def __init__(self, name: str = "simpli-agent"):
        self.name = name
        self.spans: list[Span] = []
        self._current_span: Span | None = None
        self._otel_tracer = None

        if OTEL_AVAILABLE:
            provider = TracerProvider(resource=Resource.create({"service.name": name}))
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(name)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
        """Create a new span."""
        span = Span(
            name=name,
            trace_id=str(uuid.uuid4())[:8],
            span_id=str(uuid.uuid4())[:8],
            parent_id=self._current_span.span_id if self._current_span else None,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        prev = self._current_span
        self._current_span = span

        otel_span = None
        if self._otel_tracer:
            otel_span = self._otel_tracer.start_span(name)
            for k, v in (attributes or {}).items():
                otel_span.set_attribute(k, str(v))

        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            if otel_span:
                otel_span.record_exception(e)
            raise
        finally:
            span.finish()
            if otel_span:
                otel_span.end()
            self._current_span = prev

    def get_spans(self) -> list[Span]:
        return self.spans

    def clear(self) -> None:
        self.spans.clear()


class CallbackHandler:
    """Callback-based observability for integration with LangSmith, LangFuse, etc."""

    def __init__(self):
        self.callbacks: dict[str, list[Callable]] = {
            "on_run_start": [],
            "on_run_end": [],
            "on_tool_start": [],
            "on_tool_end": [],
            "on_error": [],
        }

    def on_run_start(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self.callbacks["on_run_start"].append(callback)

    def on_run_end(self, callback: Callable[[str, Any, float], None]) -> None:
        self.callbacks["on_run_end"].append(callback)

    def on_tool_start(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self.callbacks["on_tool_start"].append(callback)

    def on_tool_end(self, callback: Callable[[str, Any, float], None]) -> None:
        self.callbacks["on_tool_end"].append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        self.callbacks["on_error"].append(callback)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for cb in self.callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception:
                pass  # Don't let callbacks break the main flow


# Global tracer instance
_default_tracer: Tracer | None = None


def get_tracer(name: str = "simpli-agent") -> Tracer:
    """Get or create the default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer(name)
    return _default_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set a custom global tracer."""
    global _default_tracer
    _default_tracer = tracer


# Decorator for automatic tracing
def traced(name: str | None = None, attributes: dict[str, Any] | None = None):
    """Decorator to automatically trace a function."""
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.span(span_name, attributes) as span:
                span.set_attribute("function", func.__name__)
                return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "Span",
    "Tracer",
    "CallbackHandler",
    "get_tracer",
    "set_tracer",
    "traced",
    "OTEL_AVAILABLE",
]