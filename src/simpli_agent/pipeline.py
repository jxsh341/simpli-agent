"""Tool composition and pipeline support for simpli-agent."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar
from functools import wraps
import inspect

from .core import ToolCall, ToolResult

T = TypeVar("T")


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    name: str
    func: Callable[..., Any]
    args_map: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    result_key: str | None = None  # Where to store result for next step


class Pipeline:
    """Composable pipeline of tools/functions.

    Example:
        pipeline = Pipeline()
        pipeline.add(search_web, args_map=lambda q: {"query": q})
        pipeline.add(summarize, args_map=lambda r: {"text": r["search_web"]})
        pipeline.add(format_output)

        result = pipeline.run({"query": "Python news"})
    """

    def __init__(self):
        self.steps: list[PipelineStep] = []
        self._context: dict[str, Any] = {}

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        args_map: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        result_key: str | None = None,
    ) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            func: The function to call
            name: Optional name (defaults to func.__name__)
            args_map: Maps context -> function arguments
            result_key: Key to store result in context (defaults to step name)
        """
        step_name = name or func.__name__
        self.steps.append(PipelineStep(
            name=step_name,
            func=func,
            args_map=args_map,
            result_key=result_key or step_name,
        ))
        return self

    def __or__(self, other: "Pipeline | Callable") -> "Pipeline":
        """Compose pipelines with | operator."""
        if isinstance(other, Pipeline):
            new_pipeline = Pipeline()
            new_pipeline.steps = self.steps + other.steps
            return new_pipeline
        else:
            return self.add(other)

    def run(self, initial_input: dict[str, Any] | Any = None) -> dict[str, Any]:
        """Execute the pipeline with initial input."""
        context = {}
        if initial_input is not None:
            if isinstance(initial_input, dict):
                context.update(initial_input)
            else:
                context["input"] = initial_input

        for step in self.steps:
            if step.args_map:
                args = step.args_map(context)
            else:
                # Pass all context as kwargs
                args = context

            result = step.func(**args)
            context[step.result_key] = result

        return context

    async def run_async(self, initial_input: dict[str, Any] | Any = None) -> dict[str, Any]:
        """Execute the pipeline asynchronously."""
        context = {}
        if initial_input is not None:
            if isinstance(initial_input, dict):
                context.update(initial_input)
            else:
                context["input"] = initial_input

        for step in self.steps:
            if step.args_map:
                args = step.args_map(context)
            else:
                args = context

            if inspect.iscoroutinefunction(step.func):
                result = await step.func(**args)
            else:
                result = step.func(**args)
            context[step.result_key] = result

        return context

    def to_tool(self, name: str, description: str = "") -> Callable[..., Any]:
        """Convert pipeline to a single tool function."""
        @wraps(self.run)
        def tool_func(**kwargs) -> dict[str, Any]:
            return self.run(kwargs)

        tool_func.__name__ = name
        tool_func.__doc__ = description or f"Pipeline: {' -> '.join(s.name for s in self.steps)}"
        return tool_func


def pipe(*funcs: Callable) -> Pipeline:
    """Create a pipeline from functions with auto args mapping.

    Example:
        def search(query: str) -> list[str]: ...
        def summarize(text: str) -> str: ...

        pipeline = pipe(search, summarize)
        result = pipeline.run({"query": "Python"})
        # Runs search(query="Python"), then summarize(text=<search_result>)
    """
    import inspect
    pipeline = Pipeline()
    prev_result_key = "input"

    for i, func in enumerate(funcs):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Create args_map that extracts needed params from context
        def make_args_map(params, prev_key):
            def args_map(ctx):
                if len(params) == 1:
                    # Single param: use previous result or input
                    return {params[0]: ctx.get(prev_key, ctx.get("input"))}
                else:
                    # Multiple params: try to match from context
                    return {p: ctx.get(p, ctx.get(prev_key)) for p in params}
            return args_map

        result_key = f"step_{i}" if i > 0 else func.__name__
        pipeline.add(
            func,
            args_map=make_args_map(params, prev_result_key),
            result_key=result_key,
        )
        prev_result_key = result_key

    return pipeline


class ToolChain:
    """Chain tools where each tool's output feeds into the next.

    Unlike Pipeline, ToolChain is simpler - just sequential function calls.
    """

    def __init__(self):
        self.tools: list[Callable] = []

    def add(self, func: Callable) -> "ToolChain":
        self.tools.append(func)
        return self

    def __or__(self, other: "ToolChain | Callable") -> "ToolChain":
        if isinstance(other, ToolChain):
            new_chain = ToolChain()
            new_chain.tools = self.tools + other.tools
            return new_chain
        return self.add(other)

    def run(self, input: Any) -> Any:
        result = input
        for tool in self.tools:
            if isinstance(result, dict):
                result = tool(**result)
            else:
                result = tool(result)
        return result

    async def run_async(self, input: Any) -> Any:
        result = input
        for tool in self.tools:
            if isinstance(result, dict):
                result = tool(**result)
            else:
                result = tool(result)
            if inspect.iscoroutine(result):
                result = await result
        return result


def compose(*funcs: Callable) -> Callable:
    """Compose functions right-to-left: compose(f, g)(x) = f(g(x))."""
    def composed(x: Any) -> Any:
        result = x
        for func in reversed(funcs):
            result = func(result)
        return result
    return composed


# Convenience: register pipeline as agent tool
def register_pipeline(agent, pipeline: Pipeline, name: str | None = None) -> None:
    """Register a pipeline as a tool on an agent."""
    tool_func = pipeline.to_tool(name or "pipeline")
    agent.tool(tool_func)


__all__ = [
    "Pipeline",
    "PipelineStep",
    "ToolChain",
    "pipe",
    "compose",
    "register_pipeline",
]