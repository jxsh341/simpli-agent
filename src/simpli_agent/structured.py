"""Structured output validation with Pydantic models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints
from functools import wraps

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore

T = TypeVar("T", bound=type[BaseModel])


def validate_output(model: T) -> Callable[[Callable[..., Any]], Callable[..., T]]:
    """Decorator to validate tool return value against a Pydantic model."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic required for validate_output. Install with: pip install pydantic")

    def decorator(func: Callable[..., Any]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            if isinstance(result, model):
                return result
            if isinstance(result, dict):
                return model.model_validate(result)
            if isinstance(result, (list, tuple)):
                return model.model_validate({"items": result})  # type: ignore
            raise ValidationError(f"Cannot validate {type(result)} as {model}")
        return wrapper
    return decorator


def parse_structured_output(model: T, content: str) -> T:
    """Parse LLM text output into a Pydantic model."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic required. Install with: pip install pydantic")

    import json
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValidationError(f"Could not extract JSON from: {content}")

    return model.model_validate(data)


def generate_output_schema(model: T) -> dict[str, Any]:
    """Generate JSON Schema for a Pydantic model to use as response_format."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic required. Install with: pip install pydantic")

    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": schema,
            "strict": True,
        },
    }


class StructuredTool:
    """Tool that returns validated Pydantic models."""

    def __init__(
        self,
        func: Callable[..., Any],
        output_model: type[BaseModel] | None = None,
    ):
        self.func = func
        self.output_model = output_model
        self.name = func.__name__
        self.schema = generate_tool_schema(func)

    def __call__(self, **kwargs: Any) -> Any:
        result = self.func(**kwargs)
        if self.output_model and PYDANTIC_AVAILABLE:
            return parse_structured_output(self.output_model, str(result))
        return result


def generate_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Import here to avoid circular dependency."""
    from .decorators import generate_tool_schema as _generate_tool_schema
    return _generate_tool_schema(func)


__all__ = [
    "validate_output",
    "parse_structured_output",
    "generate_output_schema",
    "StructuredTool",
    "PYDANTIC_AVAILABLE",
]