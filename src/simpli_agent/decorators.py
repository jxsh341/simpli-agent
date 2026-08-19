"""Utilities for converting Python callables into LLM tool schemas."""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Union, get_args, get_origin, get_type_hints

_JSON_TYPE_MAPPING: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """Return a JSON Schema fragment for a Python type annotation."""
    if annotation is inspect.Signature.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            schema = _annotation_to_schema(non_none_args[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_annotation_to_schema(arg) for arg in non_none_args]}

    if origin in (list, Sequence, tuple):
        item_schema = _annotation_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin in (dict, Mapping):
        return {"type": "object"}

    return {"type": _JSON_TYPE_MAPPING.get(annotation, "string")}


def generate_tool_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Generate an OpenAI-compatible function tool schema for ``func``."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = inspect.getdoc(func) or "No description provided."

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        param_type = type_hints.get(name, param.annotation)
        properties[name] = {
            **_annotation_to_schema(param_type),
            "description": f"Parameter {name}",
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
