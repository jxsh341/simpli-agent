"""Shared types for simpli-agent to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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


@dataclass
class ToolProgress:
    """Represents a progress update from a streaming tool."""
    call_id: str | None
    name: str
    progress: Any  # Progress data (can be string, dict, custom object)
    is_final: bool = False
    error: str | None = None


__all__ = ["ToolCall", "ToolResult", "ToolProgress"]