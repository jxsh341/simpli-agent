"""Prompt template system for simpli-agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import ToolCall, ToolResult


@dataclass
class PromptTemplate:
    """Jinja2-like prompt template with simple variable substitution.
    
    Example:
        template = PromptTemplate("System: {{ system_prompt }}")
        rendered = template.render(system_prompt="You are helpful")
    """
    template: str
    filters: dict[str, Callable[[Any], str]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Add default filters
        self.filters.setdefault("json", lambda x: str(x))
        self.filters.setdefault("str", str)
        self.filters.setdefault("repr", repr)
    
    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        result = self.template
        
        # Handle {{ variable | filter }} syntax
        def replace_var(match):
            full = match.group(0)
            inner = match.group(1).strip()
            
            # Split by pipe for filters
            parts = [p.strip() for p in inner.split("|")]
            var_name = parts[0]
            filters = parts[1:] if len(parts) > 1 else []
            
            if var_name not in kwargs:
                return f"{{{{ {inner} }}}}"  # Keep as-is if not found
            
            value = kwargs[var_name]
            
            # Apply filters
            for filter_name in filters:
                if filter_name in self.filters:
                    value = self.filters[filter_name](value)
                else:
                    # Try to call as method
                    if hasattr(value, filter_name):
                        value = getattr(value, filter_name)()
                    else:
                        value = f"{{{{ {inner} }}}}"
            
            return str(value)
        
        # Replace {{ variable }} or {{ variable | filter }}
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        result = re.sub(pattern, replace_var, result)
        
        return result
    
    def __add__(self, other: "PromptTemplate") -> "PromptTemplate":
        """Concatenate two templates."""
        return PromptTemplate(self.template + other.template, self.filters)


# Built-in template parts
SYSTEM_PROMPT = PromptTemplate("{{ system_prompt }}")

HISTORY_TEMPLATE = PromptTemplate("{%- for msg in history %}{{ msg.role }}: {{ msg.content }}{%- endfor %}")

TOOLS_TEMPLATE = PromptTemplate("Available tools:{%- for tool in tools %}- {{ tool.name }}: {{ tool.description }}  Parameters: {{ tool.parameters | json }}{%- endfor %}")

TASK_TEMPLATE = PromptTemplate("Task: {{ task }}")

# Default agent prompt template
DEFAULT_AGENT_TEMPLATE = PromptTemplate("{{ system_prompt }}\n\n{{ history }}\n\n{{ tools }}\n\n{{ task }}")


def format_history(messages: list[dict[str, Any]]) -> str:
    """Format conversation history for prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_tools(tools: list[dict[str, Any]]) -> str:
    """Format tool schemas for prompt."""
    lines = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            lines.append(f"- {func['name']}: {func.get('description', '')}")
            params = func.get("parameters", {})
            if params:
                lines.append(f"  Parameters: {params}")
    return "\n".join(lines)


def render_default_prompt(
    system_prompt: str = "",
    history: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    task: str = "",
) -> str:
    """Render the default agent prompt with all components."""
    return DEFAULT_AGENT_TEMPLATE.render(
        system_prompt=system_prompt,
        history=format_history(history or []),
        tools=format_tools(tools or []),
        task=task,
    )


class PromptBuilder:
    """Fluent builder for constructing prompts programmatically."""
    
    def __init__(self):
        self._parts: list[tuple[str, str]] = []  # (section_name, content)
    
    def system(self, prompt: str) -> "PromptBuilder":
        self._parts.append(("system", prompt))
        return self
    
    def user(self, prompt: str) -> "PromptBuilder":
        self._parts.append(("user", prompt))
        return self
    
    def assistant(self, prompt: str) -> "PromptBuilder":
        self._parts.append(("assistant", prompt))
        return self
    
    def tool_result(self, tool_name: str, result: str) -> "PromptBuilder":
        self._parts.append(("tool", f"{tool_name}: {result}"))
        return self
    
    def section(self, name: str, content: str) -> "PromptBuilder":
        self._parts.append((name, content))
        return self
    
    def history(self, messages: list[dict[str, Any]]) -> "PromptBuilder":
        formatted = format_history(messages)
        self._parts.append(("history", formatted))
        return self
    
    def tools(self, tool_schemas: list[dict[str, Any]]) -> "PromptBuilder":
        formatted = format_tools(tool_schemas)
        self._parts.append(("tools", formatted))
        return self
    
    def build(self, separator: str = "\n\n") -> str:
        """Build the final prompt string."""
        sections = []
        for name, content in self._parts:
            if content.strip():
                sections.append(f"{name.upper()}:\n{content}")
        return separator.join(sections)
    
    def render(self, template: PromptTemplate, **kwargs) -> str:
        """Render with a template, using built parts as defaults."""
        defaults = dict(self._parts)
        defaults.update(kwargs)
        return template.render(**defaults)


# Specialized templates for common patterns
RESEARCH_TEMPLATE = PromptTemplate("You are a research assistant. Your task is to {{ task }}.\n\nAvailable tools:\n{{ tools }}\n\nPlease use the tools to gather information and provide a comprehensive answer.")

CODING_TEMPLATE = PromptTemplate("You are a coding assistant. {{ task }}\n\nAvailable tools:\n{{ tools }}\n\nWrite clean, well-documented code. Explain your approach.")

ANALYSIS_TEMPLATE = PromptTemplate("You are an analyst. Analyze: {{ task }}\n\nAvailable tools:\n{{ tools }}\n\nProvide insights with supporting evidence.")


__all__ = [
    "PromptTemplate",
    "PromptBuilder",
    "format_history",
    "format_tools",
    "render_default_prompt",
    "SYSTEM_PROMPT",
    "HISTORY_TEMPLATE",
    "TOOLS_TEMPLATE",
    "TASK_TEMPLATE",
    "DEFAULT_AGENT_TEMPLATE",
    "RESEARCH_TEMPLATE",
    "CODING_TEMPLATE",
    "ANALYSIS_TEMPLATE",
]