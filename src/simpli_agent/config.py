"""Configuration management for simpli-agent."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class BackendConfig:
    """Backend configuration."""
    type: str = "hermes"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None


@dataclass
class MemoryConfig:
    """Memory configuration."""
    db_path: str = ":memory:"
    semantic: bool = False
    embedding_dim: int = 1536


@dataclass
class ToolConfig:
    """Tool configuration."""
    parallel: bool = True
    timeout: float = 30.0
    max_turns: int = 10


@dataclass
class TracingConfig:
    """Tracing configuration."""
    enabled: bool = False
    backend: str = "console"  # console, otel, langfuse, langsmith
    endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    backend: BackendConfig = field(default_factory=BackendConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    system_prompt: Optional[str] = None


def load_config(path: str | Path) -> AgentConfig:
    """Load configuration from TOML or YAML file."""
    path = Path(path)
    content = path.read_text()

    if path.suffix in (".toml",):
        if tomllib is None:
            raise ImportError("tomli required for TOML config on Python < 3.11. Install with: pip install tomli")
        data = tomllib.loads(content)
    elif path.suffix in (".yaml", ".yml") and YAML_AVAILABLE:
        data = yaml.safe_load(content)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return parse_config(data)


def parse_config(data: dict[str, Any]) -> AgentConfig:
    """Parse configuration dictionary into AgentConfig."""
    backend_data = data.get("backend", {})
    memory_data = data.get("memory", {})
    tools_data = data.get("tools", {})
    tracing_data = data.get("tracing", {})

    return AgentConfig(
        backend=BackendConfig(**backend_data),
        memory=MemoryConfig(**memory_data),
        tools=ToolConfig(**tools_data),
        tracing=TracingConfig(**tracing_data),
        system_prompt=data.get("system_prompt"),
    )


def create_agent_from_config(config: AgentConfig) -> "Agent":
    """Create an Agent instance from configuration."""
    from .core import Agent
    from .backends import HermesBackend, OpenAIBackend, AnthropicBackend, OllamaBackend

    # Resolve backend
    backend_type = config.backend.type
    if backend_type == "openai":
        backend = OpenAIBackend(
            api_key=config.backend.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.backend.base_url,
            organization=config.backend.organization,
        )
    elif backend_type == "anthropic":
        backend = AnthropicBackend(
            api_key=config.backend.api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=config.backend.base_url,
        )
    elif backend_type == "ollama":
        backend = OllamaBackend(base_url=config.backend.base_url)
    else:
        backend = HermesBackend()

    # Create agent
    agent = Agent(
        model=config.backend.model,
        backend=backend,
        db_path=config.memory.db_path,
        system_prompt=config.system_prompt,
        parallel_tools=config.tools.parallel,
        max_turns=config.tools.max_turns,
        semantic_memory=config.memory.semantic,
    )

    return agent


__all__ = [
    "BackendConfig",
    "MemoryConfig",
    "ToolConfig",
    "TracingConfig",
    "AgentConfig",
    "load_config",
    "parse_config",
    "create_agent_from_config",
]