"""Public package exports for simpli-agent."""

from .config import AgentConfig, load_config, create_agent_from_config
from .core import Agent, AsyncMiddleware, Middleware
from .decorators import generate_tool_schema
from .memory import SQLiteMemory
from .semantic_memory import SemanticMemory, SQLITE_VEC_AVAILABLE
from .structured import (
    PYDANTIC_AVAILABLE,
    StructuredTool,
    generate_output_schema,
    parse_structured_output,
    validate_output,
)
from .tracing import CallbackHandler, Tracer, get_tracer, traced
from .middleware import (
    logging_middleware,
    timing_middleware,
    retry_middleware,
    cache_middleware,
    rate_limit_middleware,
    transform_args_middleware,
    transform_result_middleware,
    async_retry_middleware,
    async_cache_middleware,
)

try:
    from .backends import OpenAIBackend, AnthropicBackend, OllamaBackend
except ImportError:
    OpenAIBackend = None  # type: ignore
    AnthropicBackend = None  # type: ignore
    OllamaBackend = None  # type: ignore

__all__ = [
    "Agent",
    "AsyncMiddleware",
    "Middleware",
    "AgentConfig",
    "load_config",
    "create_agent_from_config",
    "SQLiteMemory",
    "SemanticMemory",
    "SQLITE_VEC_AVAILABLE",
    "generate_tool_schema",
    "PYDANTIC_AVAILABLE",
    "StructuredTool",
    "generate_output_schema",
    "parse_structured_output",
    "validate_output",
    "Tracer",
    "CallbackHandler",
    "get_tracer",
    "traced",
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
if OpenAIBackend:
    __all__.append("OpenAIBackend")
if AnthropicBackend:
    __all__.append("AnthropicBackend")
if OllamaBackend:
    __all__.append("OllamaBackend")
