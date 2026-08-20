"""Public package exports for simpli-agent."""

from .config import AgentConfig, load_config, create_agent_from_config
from .core import Agent, AsyncMiddleware, Middleware
from .costs import CostTracker, TokenUsage, get_tracker, estimate_tokens
from .debug import DebugREPL, AgentDebugger, debug, debug_agent, Breakpoint
from .decorators import generate_tool_schema
from .eval import (
    TestCase,
    TestResult,
    EvaluationResult,
    Evaluator,
    evaluate,
    evaluate_async,
    benchmark,
)
from .memory import SQLiteMemory
from .multiagent import AgentRegistry, AgentTeam, AgentMessage, DelegationResult, create_team
from .pipeline import Pipeline, ToolChain, pipe, compose, register_pipeline
from .prompts import (
    PromptTemplate,
    PromptBuilder,
    format_history,
    format_tools,
    render_default_prompt,
    SYSTEM_PROMPT,
    HISTORY_TEMPLATE,
    TOOLS_TEMPLATE,
    TASK_TEMPLATE,
    DEFAULT_AGENT_TEMPLATE,
    RESEARCH_TEMPLATE,
    CODING_TEMPLATE,
    ANALYSIS_TEMPLATE,
)
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
from .types import ToolCall, ToolResult, ToolProgress

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
    "CostTracker",
    "TokenUsage",
    "get_tracker",
    "estimate_tokens",
    "Pipeline",
    "ToolChain",
    "pipe",
    "compose",
    "register_pipeline",
    "TestCase",
    "TestResult",
    "EvaluationResult",
    "Evaluator",
    "evaluate",
    "evaluate_async",
    "benchmark",
    "ToolCall",
    "ToolResult",
    "ToolProgress",
    "AgentRegistry",
    "AgentTeam",
    "AgentMessage",
    "DelegationResult",
    "create_team",
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
    "DebugREPL",
    "AgentDebugger",
    "debug",
    "debug_agent",
    "Breakpoint",
]
if OpenAIBackend:
    __all__.append("OpenAIBackend")
if AnthropicBackend:
    __all__.append("AnthropicBackend")
if OllamaBackend:
    __all__.append("OllamaBackend")
