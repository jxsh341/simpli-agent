"""Cost and token tracking for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict
import threading


# Model pricing (USD per 1M tokens) - update as needed
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Ollama (local - free)
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    # Generic fallbacks
    "default": {"input": 0.0, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def cost(self, model: str) -> float:
        """Calculate cost for this usage."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    """Tracks token usage and costs across agent runs."""
    model: str
    _usage: TokenUsage = field(default_factory=TokenUsage)
    _run_usage: list[TokenUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_usage(self, prompt: int, completion: int) -> TokenUsage:
        """Add token usage and return the usage for this call."""
        usage = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
        with self._lock:
            self._usage = self._usage + usage
            self._run_usage.append(usage)
        return usage

    def add_usage_obj(self, usage: TokenUsage) -> TokenUsage:
        """Add a TokenUsage object."""
        with self._lock:
            self._usage = self._usage + usage
            self._run_usage.append(usage)
        return usage

    @property
    def total_usage(self) -> TokenUsage:
        return self._usage

    @property
    def total_cost(self) -> float:
        return self._usage.cost(self.model)

    @property
    def run_count(self) -> int:
        return len(self._run_usage)

    def last_run_usage(self) -> TokenUsage | None:
        return self._run_usage[-1] if self._run_usage else None

    def last_run_cost(self) -> float | None:
        usage = self.last_run_usage()
        return usage.cost(self.model) if usage else None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_runs": self.run_count,
            "total_prompt_tokens": self._usage.prompt_tokens,
            "total_completion_tokens": self._usage.completion_tokens,
            "total_tokens": self._usage.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_tokens_per_run": round(self._usage.total_tokens / self.run_count, 1) if self.run_count else 0,
            "avg_cost_per_run": round(self.total_cost / self.run_count, 6) if self.run_count else 0,
        }

    def reset(self) -> None:
        with self._lock:
            self._usage = TokenUsage()
            self._run_usage.clear()

    def set_model(self, model: str) -> None:
        self.model = model


# Global tracker for simple use cases
_default_tracker: CostTracker | None = None


def get_tracker(model: str = "gpt-4o") -> CostTracker:
    """Get or create the default cost tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = CostTracker(model)
    return _default_tracker


def set_tracker(tracker: CostTracker) -> None:
    """Set a custom global tracker."""
    global _default_tracker
    _default_tracker = tracker


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token for English)."""
    return max(1, len(text) // 4)


def track_call(model: str, prompt: str, completion: str) -> TokenUsage:
    """Convenience function to track a call with string inputs."""
    tracker = get_tracker(model)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    return tracker.add_usage(prompt_tokens, completion_tokens)


__all__ = [
    "TokenUsage",
    "CostTracker",
    "MODEL_PRICING",
    "get_tracker",
    "set_tracker",
    "estimate_tokens",
    "track_call",
]