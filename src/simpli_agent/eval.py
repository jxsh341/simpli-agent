"""Evaluation harness for agent testing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import Agent


@dataclass
class TestCase:
    """A single test case for evaluation."""
    input: str
    expected_contains: Optional[str] = None
    expected_not_contains: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    expected_output_type: Optional[type] = None
    custom_check: Optional[Callable[[str], bool]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single test case."""
    case: TestCase
    passed: bool
    output: str
    duration: float
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregate evaluation results."""
    total: int
    passed: int
    failed: int
    duration: float
    results: list[TestResult]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Evaluation Summary",
            f"==================",
            f"Total: {self.total}",
            f"Passed: {self.passed}",
            f"Failed: {self.failed}",
            f"Pass Rate: {self.pass_rate:.1%}",
            f"Duration: {self.duration:.2f}s",
        ]
        if self.metrics:
            lines.append("Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def check_contains(output: str, expected: str) -> bool:
    """Check if output contains expected substring."""
    return expected.lower() in output.lower()


def check_not_contains(output: str, expected: str) -> bool:
    """Check if output does not contain substring."""
    return expected.lower() not in output.lower()


def check_tool_used(agent: Agent, tool_name: str) -> bool:
    """Check if a tool was called during the last run."""
    # This is a heuristic - we check if the tool was registered
    # Actual tool usage tracking would need middleware
    return tool_name in agent.tools


def run_case(agent: Agent, case: TestCase) -> TestResult:
    """Run a single test case."""
    start = time.time()
    tools_before = set(agent.tools.keys()) if hasattr(agent, 'tools') else set()
    
    try:
        output = agent.run(case.input)
        duration = time.time() - start
        
        # Determine tools used (approximate)
        tools_used = []
        for tool_name in agent.tools:
            if tool_name.lower() in output.lower():
                tools_used.append(tool_name)
        
        # Run checks
        checks = []
        
        if case.expected_contains:
            checks.append(("contains", check_contains(output, case.expected_contains)))
        
        if case.expected_not_contains:
            checks.append(("not_contains", check_not_contains(output, case.expected_not_contains)))
        
        if case.expected_tool:
            checks.append(("tool", case.expected_tool in tools_used))
        
        if case.expected_tools:
            checks.append(("tools", all(t in tools_used for t in case.expected_tools)))
        
        if case.expected_output_type:
            checks.append(("type", isinstance(output, case.expected_output_type)))
        
        if case.custom_check:
            checks.append(("custom", case.custom_check(output)))
        
        passed = all(result for _, result in checks)
        
        return TestResult(
            case=case,
            passed=passed,
            output=output,
            duration=duration,
            tools_used=tools_used,
            metrics={name: result for name, result in checks},
        )
    
    except Exception as e:
        return TestResult(
            case=case,
            passed=False,
            output="",
            duration=time.time() - start,
            error=str(e),
        )


def evaluate(
    agent: Agent,
    cases: list[TestCase],
    *,
    parallel: bool = False,
    max_workers: int = 4,
    progress: Optional[Callable[[int, int], None]] = None,
) -> EvaluationResult:
    """Evaluate an agent against test cases.

    Args:
        agent: The agent to evaluate
        cases: List of test cases
        parallel: Run cases in parallel
        max_workers: Max parallel workers
        progress: Optional callback(current, total)
    
    Returns:
        EvaluationResult with aggregate metrics
    """
    start = time.time()
    results: list[TestResult] = []
    
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_case, agent, case): case for case in cases}
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if progress:
                    progress(i + 1, len(cases))
    else:
        for i, case in enumerate(cases):
            results.append(run_case(agent, case))
            if progress:
                progress(i + 1, len(cases))
    
    duration = time.time() - start
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    # Aggregate metrics
    total_latency = sum(r.duration for r in results)
    metrics = {
        "avg_latency": total_latency / len(results) if results else 0,
        "total_latency": total_latency,
        "tools_usage": {},
    }
    
    for r in results:
        for tool in r.tools_used:
            metrics["tools_usage"][tool] = metrics["tools_usage"].get(tool, 0) + 1
    
    return EvaluationResult(
        total=len(cases),
        passed=passed,
        failed=failed,
        duration=duration,
        results=results,
        metrics=metrics,
    )


def evaluate_async(
    agent: Agent,
    cases: list[TestCase],
    *,
    max_concurrent: int = 4,
) -> EvaluationResult:
    """Async version of evaluate using agent.run_async."""
    import asyncio
    
    async def run_case_async(case: TestCase) -> TestResult:
        start = time.time()
        try:
            output = await agent.run_async(case.input)
            duration = time.time() - start
            
            checks = []
            if case.expected_contains:
                checks.append(check_contains(output, case.expected_contains))
            if case.expected_not_contains:
                checks.append(check_not_contains(output, case.expected_not_contains))
            if case.custom_check:
                checks.append(case.custom_check(output))
            
            passed = all(checks)
            return TestResult(
                case=case,
                passed=passed,
                output=output,
                duration=duration,
            )
        except Exception as e:
            return TestResult(
                case=case,
                passed=False,
                output="",
                duration=time.time() - start,
                error=str(e),
            )
    
    async def run_all():
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited(case):
            async with semaphore:
                return await run_case_async(case)
        
        return await asyncio.gather(*[limited(c) for c in cases])
    
    start = time.time()
    results = asyncio.run(run_all())
    duration = time.time() - start
    
    passed = sum(1 for r in results if r.passed)
    return EvaluationResult(
        total=len(cases),
        passed=passed,
        failed=len(cases) - passed,
        duration=duration,
        results=results,
    )


class Evaluator:
    """Stateful evaluator for running multiple evaluations."""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.history: list[EvaluationResult] = []
    
    def run(self, cases: list[TestCase], **kwargs) -> EvaluationResult:
        result = evaluate(self.agent, cases, **kwargs)
        self.history.append(result)
        return result
    
    def run_async(self, cases: list[TestCase], **kwargs) -> EvaluationResult:
        result = evaluate_async(self.agent, cases, **kwargs)
        self.history.append(result)
        return result
    
    def compare(self, other: "Evaluator") -> dict[str, Any]:
        """Compare this evaluator's last run with another."""
        if not self.history or not other.history:
            return {"error": "No history to compare"}
        
        a = self.history[-1]
        b = other.history[-1]
        
        return {
            "pass_rate_diff": a.pass_rate - b.pass_rate,
            "avg_latency_diff": a.metrics.get("avg_latency", 0) - b.metrics.get("avg_latency", 0),
            "a": a.summary(),
            "b": b.summary(),
        }


def benchmark(
    agent: Agent,
    inputs: list[str],
    *,
    runs: int = 10,
    warmup: int = 2,
) -> dict[str, Any]:
    """Benchmark agent latency."""
    # Warmup
    for _ in range(warmup):
        agent.run(inputs[0])
    
    latencies = []
    for _ in range(runs):
        for inp in inputs:
            start = time.time()
            agent.run(inp)
            latencies.append(time.time() - start)
    
    latencies.sort()
    return {
        "mean": sum(latencies) / len(latencies),
        "median": latencies[len(latencies) // 2],
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "min": latencies[0],
        "max": latencies[-1],
        "samples": len(latencies),
    }


__all__ = [
    "TestCase",
    "TestResult",
    "EvaluationResult",
    "Evaluator",
    "evaluate",
    "evaluate_async",
    "benchmark",
    "check_contains",
    "check_not_contains",
]