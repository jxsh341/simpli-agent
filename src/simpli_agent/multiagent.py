"""Multi-agent coordination primitives for simpli-agent."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

from .core import Agent
from .types import ToolCall, ToolResult


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str | None  # None = broadcast
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class DelegationResult:
    """Result of a delegation to another agent."""
    agent_name: str
    task: str
    result: str
    success: bool
    error: str | None = None


class AgentRegistry:
    """Registry for managing multiple agents."""
    
    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._message_handlers: dict[str, list[callable]] = {}
    
    def register(self, name: str, agent: Agent) -> "AgentRegistry":
        """Register an agent by name."""
        self._agents[name] = agent
        return self
    
    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self._agents.keys())}")
        return self._agents[name]
    
    def __getitem__(self, name: str) -> Agent:
        return self.get(name)
    
    def __contains__(self, name: str) -> bool:
        return name in self._agents
    
    def names(self) -> list[str]:
        return list(self._agents.keys())
    
    def broadcast(self, message: str, from_agent: str | None = None, **metadata) -> list[DelegationResult]:
        """Broadcast a message to all registered agents."""
        results = []
        for name, agent in self._agents.items():
            if from_agent and name == from_agent:
                continue
            try:
                result = agent.run(message)
                results.append(DelegationResult(
                    agent_name=name,
                    task=message,
                    result=result,
                    success=True,
                ))
            except Exception as e:
                results.append(DelegationResult(
                    agent_name=name,
                    task=message,
                    result="",
                    success=False,
                    error=str(e),
                ))
        return results
    
    async def broadcast_async(self, message: str, from_agent: str | None = None, **metadata) -> list[DelegationResult]:
        """Broadcast a message to all registered agents asynchronously."""
        import asyncio
        
        async def run_one(name: str, agent: Agent):
            if from_agent and name == from_agent:
                return None
            try:
                result = await agent.run_async(message)
                return DelegationResult(
                    agent_name=name,
                    task=message,
                    result=result,
                    success=True,
                )
            except Exception as e:
                return DelegationResult(
                    agent_name=name,
                    task=message,
                    result="",
                    success=False,
                    error=str(e),
                )
        
        tasks = [run_one(name, agent) for name, agent in self._agents.items()]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    def delegate(self, from_agent: str, to_agent: str, task: str) -> DelegationResult:
        """Delegate a task from one agent to another."""
        if from_agent not in self._agents:
            raise KeyError(f"Source agent '{from_agent}' not found")
        if to_agent not in self._agents:
            raise KeyError(f"Target agent '{to_agent}' not found")
        
        # Add context about the delegation
        delegated_task = f"[Delegated from {from_agent}] {task}"
        
        try:
            result = self._agents[to_agent].run(delegated_task)
            return DelegationResult(
                agent_name=to_agent,
                task=task,
                result=result,
                success=True,
            )
        except Exception as e:
            return DelegationResult(
                agent_name=to_agent,
                task=task,
                result="",
                success=False,
                error=str(e),
            )
    
    async def delegate_async(self, from_agent: str, to_agent: str, task: str) -> DelegationResult:
        """Delegate a task asynchronously."""
        if from_agent not in self._agents:
            raise KeyError(f"Source agent '{from_agent}' not found")
        if to_agent not in self._agents:
            raise KeyError(f"Target agent '{to_agent}' not found")
        
        delegated_task = f"[Delegated from {from_agent}] {task}"
        
        try:
            result = await self._agents[to_agent].run_async(delegated_task)
            return DelegationResult(
                agent_name=to_agent,
                task=task,
                result=result,
                success=True,
            )
        except Exception as e:
            return DelegationResult(
                agent_name=to_agent,
                task=task,
                result="",
                success=False,
                error=str(e),
            )


class AgentTeam:
    """A team of agents that work together on tasks."""
    
    def __init__(self, name: str = "team"):
        self.name = name
        self.agents: dict[str, Agent] = {}
        self.registry = AgentRegistry()
    
    def add(self, name: str, agent: Agent) -> "AgentTeam":
        """Add an agent to the team."""
        self.agents[name] = agent
        self.registry.register(name, agent)
        return self
    
    def __getitem__(self, name: str) -> Agent:
        return self.agents[name]
    
    def run_collaborative(self, task: str, *, strategy: str = "sequential") -> dict[str, Any]:
        """Run a collaborative task with the team.
        
        Strategies:
        - sequential: Each agent builds on previous agent's output
        - parallel: All agents work on the same task independently
        - pipeline: Agents form a pipeline (requires ordered agents)
        """
        if strategy == "sequential":
            return self._run_sequential(task)
        elif strategy == "parallel":
            return self._run_parallel(task)
        elif strategy == "pipeline":
            return self._run_pipeline(task)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _run_sequential(self, task: str) -> dict[str, Any]:
        """Run agents sequentially, each building on previous output."""
        results = {}
        previous_output = task
        
        for name, agent in self.agents.items():
            prompt = f"Previous context: {previous_output}\n\nYour task: {task}"
            result = agent.run(prompt)
            results[name] = result
            previous_output = result
        
        return {
            "strategy": "sequential",
            "results": results,
            "final_output": list(results.values())[-1] if results else "",
        }
    
    def _run_parallel(self, task: str) -> dict[str, Any]:
        """Run all agents in parallel on the same task."""
        results = {}
        for name, agent in self.agents.items():
            try:
                result = agent.run(task)
                results[name] = {"result": result, "success": True}
            except Exception as e:
                results[name] = {"result": "", "success": False, "error": str(e)}
        
        return {
            "strategy": "parallel",
            "results": results,
        }
    
    def _run_pipeline(self, task: str) -> dict[str, Any]:
        """Run agents as a pipeline (ordered by addition)."""
        results = {}
        current_input = task
        
        for name, agent in self.agents.items():
            result = agent.run(current_input)
            results[name] = result
            current_input = result
        
        return {
            "strategy": "pipeline",
            "results": results,
            "final_output": list(results.values())[-1] if results else "",
        }
    
    async def run_collaborative_async(self, task: str, *, strategy: str = "sequential") -> dict[str, Any]:
        """Async version of run_collaborative."""
        if strategy == "sequential":
            results = {}
            previous_output = task
            for name, agent in self.agents.items():
                prompt = f"Previous context: {previous_output}\n\nYour task: {task}"
                result = await agent.run_async(prompt)
                results[name] = result
                previous_output = result
            return {"strategy": "sequential", "results": results, "final_output": list(results.values())[-1]}
        
        elif strategy == "parallel":
            import asyncio
            async def run_one(name, agent):
                try:
                    result = await agent.run_async(task)
                    return name, {"result": result, "success": True}
                except Exception as e:
                    return name, {"result": "", "success": False, "error": str(e)}
            
            tasks = [run_one(name, agent) for name, agent in self.agents.items()]
            results_list = await asyncio.gather(*tasks)
            results = dict(results_list)
            return {"strategy": "parallel", "results": results}
        
        elif strategy == "pipeline":
            results = {}
            current_input = task
            for name, agent in self.agents.items():
                result = await agent.run_async(current_input)
                results[name] = result
                current_input = result
            return {"strategy": "pipeline", "results": results, "final_output": list(results.values())[-1]}
        
        raise ValueError(f"Unknown strategy: {strategy}")


# Convenience function for quick multi-agent setup
def create_team(
    agents: dict[str, Agent],
    name: str = "team",
) -> AgentTeam:
    """Create an agent team from a dictionary of agents."""
    team = AgentTeam(name)
    for name, agent in agents.items():
        team.add(name, agent)
    return team


__all__ = [
    "AgentMessage",
    "DelegationResult",
    "AgentRegistry",
    "AgentTeam",
    "create_team",
]