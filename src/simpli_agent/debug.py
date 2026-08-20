"""Debug REPL for interactive agent debugging."""

from __future__ import annotations

import cmd
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from .core import Agent
from .types import ToolCall, ToolResult


@dataclass
class Breakpoint:
    """A breakpoint condition."""
    tool_name: str | None = None  # None = break on any tool
    condition: str | None = None  # Python expression to evaluate
    enabled: bool = True


class DebugREPL(cmd.Cmd):
    """Interactive debugger for agent execution."""
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║                    Simpli-Agent Debug REPL                     ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                     ║
║    step, s       - Execute next tool call                     ║
║    continue, c   - Continue execution                         ║
║    break, b      - Set breakpoint on tool                     ║
║    watch, w      - Watch variable/expression                  ║
║    inspect, i    - Inspect agent state                        ║
║    history, h    - Show conversation history                  ║
║    tools, t      - List registered tools                      ║
║    memory, m     - Show memory contents                       ║
║    cost          - Show token usage/cost                      ║
║    breakpoints   - List breakpoints                           ║
║    clear         - Clear all breakpoints                      ║
║    quit, q       - Exit debugger                              ║
║    help          - Show this help                             ║
╚══════════════════════════════════════════════════════════════╝
"""
    prompt = "(debug) "
    
    def __init__(self, agent: Agent, prompt: str):
        super().__init__()
        self.agent = agent
        self.original_prompt = prompt
        self.breakpoints: list[Breakpoint] = []
        self.watches: list[str] = []
        self._paused = False
        self._pending_tool_call: ToolCall | None = None
        self._tool_results: list[ToolResult] = []
        self._current_response = ""
        
    def _should_break(self, tool_call: ToolCall) -> bool:
        """Check if we should break on this tool call."""
        for bp in self.breakpoints:
            if not bp.enabled:
                continue
            if bp.tool_name and bp.tool_name != tool_call.name:
                continue
            if bp.condition:
                try:
                    # Evaluate condition in context of tool_call
                    context = {"call": tool_call, "args": tool_call.arguments}
                    if not eval(bp.condition, {"__builtins__": {}}, context):
                        continue
                except Exception:
                    continue
            return True
        return False
    
    def do_step(self, arg: str) -> bool:
        """Execute the next tool call and pause."""
        if self._pending_tool_call:
            # We have a pending tool call, execute it
            result = self.agent._execute_tool(self._pending_tool_call)
            self._tool_results.append(result)
            self._pending_tool_call = None
            print(f"Tool result: {result.result}")
            if result.error:
                print(f"Error: {result.error}")
        else:
            print("No pending tool call. Use 'continue' to run.")
        return False
    
    def do_continue(self, arg: str) -> bool:
        """Continue execution until next breakpoint or completion."""
        self._paused = False
        return True  # Exit the REPL loop
    
    def do_break(self, arg: str) -> bool:
        """Set a breakpoint. Usage: break [tool_name] [if condition]"""
        parts = arg.split(" if ")
        tool_name = parts[0].strip() if parts[0].strip() else None
        condition = parts[1].strip() if len(parts) > 1 else None
        
        bp = Breakpoint(tool_name=tool_name, condition=condition)
        self.breakpoints.append(bp)
        desc = f"Breakpoint on {tool_name or 'any tool'}"
        if condition:
            desc += f" if {condition}"
        print(desc)
        return False
    
    def do_watch(self, arg: str) -> bool:
        """Watch an expression. Usage: watch expression"""
        if arg:
            self.watches.append(arg)
            print(f"Watching: {arg}")
        else:
            print("Usage: watch <expression>")
        return False
    
    def do_inspect(self, arg: str) -> bool:
        """Inspect agent state."""
        what = arg.strip().lower()
        
        if what in ("", "agent", "state"):
            print(f"Model: {self.agent.model}")
            print(f"Backend: {self.agent.backend_type}")
            print(f"System prompt: {self.agent.system_prompt or 'None'}")
            print(f"Parallel tools: {self.agent.parallel_tools}")
            print(f"Max turns: {self.agent.max_turns}")
            print(f"Semantic memory: {self.agent.semantic_memory is not None}")
            print(f"Tools: {list(self.agent.tools.keys())}")
            print(f"Middleware: {len(self.agent._middleware)} sync, {len(self.agent._async_middleware)} async")
            
        elif what == "tools":
            self.do_tools("")
            
        elif what == "memory":
            self.do_memory("")
            
        elif what == "cost":
            self.do_cost("")
            
        elif what == "call" and self._pending_tool_call:
            call = self._pending_tool_call
            print(f"Tool: {call.name}")
            print(f"Arguments: {call.arguments}")
            print(f"Call ID: {call.call_id}")
            
        else:
            print(f"Unknown inspection target: {what}")
            print("Available: agent, tools, memory, cost, call")
        return False
    
    def do_history(self, arg: str) -> bool:
        """Show conversation history."""
        limit = int(arg) if arg.isdigit() else 20
        messages = self.agent.history(limit=limit)
        if not messages:
            print("No history")
            return False
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:100]
            print(f"  [{role}] {content}")
        return False
    
    def do_tools(self, arg: str) -> bool:
        """List registered tools with schemas."""
        if not self.agent.tools:
            print("No tools registered")
            return False
        for name, func in self.agent.tools.items():
            schema = next((s for s in self.agent.tool_schemas if s["function"]["name"] == name), None)
            desc = schema["function"]["description"] if schema else "No description"
            params = schema["function"]["parameters"] if schema else {}
            confirm = self.agent.tool_confirm.get(name, False)
            output_model = self.agent.tool_output_models.get(name)
            print(f"  {name}: {desc}")
            print(f"    Params: {params}")
            if confirm:
                print(f"    [CONFIRM REQUIRED]")
            if output_model:
                print(f"    Output: {output_model.__name__}")
        return False
    
    def do_memory(self, arg: str) -> bool:
        """Show memory contents."""
        messages = self.agent.history()
        print(f"Messages: {len(messages)}")
        for msg in messages[-10:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:80]
            print(f"  [{role}] {content}")
        
        if self.agent.semantic_memory:
            print("\nSemantic memory: enabled")
        return False
    
    def do_cost(self, arg: str) -> bool:
        """Show token usage and cost."""
        usage = self.agent.usage()
        print(f"Model: {usage['model']}")
        print(f"Runs: {usage['total_runs']}")
        print(f"Prompt tokens: {usage['total_prompt_tokens']}")
        print(f"Completion tokens: {usage['total_completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Total cost: ${usage['total_cost_usd']:.6f}")
        print(f"Avg cost/run: ${usage['avg_cost_per_run']:.6f}")
        return False
    
    def do_breakpoints(self, arg: str) -> bool:
        """List all breakpoints."""
        if not self.breakpoints:
            print("No breakpoints set")
            return False
        for i, bp in enumerate(self.breakpoints):
            status = "enabled" if bp.enabled else "disabled"
            desc = f"  {i}: {bp.tool_name or 'any tool'} [{status}]"
            if bp.condition:
                desc += f" if {bp.condition}"
            print(desc)
        return False
    
    def do_clear(self, arg: str) -> bool:
        """Clear breakpoints. Usage: clear [index]"""
        if arg and arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(self.breakpoints):
                self.breakpoints.pop(idx)
                print(f"Cleared breakpoint {idx}")
            else:
                print("Invalid breakpoint index")
        else:
            self.breakpoints.clear()
            print("All breakpoints cleared")
        return False
    
    def do_quit(self, arg: str) -> bool:
        """Exit the debugger."""
        print("Exiting debugger...")
        return True
    
    do_q = do_quit
    do_exit = do_quit
    
    def do_s(self, arg: str) -> bool:
        return self.do_step(arg)
    
    def do_c(self, arg: str) -> bool:
        return self.do_continue(arg)
    
    def do_b(self, arg: str) -> bool:
        return self.do_break(arg)
    
    def do_i(self, arg: str) -> bool:
        return self.do_inspect(arg)
    
    def do_h(self, arg: str) -> bool:
        return self.do_history(arg)
    
    def do_w(self, arg: str) -> bool:
        return self.do_watch(arg)
    
    def default(self, line: str) -> bool:
        """Evaluate Python expression."""
        try:
            # Try to eval first
            result = eval(line, {"__builtins__": {}}, {
                "agent": self.agent,
                "call": self._pending_tool_call,
                "results": self._tool_results,
            })
            if result is not None:
                print(result)
        except SyntaxError:
            try:
                exec(line, {"__builtins__": {}}, {
                    "agent": self.agent,
                    "call": self._pending_tool_call,
                    "results": self._tool_results,
                })
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        return False
    
    def emptyline(self) -> bool:
        return False


def debug_agent(agent: Agent, prompt: str) -> Any:
    """Run an agent with interactive debugging.
    
    Usage:
        result = debug_agent(agent, "Your prompt here")
    
    This will start an interactive REPL where you can:
    - step through tool calls
    - set breakpoints
    - inspect state
    - watch expressions
    """
    repl = DebugREPL(agent, prompt)
    
    # We need to hook into the agent's execution
    # For now, we'll run the agent and intercept tool calls
    # This is a simplified version - full integration would need
    # more hooks in the agent core
    
    print(f"Debugging agent with prompt: {prompt}")
    print(f"Model: {agent.model}, Backend: {agent.backend_type}")
    print(f"Tools: {list(agent.tools.keys())}")
    print()
    
    # Run the agent normally for now
    # In a full implementation, we'd patch _execute_tool to pause
    result = agent.run(prompt)
    
    # Then enter REPL for post-mortem inspection
    repl.cmdloop()
    
    return result


class AgentDebugger:
    """Context manager for debugging an agent run."""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self._original_execute_tool = agent._execute_tool
        self._original_execute_tool_async = agent._execute_tool_async
        self.breakpoints: list[Breakpoint] = []
        self._paused = False
        self._pending_call: ToolCall | None = None
        
    def __enter__(self) -> "AgentDebugger":
        # Patch the agent's tool execution
        self.agent._execute_tool = self._debug_execute_tool
        self.agent._execute_tool_async = self._debug_execute_tool_async
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original methods
        self.agent._execute_tool = self._original_execute_tool
        self.agent._execute_tool_async = self._original_execute_tool_async
    
    def _should_break(self, call: ToolCall) -> bool:
        for bp in self.breakpoints:
            if not bp.enabled:
                continue
            if bp.tool_name and bp.tool_name != call.name:
                continue
            if bp.condition:
                try:
                    if not eval(bp.condition, {"__builtins__": {}}, {"call": call, "args": call.arguments}):
                        continue
                except Exception:
                    continue
            return True
        return False
    
    def _debug_execute_tool(self, call: ToolCall) -> ToolResult:
        self._pending_call = call
        
        if self._should_break(call):
            print(f"\n🔴 Breakpoint hit: {call.name}({call.arguments})")
            self._enter_repl()
        
        result = self._original_execute_tool(call)
        
        if self._paused:
            print(f"Tool completed: {call.name} -> {result.result}")
            if result.error:
                print(f"Error: {result.error}")
        
        self._pending_call = None
        return result
    
    async def _debug_execute_tool_async(self, call: ToolCall) -> ToolResult:
        self._pending_call = call
        
        if self._should_break(call):
            print(f"\n🔴 Breakpoint hit: {call.name}({call.arguments})")
            self._enter_repl()
        
        result = await self._original_execute_tool_async(call)
        
        if self._paused:
            print(f"Tool completed: {call.name} -> {result.result}")
            if result.error:
                print(f"Error: {result.error}")
        
        self._pending_call = None
        return result
    
    def _enter_repl(self) -> None:
        """Enter the interactive REPL."""
        self._paused = True
        repl = DebugREPL(self.agent, "")
        repl.breakpoints = self.breakpoints
        repl._pending_tool_call = self._pending_call
        repl.cmdloop()
        self.breakpoints = repl.breakpoints
    
    def add_breakpoint(self, tool_name: str | None = None, condition: str | None = None) -> None:
        """Add a breakpoint."""
        self.breakpoints.append(Breakpoint(tool_name=tool_name, condition=condition))
    
    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self.breakpoints.clear()


def debug(agent: Agent) -> AgentDebugger:
    """Create a debugger context for an agent.
    
    Usage:
        with debug(agent) as dbg:
            dbg.add_breakpoint("search_web")
            result = agent.run("Search for Python news")
    """
    return AgentDebugger(agent)


__all__ = [
    "DebugREPL",
    "debug_agent",
    "AgentDebugger",
    "debug",
    "Breakpoint",
]