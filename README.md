# simpli-agent

`simpli-agent` is a small, embedded Python runtime for building tool-using agents without external setup wizards, daemon management, or large configuration files.

## Quick start

```python
from simpli_agent import Agent

agent = Agent(model="gpt-4o", backend="hermes")

@agent.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print(agent.run("Add 2 and 3"))
```

## Features

- **Agent**: High-level facade for tool registration, execution, and state with sync/async support
- **generate_tool_schema**: Automatic conversion from Python signatures (including `Optional`, `Literal`, Pydantic models) to LLM-compatible tool schemas
- **SQLiteMemory**: Embedded conversation persistence with FTS5 full-text search fallback
- **SemanticMemory**: Vector-based semantic search with sqlite-vec or numpy fallback
- **Backends**: Extensible backend abstraction with Hermes (local), OpenAI, Anthropic, Ollama
- **Structured Output**: Pydantic model validation and JSON Schema generation
- **Tool Composition**: Pipeline, ToolChain, pipe() for composing tools
- **Middleware**: Logging, timing, retry, caching, rate-limiting interceptors
- **Cost Tracking**: Token usage and cost estimation per model
- **Evaluation**: Test harness with benchmarking
- **Multi-Agent**: AgentTeam, AgentRegistry for coordination
- **Prompt Templates**: Jinja2-like templates with PromptBuilder
- **Debug REPL**: Interactive debugging with breakpoints
- **Serialization**: Save/load agent state to JSON
- **Context Manager**: `with Agent() as agent:` for automatic cleanup

## Installation

```bash
# Core
pip install simpli-agent

# With OpenAI backend
pip install simpli-agent[openai]

# With Pydantic for structured output
pip install simpli-agent[pydantic]

# With Anthropic backend
pip install simpli-agent[anthropic]

# With Ollama backend
pip install simpli-agent[ollama]

# With CLI
pip install simpli-agent[cli]

# All features
pip install simpli-agent[all]
```

## Backends

```python
# Local (no API key needed)
agent = Agent(model="gpt-4o", backend="hermes")

# OpenAI
agent = Agent(model="gpt-4o", backend="openai")  # needs OPENAI_API_KEY

# Anthropic
agent = Agent(model="claude-3-5-sonnet-20241022", backend="anthropic")  # needs ANTHROPIC_API_KEY

# Ollama (local models)
agent = Agent(model="llama3.1", backend="ollama")  # needs Ollama running
```

## Examples

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from simpli_agent import Agent

class Weather(BaseModel):
    city: str
    temperature: float
    conditions: str

agent = Agent(model="gpt-4o", backend="openai")

@agent.tool
def get_weather(city: str) -> Weather:
    """Get weather for a city."""
    return Weather(city=city, temperature=72.5, conditions="Sunny")

result = agent.run("What's the weather in Tokyo?", output_model=Weather)
print(result.city, result.temperature, result.conditions)
```

### Semantic Memory

```python
from simpli_agent import Agent

agent = Agent(semantic_memory=True)
agent.run("The capital of France is Paris")
agent.run("The capital of Japan is Tokyo")
agent.run("The capital of Germany is Berlin")

# Semantic search finds conceptually related content
results = agent.search_memory("capital of France", semantic=True)
for r in results:
    print(f"Score: {r.get('score', 'N/A'):.3f} - {r['content']}")
```

### Tool Composition

```python
from simpli_agent import Agent, pipe, Pipeline

agent = Agent(model="gpt-4o", backend="hermes")

@agent.tool
def search(query: str) -> list[str]:
    return [f"Result 1 for {query}", f"Result 2 for {query}"]

@agent.tool
def summarize(items: list[str]) -> str:
    return f"Summary of {len(items)} items"

# Pipe auto-maps arguments from function signatures
research_pipeline = pipe(search, summarize)
agent.tool(research_pipeline.to_tool("research"))

agent.run("Research Python async patterns")
```

### Middleware

```python
from simpli_agent import Agent, logging_middleware, retry_middleware, cache_middleware

agent = Agent(model="gpt-4o", backend="hermes")

# Add cross-cutting concerns
agent.use(logging_middleware())
agent.use(retry_middleware(max_retries=3))
agent.use(cache_middleware(ttl=60))  # Cache tool results for 60s

@agent.tool
def flaky_api_call(x: int) -> int:
    return x * 2  # Will be retried on failure, cached on success
```

### Cost Tracking

```python
from simpli_agent import Agent

agent = Agent(model="gpt-4o", backend="openai")
agent.run("Hello world")
agent.run("How are you?")

print(agent.usage())
# {'model': 'gpt-4o', 'total_runs': 2, 'total_prompt_tokens': 45,
#  'total_completion_tokens': 32, 'total_cost_usd': 0.00042, ...}

print(f"Total cost: ${agent.cost.total_cost:.6f}")
```

### Evaluation & Benchmarking

```python
from simpli_agent import Agent, evaluate, TestCase, benchmark

agent = Agent(model="gpt-4o", backend="hermes")
agent.tool(lambda q: f"Result for {q}", name="search")

cases = [
    TestCase(input="What's 2+2?", expected_contains="4"),
    TestCase(input="Search for Python", expected_tool="search"),
    TestCase(input="Test", custom_check=lambda out: len(out) > 10),
]

result = evaluate(agent, cases)
print(result.summary())
# Evaluation Summary
# Total: 3, Passed: 3, Failed: 0, Pass Rate: 100%

# Benchmark latency
bench = benchmark(agent, ["query1", "query2"], runs=10)
print(f"P50: {bench['median']:.3f}s, P95: {bench['p95']:.3f}s")
```

### Multi-Agent Coordination

```python
from simpli_agent import Agent, AgentTeam, AgentRegistry

researcher = Agent(tools=[search])
writer = Agent(tools=[write])

# Team with collaboration strategies
team = AgentTeam().add("researcher", researcher).add("writer", writer)

# Sequential - each builds on previous output
result = team.run_collaborative("Write about Python", strategy="sequential")

# Parallel - all work independently
result = team.run_collaborative("Analyze Python", strategy="parallel")

# Pipeline - ordered chain
result = team.run_collaborative("Create content", strategy="pipeline")

# Registry for delegation
registry = AgentRegistry().register("researcher", researcher).register("writer", writer)
registry.delegate("researcher", "writer", "Summarize findings")
registry.broadcast("Shutdown")
```

### Prompt Templates

```python
from simpli_agent import PromptTemplate, PromptBuilder, render_default_prompt

# Template with filters
template = PromptTemplate("Hello {{ name | upper }}!")
print(template.render(name="world"))  # "Hello WORLD!"

# Fluent builder
prompt = (PromptBuilder()
    .system("You are helpful")
    .user("What is 2+2?")
    .build())

# Standard agent prompt
prompt = render_default_prompt(
    system_prompt="You are a helpful assistant",
    history=[{"role": "user", "content": "Hi"}],
    tools=[{"type": "function", "function": {"name": "add", "description": "Add numbers"}}],
    task="Add 5 and 3"
)

# Pre-built templates
from simpli_agent import RESEARCH_TEMPLATE, CODING_TEMPLATE, ANALYSIS_TEMPLATE
```

### Debug REPL

```python
from simpli_agent import Agent, debug

agent = Agent(model="gpt-4o", backend="openai")
agent.tool(search_web)

# Context manager with breakpoints
with debug(agent) as dbg:
    dbg.add_breakpoint("search_web")
    dbg.add_breakpoint("search_web", "args['query'] == 'Python'")
    result = agent.run("Search for Python news")

# Post-mortem REPL
from simpli_agent import DebugREPL
repl = DebugREPL(agent, "prompt")
repl.cmdloop()
```

**REPL Commands**: `step`/`s`, `continue`/`c`, `break tool [if cond]`/`b`, `watch expr`/`w`, `inspect [agent|tools|memory|cost]`/`i`, `history [n]`/`h`, `tools`/`t`, `cost`, `breakpoints`, `clear [idx]`, `quit`/`q`

### Serialization

```python
from simpli_agent import Agent

agent = Agent(model="gpt-4o", backend="hermes", system_prompt="You are helpful")
agent.tool(lambda a, b: a + b, name="add")

agent.run("Add 2 and 3")

# Save to file
agent.save("agent.json")

# Load from file (tools must be re-registered)
agent2 = Agent.load("agent.json", backend="hermes")
# agent2 now has history, config, but tools need re-registration
```

### Streaming Tool Progress

```python
from simpli_agent import Agent, ToolProgress

agent = Agent(model="gpt-4o", backend="openai")

@agent.tool
def long_task(steps: int):
    for i in range(steps):
        yield f"Step {i+1}/{steps}"  # Progress updates
    return "Done!"  # Final result

for item in agent.stream_with_tools("Run long task with 3 steps"):
    if isinstance(item, str):
        print(f"LLM: {item}", end="", flush=True)
    elif isinstance(item, ToolProgress):
        print(f"\nTOOL [{item.name}]: {item.progress} (final={item.is_final})")
```

### Configuration File

```toml
# config.toml
[backend]
type = "openai"
model = "gpt-4o"

[memory]
db_path = "conversation.db"
semantic = true

[tools]
parallel = true
max_turns = 10

[tracing]
enabled = false

system_prompt = "You are a helpful assistant."
```

```python
from simpli_agent import load_config, create_agent_from_config

config = load_config("config.toml")
agent = create_agent_from_config(config)
```

### CLI

```bash
# Interactive chat
simpli-agent chat --backend openai --model gpt-4o

# Single prompt
simpli-agent run "What's the weather in Tokyo?" --stream

# With config
simpli-agent chat -c config.toml

# Start API server
simpli-agent serve -c config.toml --port 8000
```

## Architecture

```
simpli-agent/
├── core.py           # Agent class, tool execution, middleware, serialization
├── costs.py          # TokenUsage, CostTracker, model pricing
├── debug.py          # DebugREPL, AgentDebugger, Breakpoint
├── eval.py           # TestCase, evaluate, benchmark, Evaluator
├── middleware.py     # Built-in middleware (logging, retry, cache, etc.)
├── multiagent.py     # AgentRegistry, AgentTeam, create_team
├── pipeline.py       # Pipeline, ToolChain, pipe, compose
├── prompts.py        # PromptTemplate, PromptBuilder, templates
├── semantic_memory.py# VectorMemory, SemanticMemory
├── structured.py     # Pydantic validation, output schemas
├── tracing.py        # Tracer, CallbackHandler, OpenTelemetry
├── types.py          # ToolCall, ToolResult, ToolProgress
├── config.py         # AgentConfig, load_config
├── cli.py            # Typer CLI with chat/run/serve
├── backends/
│   ├── base.py       # Backend abstract class
│   ├── hermes.py     # Local deterministic backend
│   ├── openai.py     # OpenAI API backend
│   ├── anthropic.py  # Anthropic API backend
│   └── ollama.py     # Ollama local models
├── decorators.py     # generate_tool_schema
├── memory.py         # SQLiteMemory with FTS5
└── decorators.py     # Tool schema generation
```

## Design Philosophy

- **Minimal kernel**: Core agent loop is ~200 lines
- **Composable**: Features are independent modules that compose
- **No framework lock-in**: Pure Python functions, no DSL
- **Embeddable**: No external services required (except API backends)
- **Extensible**: Middleware, backends, templates all pluggable

## License

MIT