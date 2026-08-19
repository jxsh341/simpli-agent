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
- **Backends**: Extensible backend abstraction with Hermes (local) and OpenAI (API) implementations
- **Context manager**: `with Agent() as agent:` for automatic resource cleanup

## Examples

### With OpenAI backend (requires `pip install simpli-agent[openai]`)

```python
from simpli_agent import Agent

agent = Agent(model="gpt-4o", backend="openai")

@agent.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}"

print(agent.run("What's the weather in Tokyo?"))
```

### Async usage

```python
import asyncio
from simpli_agent import Agent

async def main():
    agent = Agent(model="gpt-4o", backend="hermes")
    response = await agent.run_async("Hello")
    print(response)

asyncio.run(main())
```

### Streaming

```python
from simpli_agent import Agent

agent = Agent(model="gpt-4o", backend="hermes")
for chunk in agent.stream("Write a poem"):
    print(chunk, end="", flush=True)
```

### Persistent memory

```python
from simpli_agent import Agent

agent = Agent(db_path="conversation.db")
agent.run("Remember my name is Alice")
# ... later ...
agent.run("What's my name?")  # Remembers from SQLite
```

The package provides:

- `Agent`: a high-level facade for tool registration, execution, and state.
- `generate_tool_schema`: automatic conversion from Python signatures to LLM-compatible tool schemas.
- `SQLiteMemory`: embedded conversation persistence with an optional FTS5 search index.
- `Backend`, `HermesBackend`, `OpenAIBackend`: an extensible backend abstraction with local and API implementations.
