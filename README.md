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

The package provides:

- `Agent`: a high-level facade for tool registration, execution, and state.
- `generate_tool_schema`: automatic conversion from Python signatures to LLM-compatible tool schemas.
- `SQLiteMemory`: embedded conversation persistence with an optional FTS5 search index.
- `Backend` and `HermesBackend`: an extensible backend abstraction with a deterministic local Hermes placeholder.
