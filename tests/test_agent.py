from simpli_agent import Agent, generate_tool_schema
from simpli_agent.memory import SQLiteMemory


def test_generate_tool_schema_marks_required_and_types():
    def greet(name: str, excited: bool = False) -> str:
        """Greet a person."""
        return f"Hello {name}{'!' if excited else '.'}"

    schema = generate_tool_schema(greet)

    function = schema["function"]
    assert function["name"] == "greet"
    assert function["description"] == "Greet a person."
    assert function["parameters"]["required"] == ["name"]
    assert function["parameters"]["properties"]["name"]["type"] == "string"
    assert function["parameters"]["properties"]["excited"]["type"] == "boolean"


def test_generate_tool_schema_optional_and_literal():
    from typing import Optional, Literal

    def configure(mode: Literal["fast", "slow"], timeout: Optional[int] = 30) -> str:
        """Configure settings.

        Args:
            mode: Operation mode.
            timeout: Timeout in seconds.
        """
        return f"{mode}:{timeout}"

    schema = generate_tool_schema(configure)
    props = schema["function"]["parameters"]["properties"]

    assert props["mode"]["type"] == "string"
    assert props["mode"]["enum"] == ["fast", "slow"]
    assert props["timeout"]["type"] == "integer"
    assert props["timeout"]["nullable"] is True
    assert schema["function"]["parameters"]["required"] == ["mode"]


def test_agent_registers_tool_and_persists_history():
    agent = Agent(model="test-model")

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    response = agent.run("Add 2 and 3")

    assert "test-model" in response
    assert "1 tool(s)" in response
    assert agent.tools["add"] is add
    assert [message["role"] for message in agent.history()] == ["user", "assistant"]


def test_memory_search_finds_messages():
    agent = Agent()
    agent.run("Remember the launch code is bluebird")

    results = agent.search_memory("bluebird")

    assert results
    assert results[0]["role"] == "user"


def test_agent_context_manager():
    with Agent() as agent:
        agent.run("hello")
    assert agent.memory.connection is not None  # connection still exists but closed


def test_agent_system_prompt():
    agent = Agent(system_prompt="You are a helpful assistant.")
    assert agent.system_prompt == "You are a helpful assistant."


def test_agent_stream():
    agent = Agent(model="test-model")
    chunks = list(agent.stream("test"))
    assert len(chunks) > 0


def test_memory_persistence():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name

    try:
        agent1 = Agent(db_path=db_path)
        agent1.run("Remember: the secret is 42")
        agent1.close()

        agent2 = Agent(db_path=db_path)
        results = agent2.search_memory("secret")
        assert results
        assert "42" in results[0]["content"]
        agent2.close()
    finally:
        os.unlink(db_path)


def test_memory_fts_fallback():
    memory = SQLiteMemory(":memory:")
    # Force FTS to be disabled by corrupting the vtable
    memory.connection.execute("DROP TABLE IF EXISTS history_fts")
    memory.fts_enabled = False

    memory.add_message("user", "test content")
    results = memory.search("content")
    assert results
    assert results[0]["content"] == "test content"
    memory.close()


import asyncio


async def test_agent_async():
    agent = Agent(model="test-model")
    response = await agent.run_async("test async")
    assert "test-model" in response


async def test_agent_stream_async():
    agent = Agent(model="test-model")
    chunks = []
    async for chunk in agent.stream_async("test"):
        chunks.append(chunk)
    assert len(chunks) > 0


def test_async_functionality():
    asyncio.run(test_agent_async())
    asyncio.run(test_agent_stream_async())
