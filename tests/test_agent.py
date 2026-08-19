from simpli_agent import Agent, generate_tool_schema


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
