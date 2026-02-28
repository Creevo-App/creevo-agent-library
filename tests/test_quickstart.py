"""
Tests that verify the quick start and README code examples work correctly.

These tests use fake LLMs (no API keys needed) to validate that:
- All documented imports resolve
- Constructors accept the documented parameters
- The agent loop runs and returns a result
- Tool and subagent patterns from the README work end-to-end
"""

import os
import pytest

from conftest import QueueLLM, FakeLLM, make_text_message


# -- Quick Start imports (mirrors README exactly) --

def test_quickstart_imports():
    """All imports from the README quick start resolve."""
    from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory

    assert Agent is not None
    assert GeminiLLM is not None
    assert StopTool is not None
    assert FullCompressionMemory is not None


def test_quickstart_agent_run():
    """The quick start agent pattern runs end-to-end with a fake LLM."""
    from CAL import Agent, StopTool, FullCompressionMemory
    from CAL.content_blocks import TextBlock
    from CAL.message import MessageRole

    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "Hello! I can help with many things.")])
    summarizer_llm = FakeLLM()
    memory = FullCompressionMemory(summarizer_llm=summarizer_llm, max_tokens=50000)

    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant.",
        max_calls=1,
        max_tokens=4096,
        memory=memory,
        agent_name="my-agent",
        tools=[StopTool()]
    )

    result = agent.run("Hello, how can you help me?")

    assert result is not None
    assert result.role == MessageRole.ASSISTANT
    assert isinstance(result.content, list)
    assert len(result.content) > 0
    assert isinstance(result.content[0], TextBlock)
    assert "Hello" in result.content[0].text


# -- @tool decorator pattern (mirrors README) --

def test_tool_decorator_pattern():
    """The @tool decorator from the README creates a valid tool."""
    from CAL import tool

    @tool
    async def my_tool(param1: str, param2: int):
        """Tool description"""
        return {
            "content": [{"type": "text", "text": f"Result: {param1}, {param2}"}],
            "metadata": {}
        }

    assert my_tool.name == "my_tool"
    schema = my_tool.get_schema()
    assert schema["name"] == "my_tool"
    assert schema["description"] == "Tool description"
    assert "param1" in schema["input_schema"]["properties"]
    assert "param2" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["param1"]["type"] == "string"
    assert schema["input_schema"]["properties"]["param2"]["type"] == "integer"


@pytest.mark.asyncio
async def test_tool_decorator_executes():
    """A @tool decorated function executes and returns a ToolResultBlock."""
    from CAL import tool
    from CAL.content_blocks import TextBlock

    @tool
    async def greet(name: str):
        """Greet someone."""
        return {
            "content": [{"type": "text", "text": f"Hello, {name}!"}],
            "metadata": {}
        }

    result = await greet.execute(tool_use_id="test-1", name="World")

    assert not result.is_error
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "Hello, World!"


# -- @tool(is_read_tool=True) pattern --

def test_tool_decorator_with_options():
    """The @tool(is_read_tool=True) pattern from the README works."""
    from CAL import tool

    @tool(is_read_tool=True)
    async def read_file(path: str):
        """Read a file from disk."""
        return {
            "content": [{"type": "text", "text": f"contents of {path}"}],
            "metadata": {}
        }

    assert read_file.name == "read_file"
    assert read_file.is_read_tool is True


# -- Agent with tool calls (end-to-end loop) --

@pytest.mark.asyncio
async def test_agent_with_tool_calls():
    """Agent runs tools and returns final response."""
    from CAL import Agent, StopTool, FullCompressionMemory, tool
    from CAL.content_blocks import TextBlock, ToolUseBlock
    from CAL.message import Message, MessageRole

    @tool
    async def add(a: int, b: int):
        """Add two numbers."""
        return {
            "content": [{"type": "text", "text": str(a + b)}],
            "metadata": {}
        }

    # LLM first calls the tool, then returns a text response
    llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="call-1", name="add", input={"a": 2, "b": 3})],
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content=[TextBlock(text="The answer is 5")],
        ),
    ])
    memory = FullCompressionMemory(summarizer_llm=FakeLLM(), max_tokens=50000)

    agent = Agent(
        llm=llm,
        system_prompt="You are a calculator.",
        max_calls=2,
        max_tokens=4096,
        memory=memory,
        agent_name="calc-agent",
        tools=[StopTool(), add],
    )

    result = await agent.run_async("What is 2 + 3?")

    assert result.content[0].text == "The answer is 5"

    # Verify tool result is in history
    history = memory.get_history()
    tool_result_msg = history[2]  # user prompt, assistant tool call, tool result
    assert tool_result_msg.role == MessageRole.USER
    assert tool_result_msg.content[0].content[0].text == "5"


# -- @subagent pattern (mirrors README / wiki) --

def test_subagent_decorator_pattern():
    """The @subagent decorator from the README creates a valid SubAgentTool."""
    from CAL import subagent, tool, StopTool
    from CAL.subagent import SubAgentTool

    @tool
    async def review_file(path: str):
        """Review a file for issues."""
        return {"content": [{"type": "text", "text": "looks good"}], "metadata": {}}

    fake_llm = FakeLLM()

    @subagent(
        system_prompt="You are a code reviewer.",
        tools=[review_file],
        llm=fake_llm,
        max_calls=5,
    )
    async def code_reviewer(task: str):
        """Delegate code review to a specialized sub-agent."""
        pass

    assert isinstance(code_reviewer, SubAgentTool)
    assert code_reviewer.name == "code_reviewer"

    schema = code_reviewer.get_schema()
    assert schema["name"] == "code_reviewer"
    assert "task" in schema["input_schema"]["properties"]


# -- StopTool --

@pytest.mark.asyncio
async def test_stop_tool_ends_agent():
    """When the LLM calls stop, the agent loop ends cleanly."""
    from CAL import Agent, StopTool, FullCompressionMemory
    from CAL.content_blocks import ToolUseBlock, ToolResultBlock
    from CAL.message import Message, MessageRole

    llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="call-1", name="stop", input={})],
        ),
    ])
    memory = FullCompressionMemory(summarizer_llm=FakeLLM(), max_tokens=50000)

    agent = Agent(
        llm=llm,
        system_prompt="You are helpful.",
        max_calls=10,
        max_tokens=4096,
        memory=memory,
        agent_name="stop-test",
        tools=[StopTool()],
    )

    result = await agent.run_async("Done?")

    # The result is the message containing the stop tool call
    assert result is not None

    # Memory should contain: user prompt, assistant stop call, tool result
    history = memory.get_history()
    assert len(history) == 3
    # Last message should be the tool result for the stop call
    last = history[-1]
    assert last.role == MessageRole.USER
    assert isinstance(last.content[0], ToolResultBlock)
    assert last.content[0].name == "stop"


# -- Memory serialization round-trip --

def test_memory_serialization_roundtrip():
    """Memory can be serialized to JSON and restored."""
    from CAL import FullCompressionMemory
    from CAL.message import Message, MessageRole

    llm = FakeLLM()
    memory = FullCompressionMemory(summarizer_llm=llm, max_tokens=50000, agent_name="test")

    memory.add_message(Message(role=MessageRole.USER, content="Hello"))
    memory.add_message(Message(role=MessageRole.ASSISTANT, content="Hi there!"))

    # Serialize
    json_str = memory.to_json()
    assert isinstance(json_str, str)
    assert "Hello" in json_str

    # Restore
    restored = FullCompressionMemory.from_json(
        data=json_str,
        summarizer_llm=llm,
        agent_name="test-restored",
    )

    history = restored.get_history()
    assert len(history) == 2
    assert history[0].content == "Hello"
    assert history[1].content == "Hi there!"


# -- Memory clone isolation --

def test_memory_clone_isolation():
    """Cloned memory is fully isolated from the original."""
    from CAL import FullCompressionMemory
    from CAL.message import Message, MessageRole

    llm = FakeLLM()
    memory = FullCompressionMemory(summarizer_llm=llm, max_tokens=50000, agent_name="parent")

    memory.add_message(Message(role=MessageRole.USER, content="Original prompt"))

    clone = memory.clone()
    clone.add_message(Message(role=MessageRole.ASSISTANT, content="Clone response"))

    assert len(memory.get_history()) == 1
    assert len(clone.get_history()) == 2


# -- GeminiLLM constructor accepts all documented params --

def test_geminillm_constructor_params(monkeypatch):
    """GeminiLLM accepts all documented parameters including timeout_ms and thinking_level."""
    from CAL import GeminiLLM

    # genai.Client requires an API key — set a fake one in the environment
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    llm = GeminiLLM(
        api_key="fake-key",
        model="gemini-3-flash-preview",
        max_tokens=4096,
        timeout_ms=30_000,
        thinking_level="medium",
    )

    assert llm.name == "gemini-3-flash-preview"
    assert llm.max_tokens == 4096
    assert llm.thinking_level == "medium"
    assert llm.api_key == "fake-key"


# -- MCP imports (optional) --

def test_mcp_imports_available():
    """MCP imports are available when the mcp extra is installed."""
    try:
        from CAL.mcp import connect_mcp_server, disconnect_mcp_tools, MCPTool
        assert connect_mcp_server is not None
        assert disconnect_mcp_tools is not None
        assert MCPTool is not None
    except ImportError:
        pytest.skip("MCP extra not installed")
