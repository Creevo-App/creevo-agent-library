import json

import pytest

from CAL.agent import Agent, PROGRESS_PREFIX, emit_progress
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.tool import StopTool

from conftest import FakeTool, QueueLLM, make_text_message, make_tool_use_message


def test_find_tool_normalizes_name():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    memory = FullCompressionMemory(summarizer_llm=llm)
    tool = FakeTool("my-tool")
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[tool],
    )

    assert agent._find_tool("my_tool") is tool


def test_cleanup_incomplete_conversation_removes_last_assistant_tool_call():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    memory = FullCompressionMemory(summarizer_llm=llm)
    memory.add_message(make_text_message(MessageRole.USER, "hi"))
    memory.add_message(make_tool_use_message("tool"))
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        agent_name="session",
    )

    agent._cleanup_incomplete_conversation()

    history = memory.get_history()
    assert len(history) == 1
    assert history[0].role == MessageRole.USER


def test_emit_progress_outputs_json(capsys):
    emit_progress("session-1", "event", "message", {"step": 1})

    output = capsys.readouterr().out.strip()
    assert output.startswith(PROGRESS_PREFIX)

    payload = json.loads(output[len(PROGRESS_PREFIX):])
    assert payload["agent_name"] == "session-1"
    assert payload["event"] == "event"
    assert payload["message"] == "message"
    assert payload["detail"] == {"step": 1}


@pytest.mark.asyncio
async def test_run_async_executes_tools_and_records_results():
    tool_one = FakeTool("tool_one")
    tool_two = FakeTool("tool_two")
    stop_tool = StopTool()
    tool_use_message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ToolUseBlock(id="tool-1", name="tool_one", input={"text": "one"}),
            ToolUseBlock(id="tool-2", name="tool_two", input={"text": "two"}),
        ],
    )
    stop_message = Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="stop-1", name="stop", input={})],
    )
    llm = QueueLLM([tool_use_message, stop_message])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=3,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[tool_one, tool_two, stop_tool],
    )

    result = await agent.run_async("prompt")

    assert result.content[0].name == "stop"

    history = memory.get_history()
    assert [message.role for message in history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    tool_results = history[2].content
    assert tool_results[0].name == "tool_one"
    assert tool_results[0].content[0].text == "one"
    assert tool_results[1].name == "tool_two"
    assert tool_results[1].content[0].text == "two"


@pytest.mark.asyncio
async def test_execute_tools_returns_error_for_missing_tool():
    tool = FakeTool("tool_one")
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[tool],
    )
    tool_uses = [
        ToolUseBlock(id="tool-1", name="tool_one", input={"text": "one"}),
        ToolUseBlock(id="tool-2", name="missing_tool", input={}),
    ]

    results = await agent._execute_tools(tool_uses)

    assert results[0].is_error is False
    assert results[0].content[0].text == "one"
    assert results[1].is_error is True
    assert "not found" in results[1].content


@pytest.mark.asyncio
async def test_run_async_stop_tool_short_circuits():
    stop_tool = StopTool()
    tool_use_message = make_tool_use_message("stop", tool_use_id="stop-call")
    llm = QueueLLM([tool_use_message])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[stop_tool],
    )

    result = await agent.run_async("prompt")

    assert result.content[0].name == "stop"

    history = memory.get_history()
    assert len(history) == 3
    assert history[1].role == MessageRole.ASSISTANT
    assert history[2].role == MessageRole.USER


@pytest.mark.asyncio
async def test_run_raises_when_called_in_event_loop():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        agent_name="session",
    )

    with pytest.raises(RuntimeError, match="await agent.run_async"):
        agent.run("prompt")
