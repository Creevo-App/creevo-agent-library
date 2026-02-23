import json
import threading

import pytest

from CAL.agent import Agent, PROGRESS_PREFIX, emit_progress
from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.memory_engine import DefaultMemoryEngine, ContextPolicy
from CAL.message import Message, MessageRole
from CAL.tool import StopTool

from conftest import FakeTool, FakeLogger, QueueLLM, make_text_message, make_tool_use_message


def test_agent_auto_creates_memory_engine():
    """Agent with no memory_engine creates DefaultMemoryEngine automatically."""
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
    )
    assert agent.memory_engine is not None
    assert isinstance(agent.memory_engine, DefaultMemoryEngine)


def test_agent_accepts_explicit_memory_engine():
    """Agent accepts an explicitly provided memory engine."""
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    engine = DefaultMemoryEngine()
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory_engine=engine,
    )
    assert agent.memory_engine is engine


def test_find_tool_normalizes_name():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    tool = FakeTool("my-tool")
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        tools=[tool],
    )
    assert agent._find_tool("my_tool") is tool


def test_emit_progress_outputs_json(capsys):
    emit_progress("session-1", "event", "message", {"step": 1})
    output = capsys.readouterr().out.strip()
    assert output.startswith(PROGRESS_PREFIX)
    payload = json.loads(output[len(PROGRESS_PREFIX):])
    assert payload["agent_name"] == "session-1"
    assert payload["event"] == "event"


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
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=3,
        max_tokens=10,
        tools=[tool_one, tool_two, stop_tool],
    )

    result = await agent.run_async("prompt")
    assert result.content[0].name == "stop"

    history = agent.conversation_history
    roles = [m.role for m in history]
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles


@pytest.mark.asyncio
async def test_run_async_stop_tool_short_circuits():
    stop_tool = StopTool()
    tool_use_message = make_tool_use_message("stop", tool_use_id="stop-call")
    llm = QueueLLM([tool_use_message])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[stop_tool],
    )

    result = await agent.run_async("prompt")
    assert result.content[0].name == "stop"


@pytest.mark.asyncio
async def test_execute_tools_returns_error_for_missing_tool():
    tool = FakeTool("tool_one")
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        tools=[tool],
    )
    tool_uses = [
        ToolUseBlock(id="tool-1", name="tool_one", input={"text": "one"}),
        ToolUseBlock(id="tool-2", name="missing_tool", input={}),
    ]
    results = await agent._execute_tools(tool_uses)
    assert results[0].is_error is False
    assert results[1].is_error is True
    assert "not found" in results[1].content


@pytest.mark.asyncio
async def test_run_raises_when_called_in_event_loop():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
    )
    with pytest.raises(RuntimeError, match="await agent.run_async"):
        agent.run("prompt")


@pytest.mark.asyncio
async def test_push_context_appears_in_llm_call():
    stop_message = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([stop_message])
    policy = ContextPolicy(
        total_token_budget=50000,
        system_tokens=10,
        tool_tokens=10,
        buffer_tokens=100,
        working_tokens=100,
        recent_tokens=40000,
        semantic_tokens=100,
        archive_tokens=100,
    )
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[StopTool()],
        context_policy=policy,
    )
    agent.push_context("also check the tests")

    await agent.run_async("build the app")

    call_history = llm.calls[0]["history"]
    texts = [
        block.text
        for msg in call_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    assert "build the app" in texts
    assert "also check the tests" in texts


@pytest.mark.asyncio
async def test_push_context_stale_cleared_between_runs():
    agent_ref = {}

    class PushOnStopTool(StopTool):
        async def execute(self, **kwargs) -> ToolResultBlock:
            tool_use_id = kwargs.pop("tool_use_id", "stub")
            agent_ref["agent"].push_context("stale from previous run")
            return ToolResultBlock(
                tool_use_id=tool_use_id, content="STOP",
                is_error=False, name=self.name,
            )

    stop_msg_1 = make_tool_use_message("stop", tool_use_id="s1")
    stop_msg_2 = make_tool_use_message("stop", tool_use_id="s2")
    llm = QueueLLM([stop_msg_1, stop_msg_2])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[PushOnStopTool()],
    )
    agent_ref["agent"] = agent

    await agent.run_async("first run")
    await agent.run_async("second run")

    second_run_history = llm.calls[1]["history"]
    texts = [
        block.text
        for msg in second_run_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    assert "stale from previous run" not in texts


@pytest.mark.asyncio
async def test_logger_receives_trace_events():
    stop_message = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([stop_message])
    logger = FakeLogger()
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[StopTool()],
        logger=logger,
    )

    await agent.run_async("prompt")

    event_types = [e[0] for e in logger.events]
    assert "start_trace" in event_types
    assert "end_trace" in event_types
    assert "llm" in event_types
