import json
import threading

import pytest

from CAL.agent import Agent, PROGRESS_PREFIX, emit_progress
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.tool import StopTool, Tool
from CAL.content_blocks import ToolResultBlock

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


# --- push_context tests ---


@pytest.mark.asyncio
async def test_push_context_appears_in_history():
    """Pushed context is visible to the LLM as a USER message."""
    stop_message = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([stop_message])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[StopTool()],
    )

    # Push context before running — it should be drained on the first iteration
    agent.push_context("also check the tests")

    await agent.run_async("build the app")

    # The LLM should have seen both the user prompt and the pushed context
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
async def test_push_context_mid_execution():
    """Context pushed during tool execution appears in the next LLM call."""
    agent_ref = {}

    class PushingTool(Tool):
        """A tool that pushes context onto the agent when executed."""
        def __init__(self):
            async def _fn(text: str = ""):
                return {"content": [{"type": "text", "text": "ok"}], "metadata": {}}
            super().__init__(_fn)
            self.name = "pushing_tool"

        async def execute(self, **kwargs) -> ToolResultBlock:
            tool_use_id = kwargs.pop("tool_use_id", "stub")
            # Push context while tool is executing
            agent_ref["agent"].push_context("new instructions from user")
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=[TextBlock(text="done")],
                is_error=False,
                name=self.name,
            )

    tool_use_msg = Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="t1", name="pushing_tool", input={})],
    )
    stop_msg = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([tool_use_msg, stop_msg])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=5,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[PushingTool(), StopTool()],
    )
    agent_ref["agent"] = agent

    await agent.run_async("start")

    # The second LLM call should see the pushed context
    second_call_history = llm.calls[1]["history"]
    texts = [
        block.text
        for msg in second_call_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    assert "new instructions from user" in texts

    # Pushed context should be merged into the tool results message (USER role),
    # not create a consecutive USER message that breaks Gemini role alternation.
    roles = [msg.role for msg in second_call_history]
    for i in range(1, len(roles)):
        assert not (roles[i] == MessageRole.USER and roles[i - 1] == MessageRole.USER), (
            f"Consecutive USER messages at positions {i-1} and {i}"
        )


@pytest.mark.asyncio
async def test_push_context_multiple_fifo():
    """Multiple pushed contexts appear in FIFO order."""
    stop_message = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([stop_message])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[StopTool()],
    )

    agent.push_context("first")
    agent.push_context("second")

    await agent.run_async("start")

    call_history = llm.calls[0]["history"]
    texts = [
        block.text
        for msg in call_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    first_idx = texts.index("first")
    second_idx = texts.index("second")
    assert first_idx < second_idx


@pytest.mark.asyncio
async def test_push_context_thread_safe():
    """push_context works when called from a different thread."""
    agent_ref = {}
    context_received = threading.Event()

    class SlowTool(Tool):
        def __init__(self):
            async def _fn(text: str = ""):
                return {"content": [{"type": "text", "text": "ok"}], "metadata": {}}
            super().__init__(_fn)
            self.name = "slow_tool"

        async def execute(self, **kwargs) -> ToolResultBlock:
            tool_use_id = kwargs.pop("tool_use_id", "stub")
            # Signal the other thread to push, then wait briefly
            context_received.set()
            import asyncio
            await asyncio.sleep(0.05)
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=[TextBlock(text="done")],
                is_error=False,
                name=self.name,
            )

    tool_use_msg = Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="t1", name="slow_tool", input={})],
    )
    stop_msg = make_tool_use_message("stop", tool_use_id="s1")
    llm = QueueLLM([tool_use_msg, stop_msg])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=5,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[SlowTool(), StopTool()],
    )
    agent_ref["agent"] = agent

    def push_from_thread():
        context_received.wait(timeout=5)
        agent_ref["agent"].push_context("from another thread")

    t = threading.Thread(target=push_from_thread)
    t.start()

    await agent.run_async("start")
    t.join()

    # The second LLM call should include the pushed context
    second_call_history = llm.calls[1]["history"]
    texts = [
        block.text
        for msg in second_call_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    assert "from another thread" in texts


@pytest.mark.asyncio
async def test_push_context_stale_cleared_between_runs():
    """Context pushed during the final iteration (after the last drain) is
    discarded so it doesn't leak into a subsequent run_async call."""
    agent_ref = {}

    class PushOnStopTool(Tool):
        """Pushes stale context when the stop tool fires (after the last drain)."""
        def __init__(self):
            async def _fn():
                return {"content": [{"type": "text", "text": "STOP"}], "metadata": {}}
            super().__init__(_fn)
            self.name = "stop"

        async def execute(self, **kwargs) -> ToolResultBlock:
            tool_use_id = kwargs.pop("tool_use_id", "stub")
            # Simulate an external thread pushing context while stop executes
            agent_ref["agent"].push_context("stale from previous run")
            return ToolResultBlock(
                tool_use_id=tool_use_id, content="STOP",
                is_error=False, name=self.name,
            )

    stop_msg_1 = make_tool_use_message("stop", tool_use_id="s1")
    stop_msg_2 = make_tool_use_message("stop", tool_use_id="s2")
    llm = QueueLLM([stop_msg_1, stop_msg_2])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[PushOnStopTool()],
    )
    agent_ref["agent"] = agent

    await agent.run_async("first run")

    # Second run should NOT see the stale context
    await agent.run_async("second run")

    second_run_history = llm.calls[1]["history"]
    texts = [
        block.text
        for msg in second_run_history
        for block in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(block, TextBlock)
    ]
    assert "stale from previous run" not in texts


# --- get_final_response tests ---


@pytest.mark.asyncio
async def test_get_final_response_extracts_stop_tool_final_answer():
    """get_final_response extracts final_answer from stop tool call."""
    stop_tool = StopTool()
    stop_message = Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="stop-1", name="stop", input={"final_answer": "The weather in NYC is sunny."})],
    )
    llm = QueueLLM([stop_message])
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

    await agent.run_async("What's the weather?")

    response = agent.get_final_response()
    assert response == "The weather in NYC is sunny."


@pytest.mark.asyncio
async def test_get_final_response_falls_back_to_text_block():
    """get_final_response falls back to last TextBlock when no final_answer."""
    stop_tool = StopTool()
    # Message with text and stop tool call without final_answer
    stop_message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            TextBlock(text="Here is my analysis of the weather."),
            ToolUseBlock(id="stop-1", name="stop", input={}),
        ],
    )
    llm = QueueLLM([stop_message])
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

    await agent.run_async("What's the weather?")

    response = agent.get_final_response()
    assert response == "Here is my analysis of the weather."


@pytest.mark.asyncio
async def test_get_final_response_returns_empty_when_no_response():
    """get_final_response returns empty string when no response found."""
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "")])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[],
    )

    # Empty conversation - no run yet
    response = agent.get_final_response()
    assert response == ""


@pytest.mark.asyncio
async def test_get_final_response_finds_text_from_earlier_message():
    """get_final_response finds text from earlier assistant message if last has no text."""
    stop_tool = StopTool()
    tool = FakeTool("my_tool")
    
    # First message has text and tool call
    first_message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            TextBlock(text="Let me check that for you."),
            ToolUseBlock(id="t1", name="my_tool", input={"text": "query"}),
        ],
    )
    # Second message just has stop with final_answer
    stop_message = Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="stop-1", name="stop", input={"final_answer": "The result is 42."})],
    )
    llm = QueueLLM([first_message, stop_message])
    memory = FullCompressionMemory(summarizer_llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=3,
        max_tokens=10,
        memory=memory,
        agent_name="session",
        tools=[tool, stop_tool],
    )

    await agent.run_async("What is the answer?")

    # Should get the final_answer from stop tool, not the earlier text
    response = agent.get_final_response()
    assert response == "The result is 42."
