import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.subagent import SubAgentTool
from CAL.tool import StopTool

from conftest import FakeTool, QueueLLM, make_text_message


@pytest.mark.asyncio
async def test_agent_runs_subagent_flow():
    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub ok")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="system",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )
    parent_llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="call-1", name="delegate", input={"task": "work"})],
        ),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")]),
    ])
    memory = FullCompressionMemory()
    agent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        session_id="session",
        tools=[sub_tool],
    )

    result = await agent.run_async("prompt")

    assert result.content[0].text == "done"

    history = memory.get_history()
    tool_message = history[2]
    tool_result = tool_message.content[0]
    assert tool_message.role == MessageRole.USER
    assert tool_result.name == "delegate"
    assert tool_result.content[0].text == "sub ok"


@pytest.mark.asyncio
async def test_agent_runs_tool_end_to_end():
    tool = FakeTool("tool_one")
    llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="call-1", name="tool_one", input={"text": "one"})],
        ),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")]),
    ])
    memory = FullCompressionMemory()
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        memory=memory,
        session_id="session",
        tools=[tool],
    )

    result = await agent.run_async("prompt")

    assert result.content[0].text == "done"
    assert memory.get_history()[2].content[0].content[0].text == "one"


def test_agent_run_sync_end_to_end():
    llm = QueueLLM([Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")])])
    memory = FullCompressionMemory()
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        session_id="session",
        tools=[StopTool()],
    )

    result = agent.run("prompt")

    assert result.content[0].text == "done"
