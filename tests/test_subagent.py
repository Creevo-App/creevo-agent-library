import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock
from CAL.message import MessageRole
from CAL.subagent import SubAgentTool

from conftest import QueueLLM, TrackingMemory, make_text_message


@pytest.mark.asyncio
async def test_subagent_unbound_returns_error():
    llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "ok")])
    tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="system",
        tools=[],
        llm=llm,
        max_calls=1,
    )

    result = await tool.execute(tool_use_id="tool-use-1", task="work")

    assert result.is_error is True
    assert "not bound" in result.content


@pytest.mark.asyncio
async def test_subagent_executes_with_cloned_memory(tracking_memory: TrackingMemory):
    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub ok")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="system",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )
    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    parent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=tracking_memory,
        agent_name="session",
        tools=[sub_tool],
    )

    result = await sub_tool.execute(tool_use_id="tool-use-2", task="work")

    assert parent is not None
    assert tracking_memory.clone_called is True
    assert result.is_error is False
    assert all(isinstance(block, TextBlock) for block in result.content)
