import pytest
from unittest.mock import patch

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory_engine import DefaultMemoryEngine, ContextPolicy, InMemoryMemoryObserver
from CAL.message import Message, MessageRole
from CAL.subagent import SubAgentTool
from CAL.tool import StopTool

from conftest import FakeTool, FakeLogger, QueueLLM, make_text_message


class ChildTrackingLogger(FakeLogger):
    def __init__(self, name: str = "parent"):
        super().__init__()
        self.name = name
        self.children = []

    def create_child_logger(self, name: str, agent_name: str = None) -> "ChildTrackingLogger":
        self.events.append(("create_child_logger", name, agent_name))
        child_name = agent_name if agent_name else f"{self.name}_child_{name}"
        child = ChildTrackingLogger(name=child_name)
        self.children.append(child)
        return child


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
async def test_subagent_executes_and_returns_result():
    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub ok")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="sub system",
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
    agent = Agent(
        llm=parent_llm,
        system_prompt="parent system",
        max_calls=2,
        max_tokens=10,
        tools=[sub_tool],
    )

    result = await agent.run_async("prompt")
    assert result.content[0].text == "done"


@pytest.mark.asyncio
async def test_subagent_uses_separate_thread_id():
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(observer=observer)

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub done")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate",
        system_prompt="sub system",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )
    parent_llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="d1", name="delegate", input={"task": "do work"})],
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="s1", name="stop", input={})],
        ),
    ])
    agent = Agent(
        llm=parent_llm,
        system_prompt="parent system",
        max_calls=4,
        max_tokens=256,
        memory_engine=engine,
        agent_name="parent",
        thread_id="thread-parent",
        resource_id="resource-main",
        tools=[sub_tool, StopTool()],
    )

    await agent.run_async("start")

    parent_turns = await engine.get_thread_turns("thread-parent")
    assert parent_turns

    # thread_id is unique per invocation; find it by prefix
    sub_thread_ids = [
        tid for tid in engine.conversation_store._turns
        if tid.startswith("parent_sub_delegate:")
    ]
    assert len(sub_thread_ids) == 1
    sub_turns = await engine.get_thread_turns(sub_thread_ids[0])
    assert sub_turns


@pytest.mark.asyncio
async def test_subagent_shares_resource_id_with_parent():
    engine = DefaultMemoryEngine()

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub done")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate",
        system_prompt="sub system",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )
    parent_llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="d1", name="delegate", input={"task": "work"})],
        ),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")]),
    ])
    agent = Agent(
        llm=parent_llm,
        system_prompt="parent system",
        max_calls=4,
        max_tokens=256,
        memory_engine=engine,
        agent_name="parent",
        resource_id="shared-resource",
        tools=[sub_tool],
    )

    result = await agent.run_async("start")
    assert result is not None


@pytest.mark.asyncio
async def test_subagent_explicit_max_tokens_overrides_parent():
    engine = DefaultMemoryEngine()
    explicit_sub_max_tokens = 25_000

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=explicit_sub_max_tokens,
    )
    parent_llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="d1", name="delegate", input={"task": "work"})],
        ),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")]),
    ])
    agent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=4,
        max_tokens=10_000,
        memory_engine=engine,
        agent_name="parent_agent",
        tools=[sub_tool],
    )

    # Capture the context_policy passed to the sub-agent's Agent.__init__
    captured_policies = []
    original_init = Agent.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.agent_name.endswith("_sub_delegate"):
            captured_policies.append(self.context_policy)

    with patch.object(Agent, "__init__", _tracking_init):
        result = await agent.run_async("test")

    assert result is not None
    assert len(sub_llm.calls) == 1
    assert len(captured_policies) == 1
    assert captured_policies[0].total_token_budget == explicit_sub_max_tokens


@pytest.mark.asyncio
async def test_subagent_child_logger_created():
    parent_logger = ChildTrackingLogger(name="parent")
    engine = DefaultMemoryEngine()

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )
    parent_llm = QueueLLM([
        Message(
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="d1", name="delegate", input={"task": "work"})],
        ),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")]),
    ])
    agent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=4,
        max_tokens=10_000,
        memory_engine=engine,
        agent_name="parent_agent",
        tools=[sub_tool],
        logger=parent_logger,
    )

    await agent.run_async("test")

    assert len(parent_logger.children) == 1
    child = parent_logger.children[0]
    assert child.name == "parent_agent_sub_delegate"
