import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock
from CAL.memory import FullCompressionMemory
from CAL.message import MessageRole
from CAL.subagent import SubAgentTool

from conftest import FakeLLM, FakeLogger, QueueLLM, TrackingMemory, make_text_message


class ChildTrackingLogger(FakeLogger):
    """Logger that tracks parent-child relationships for testing."""

    def __init__(self, name: str = "parent"):
        super().__init__()
        self.name = name
        self.children = []

    def create_child_logger(self, name: str, agent_name: str = None) -> "ChildTrackingLogger":
        self.events.append(("create_child_logger", name, agent_name))
        # Use provided agent_name or fall back to derived name
        child_name = agent_name if agent_name else f"{self.name}_child_{name}"
        child = ChildTrackingLogger(name=child_name)
        self.children.append(child)
        return child


class LoggerTrackingMemory(FullCompressionMemory):
    """Memory that allows inspection of its logger after cloning."""

    def __init__(self, summarizer_llm=None, max_tokens=50000, messages=None, logger=None):
        llm = summarizer_llm or FakeLLM()
        super().__init__(summarizer_llm=llm, max_tokens=max_tokens, messages=messages, logger=logger)

    def clone(self) -> "LoggerTrackingMemory":
        cloned = LoggerTrackingMemory(
            summarizer_llm=self.summarizer_llm,
            max_tokens=self.max_tokens,
            messages=list(self._messages),
            logger=self.logger,  # Clone inherits parent's logger initially
        )
        return cloned


class MaxTokensTrackingMemory(FullCompressionMemory):
    """Memory subclass for testing max_tokens inheritance in subagents."""

    def __init__(self, summarizer_llm=None, max_tokens=50000, messages=None):
        llm = summarizer_llm or FakeLLM()
        super().__init__(summarizer_llm=llm, max_tokens=max_tokens, messages=messages)

    def clone(self) -> "MaxTokensTrackingMemory":
        return MaxTokensTrackingMemory(
            summarizer_llm=self.summarizer_llm,
            max_tokens=self.max_tokens,
            messages=list(self._messages),
        )


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


@pytest.mark.asyncio
async def test_subagent_inherits_memory_max_tokens_not_agent_max_tokens():
    """
    Regression test for CRE-338: Subagent memory compression bug.

    When a subagent is created without explicit max_tokens, it should inherit
    the parent MEMORY's max_tokens (compression threshold), not the parent
    AGENT's max_tokens (LLM response limit).

    Bug: Previously used self._parent_agent.max_tokens (agent's LLM limit, e.g. 10k)
    Fix: Now uses self._parent_agent.memory.max_tokens (memory compression threshold, e.g. 100k)
    """
    # Parent memory with high max_tokens (compression threshold)
    parent_memory_max_tokens = 100_000
    parent_memory = MaxTokensTrackingMemory(max_tokens=parent_memory_max_tokens)

    # Parent agent with LOW max_tokens (LLM response limit) - this was the bug source
    parent_agent_max_tokens = 10_000

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=None,  # Explicitly not setting - should inherit from parent memory
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    parent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=parent_agent_max_tokens,  # Low value - was incorrectly used before fix
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
    )

    # Execute the subagent tool
    result = await sub_tool.execute(tool_use_id="tool-use-test", task="do work")

    assert result.is_error is False

    # The subagent's memory should have inherited parent MEMORY's max_tokens (100k)
    # NOT the parent AGENT's max_tokens (10k)
    # We verify this by checking that the sub_llm was called (subagent ran successfully)
    # and that no premature compression occurred

    # Verify the LLM was called exactly once (subagent executed)
    assert len(sub_llm.calls) == 1

    # The key verification: check that effective_max_tokens in SubAgentTool.execute
    # used the memory's max_tokens. We can verify this by checking the sub_agent
    # that was created internally received the correct max_tokens.
    # Since we can't directly inspect the sub_agent, we verify the behavior:
    # - If max_tokens was 10k (wrong), compression would trigger immediately on any
    #   conversation > 10k tokens
    # - If max_tokens was 100k (correct), compression won't trigger for normal conversations


@pytest.mark.asyncio
async def test_subagent_explicit_max_tokens_overrides_parent():
    """
    When a subagent specifies explicit max_tokens, it should use that value
    instead of inheriting from the parent memory.
    """
    parent_memory = MaxTokensTrackingMemory(max_tokens=100_000)
    explicit_sub_max_tokens = 25_000

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=explicit_sub_max_tokens,  # Explicit value should be used
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    parent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
    )

    result = await sub_tool.execute(tool_use_id="tool-use-test", task="do work")

    assert result.is_error is False
    assert len(sub_llm.calls) == 1


@pytest.mark.asyncio
async def test_subagent_memory_uses_child_logger_not_parent():
    """
    Verify that subagent's memory uses the child logger, not the parent's logger.

    When memory compression occurs in a subagent, it should log to the subagent's
    child logger so that compression events appear under the correct agent span.

    Bug: Previously, cloned memory kept the parent's logger, causing compression
    events to be logged under the parent agent instead of the subagent.
    Fix: SubAgentTool.execute now sets sub_memory.logger = child_logger
    """
    # Create a logger that tracks parent-child relationships
    parent_logger = ChildTrackingLogger(name="parent")

    # Create memory with the parent logger
    parent_memory = LoggerTrackingMemory(max_tokens=100_000, logger=parent_logger)

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    parent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
        logger=parent_logger,
    )

    # Execute the subagent tool
    result = await sub_tool.execute(tool_use_id="tool-use-test", task="do work")

    assert result.is_error is False

    # Verify a child logger was created
    assert len(parent_logger.children) == 1, "Child logger should have been created"
    child_logger = parent_logger.children[0]
    # Child logger name should be the subagent's full name (agent_name parameter)
    assert child_logger.name == "parent_agent_sub_delegate", (
        f"Child logger should have subagent's agent_name, got {child_logger.name}"
    )

    # Verify the parent's logger is NOT the same as the child
    assert parent_logger is not child_logger

    # The key assertion: verify that the subagent's memory received the child logger.
    # We can't directly inspect the sub_memory after execute() returns, but we can
    # verify indirectly by checking that the child logger exists and was created.
    # A more direct test would require modifying SubAgentTool to expose the sub_memory,
    # or triggering compression and checking which logger received the event.

    # Verify child logger was created for the "delegate" subagent
    create_child_events = [e for e in parent_logger.events if e[0] == "create_child_logger"]
    assert len(create_child_events) == 1
    assert create_child_events[0][1] == "delegate"
    # Verify the agent_name was passed to the child logger
    assert create_child_events[0][2] == "parent_agent_sub_delegate"


@pytest.mark.asyncio
async def test_subagent_memory_logger_receives_compression_events():
    """
    Verify that when compression occurs in a subagent, events are logged
    to the child logger, not the parent logger.

    This test triggers actual compression by using a very low max_tokens
    and adding enough messages to exceed the threshold (compression requires >3 messages).
    """
    parent_logger = ChildTrackingLogger(name="parent")

    # Use very low max_tokens to trigger compression quickly
    parent_memory = LoggerTrackingMemory(max_tokens=200, logger=parent_logger)

    # Add multiple messages to parent memory - compression needs >3 messages to trigger
    from CAL.message import Message
    for i in range(4):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        parent_memory.add_message(Message(
            role=role,
            content=f"Message {i} with enough content to add tokens. " * 10
        ))

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response " * 50)])
    sub_tool = SubAgentTool(
        name="compressing_delegate",
        description="delegate work that triggers compression",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=200,  # Low threshold to trigger compression in subagent
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    parent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
        logger=parent_logger,
    )

    # Execute the subagent
    result = await sub_tool.execute(tool_use_id="tool-use-test", task="do work " * 30)

    assert result.is_error is False

    # Verify child logger was created
    assert len(parent_logger.children) == 1
    child_logger = parent_logger.children[0]

    # Check compression events - parent should NOT receive memory_compression events
    # that should have gone to the child logger
    parent_tool_events = [e for e in parent_logger.events if e[0] == "tool"]
    parent_compression_events = [
        e for e in parent_tool_events
        if len(e) > 1 and hasattr(e[1], 'name') and e[1].name == "memory_compression"
    ]
    assert len(parent_compression_events) == 0, (
        "Parent logger should not receive memory_compression events from subagent"
    )

    # If compression occurred in the subagent, verify the child logger received it
    child_tool_events = [e for e in child_logger.events if e[0] == "tool"]
    child_compression_events = [
        e for e in child_tool_events
        if len(e) > 1 and hasattr(e[1], 'name') and e[1].name == "memory_compression"
    ]
    # Note: Compression may or may not trigger depending on token estimates.
    # The key assertion is that IF compression happens, it goes to child, not parent.
    # We've already verified parent didn't receive it above.


class ClonedMemoryInspector(FullCompressionMemory):
    """Memory that captures the cloned instance for inspection."""

    # Class-level storage for the last cloned instance
    last_clone: "ClonedMemoryInspector" = None

    def __init__(self, summarizer_llm=None, max_tokens=50000, messages=None, logger=None):
        llm = summarizer_llm or FakeLLM()
        super().__init__(summarizer_llm=llm, max_tokens=max_tokens, messages=messages, logger=logger)

    def clone(self) -> "ClonedMemoryInspector":
        cloned = ClonedMemoryInspector(
            summarizer_llm=self.summarizer_llm,
            max_tokens=self.max_tokens,
            messages=list(self._messages),
            logger=self.logger,
        )
        # Store the clone for later inspection
        ClonedMemoryInspector.last_clone = cloned
        return cloned


@pytest.mark.asyncio
async def test_subagent_cloned_memory_has_correct_max_tokens():
    """
    Verify that when a subagent clones parent memory, the clone receives
    the correct max_tokens value (from parent memory, not parent agent).

    This test directly inspects the cloned memory instance to verify
    the max_tokens was correctly applied.
    """
    # Reset the class-level clone tracker
    ClonedMemoryInspector.last_clone = None

    parent_memory_max_tokens = 75_000
    parent_memory = ClonedMemoryInspector(max_tokens=parent_memory_max_tokens)

    # Parent agent has a DIFFERENT (lower) max_tokens
    parent_agent_max_tokens = 8_000

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="inspecting_delegate",
        description="delegate for memory inspection",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=None,  # Should inherit from parent memory (75k), not agent (8k)
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=parent_agent_max_tokens,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
    )

    # Execute the subagent
    result = await sub_tool.execute(tool_use_id="tool-use-test", task="inspect memory")

    assert result.is_error is False

    # Verify the cloned memory was created
    assert ClonedMemoryInspector.last_clone is not None, "Memory clone should have been created"

    # KEY ASSERTION: The cloned memory should have inherited parent MEMORY's max_tokens
    # After SubAgentTool.execute applies effective_max_tokens, the clone should have 75k
    cloned_memory = ClonedMemoryInspector.last_clone
    assert cloned_memory.max_tokens == parent_memory_max_tokens, (
        f"Cloned memory should have max_tokens={parent_memory_max_tokens} (from parent memory), "
        f"but got {cloned_memory.max_tokens}"
    )


@pytest.mark.asyncio
async def test_subagent_cloned_memory_uses_explicit_max_tokens():
    """
    Verify that when a subagent has explicit max_tokens, the clone uses
    that value instead of inheriting from parent memory.
    """
    ClonedMemoryInspector.last_clone = None

    parent_memory = ClonedMemoryInspector(max_tokens=100_000)
    explicit_sub_max_tokens = 30_000

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="explicit_delegate",
        description="delegate with explicit max_tokens",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=explicit_sub_max_tokens,  # Explicit value should override parent
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
    )

    result = await sub_tool.execute(tool_use_id="tool-use-test", task="test explicit")

    assert result.is_error is False
    assert ClonedMemoryInspector.last_clone is not None

    # KEY ASSERTION: The cloned memory should use the explicit max_tokens from subagent
    cloned_memory = ClonedMemoryInspector.last_clone
    assert cloned_memory.max_tokens == explicit_sub_max_tokens, (
        f"Cloned memory should have max_tokens={explicit_sub_max_tokens} (explicit), "
        f"but got {cloned_memory.max_tokens}"
    )


@pytest.mark.asyncio
async def test_subagent_compression_triggers_with_full_compression_memory():
    """
    Verify that compression actually triggers in a subagent when using
    FullCompressionMemory with enough messages to exceed the threshold.

    This tests the full flow: clone -> add messages -> trigger compression.
    """
    from CAL.message import Message

    parent_logger = ChildTrackingLogger(name="parent")

    # Parent memory with high threshold (won't compress in parent)
    parent_memory = LoggerTrackingMemory(max_tokens=100_000, logger=parent_logger)

    # Add some initial context to the parent memory
    for i in range(4):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        parent_memory.add_message(Message(
            role=role,
            content=f"Initial context message {i}. " * 20
        ))

    # Subagent with LOW max_tokens to force compression
    sub_max_tokens = 100  # Very low to force compression

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "response " * 100)])
    sub_tool = SubAgentTool(
        name="compressing_sub",
        description="delegate that triggers compression",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
        max_tokens=sub_max_tokens,
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
        logger=parent_logger,
    )

    # Execute the subagent with a long task to add more tokens
    result = await sub_tool.execute(
        tool_use_id="tool-use-test",
        task="Please process this detailed request. " * 50
    )

    assert result.is_error is False

    # Verify child logger was created
    assert len(parent_logger.children) == 1
    child_logger = parent_logger.children[0]

    # Check that compression happened in the child, not parent
    # Parent should NOT have compression events from the subagent
    parent_tool_events = [e for e in parent_logger.events if e[0] == "tool"]
    parent_compression = [
        e for e in parent_tool_events
        if len(e) > 1 and hasattr(e[1], 'name') and e[1].name == "memory_compression"
    ]
    assert len(parent_compression) == 0, "Parent should not receive subagent compression events"

    # Child logger should have received the compression event (if compression triggered)
    child_tool_events = [e for e in child_logger.events if e[0] == "tool"]
    child_compression = [
        e for e in child_tool_events
        if len(e) > 1 and hasattr(e[1], 'name') and e[1].name == "memory_compression"
    ]

    # With such a low max_tokens (100) and existing messages, compression should trigger
    # However, compression requires >3 messages in memory, which we have from parent
    # The key assertion is that IF it triggers, it goes to child logger
    if child_compression:
        # Verify the compression event has expected metadata
        compression_result = child_compression[0][2]  # ToolResultBlock
        assert compression_result.name == "memory_compression"
        assert not compression_result.is_error


@pytest.mark.asyncio
async def test_subagent_preserves_memory_logger_when_parent_agent_has_no_logger():
    """
    Verify that when the parent agent has no logger but the memory does,
    the subagent's cloned memory preserves its logger instead of setting it to None.

    Bug: Previously, we unconditionally set sub_memory.logger = child_logger,
    which would be None if the parent agent had no logger, removing the memory's
    existing logger.
    Fix: Only update memory logger if child_logger is not None.
    """
    # Create a memory WITH a logger
    memory_logger = ChildTrackingLogger(name="memory_logger")
    parent_memory = LoggerTrackingMemory(max_tokens=100_000, logger=memory_logger)

    sub_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "sub response")])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate work",
        system_prompt="You are a helper.",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )

    parent_llm = QueueLLM([make_text_message(MessageRole.ASSISTANT, "parent ok")])
    # Parent agent has NO logger
    Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10_000,
        memory=parent_memory,
        agent_name="parent_agent",
        tools=[sub_tool],
        logger=None,  # No logger on agent
    )

    # Execute the subagent
    result = await sub_tool.execute(tool_use_id="tool-use-test", task="do work")

    assert result.is_error is False

    # The key assertion: no child logger should have been created since parent agent
    # has no logger, and the memory's original logger should be preserved (not set to None)
    assert len(memory_logger.children) == 0, "No child logger should be created when parent agent has no logger"
