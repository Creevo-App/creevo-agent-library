# Remove Legacy Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all legacy `Memory`/`FullCompressionMemory` backwards compatibility, making `DefaultMemoryEngine` the only memory system, and port LLM-based summarization into the v2 engine.

**Architecture:** The dual-path agent (legacy Memory vs v2 MemoryEngine) collapses to a single v2 path. `DefaultMemoryEngine` gains optional `summarizer_llm` and `archiver` params ported from the legacy compression flow. All tests rewritten for v2.

**Tech Stack:** Python, pytest, asyncio

---

### Task 1: Port LLM Summarization Into DefaultMemoryEngine

**Files:**
- Modify: `src/CAL/memory_engine.py:807-850` (constructor), `src/CAL/memory_engine.py:1369-1456` (archive + summarize methods)

**Step 1: Write the failing test**

Create `tests/test_memory_engine_llm_summarizer.py`:

```python
import pytest

from CAL.content_blocks import TextBlock
from CAL.memory_engine import (
    ContextPolicy,
    DefaultMemoryEngine,
    InMemoryMemoryObserver,
    TurnRecord,
)
from CAL.message import Message, MessageRole
from conftest import FakeLLM


@pytest.mark.asyncio
async def test_llm_summarizer_used_when_archiving_cold_history():
    """When summarizer_llm is provided, archive summaries use LLM output."""
    fake_llm = FakeLLM()
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(
        summarizer_llm=fake_llm,
        observer=observer,
        archive_cold_threshold=8,
        archive_keep_recent=4,
    )

    for i in range(12):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        await engine.record_turn(
            TurnRecord(
                run_id="r1",
                turn_id=f"t-{i}",
                thread_id="thread-1",
                resource_id="resource-1",
                message=Message(role=role, content=[TextBlock(text=f"Turn {i} content with enough words to matter")]),
            )
        )

    # LLM summarizer should have been called
    assert len(fake_llm.calls) >= 1, "FakeLLM should have been called for archive summarization"

    # Archive should contain LLM output
    summaries = await engine.archive_store.list_summaries("thread-1", "resource-1", limit=10)
    assert summaries
    assert "[Summary]" in summaries[0].summary

    # Observer should have recorded compression
    assert len(observer.compression_events) >= 1


@pytest.mark.asyncio
async def test_naive_summarizer_used_when_no_llm_provided():
    """When summarizer_llm is None, archive uses naive text truncation."""
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(
        summarizer_llm=None,
        observer=observer,
        archive_cold_threshold=8,
        archive_keep_recent=4,
    )

    for i in range(12):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        await engine.record_turn(
            TurnRecord(
                run_id="r1",
                turn_id=f"t-{i}",
                thread_id="thread-1",
                resource_id="resource-1",
                message=Message(role=role, content=[TextBlock(text=f"Turn {i} content with enough words to matter")]),
            )
        )

    summaries = await engine.archive_store.list_summaries("thread-1", "resource-1", limit=10)
    assert summaries
    # Naive summarizer produces "Archived context summary:" prefix
    assert "Archived context summary:" in summaries[0].summary


@pytest.mark.asyncio
async def test_archiver_writes_to_disk_when_provided():
    """When archiver is provided, archive content is written to file."""
    from CAL.compression import CompressionArchiver
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        archiver = CompressionArchiver(agent_name="test_agent", base_dir=tmpdir)
        engine = DefaultMemoryEngine(
            archiver=archiver,
            archive_cold_threshold=8,
            archive_keep_recent=4,
        )

        for i in range(12):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await engine.record_turn(
                TurnRecord(
                    run_id="r1",
                    turn_id=f"t-{i}",
                    thread_id="thread-1",
                    resource_id="resource-1",
                    message=Message(role=role, content=[TextBlock(text=f"Turn {i} content with enough words")]),
                )
            )

        assert archiver.has_archived_context(), "Archiver should have written context files"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_memory_engine_llm_summarizer.py -v`
Expected: FAIL — `DefaultMemoryEngine` does not accept `summarizer_llm` or `archiver` params.

**Step 3: Implement LLM summarization in DefaultMemoryEngine**

In `src/CAL/memory_engine.py`:

1. Add import at top (after existing imports):
```python
from .llm import LLM
```

2. Update `DefaultMemoryEngine.__init__` (lines 807-850) to accept new params:
```python
def __init__(
    self,
    conversation_store: Optional[ConversationStore] = None,
    working_store: Optional[WorkingMemoryStore] = None,
    semantic_store: Optional[SemanticMemoryStore] = None,
    archive_store: Optional[ArchiveStore] = None,
    context_policy: Optional[ContextPolicy] = None,
    retention_policy: Optional[RetentionPolicy] = None,
    observer: Optional[MemoryObserver] = None,
    pre_write_processors: Optional[List[PreWriteProcessor]] = None,
    pre_recall_processors: Optional[List[PreRecallProcessor]] = None,
    post_recall_processors: Optional[List[PostRecallProcessor]] = None,
    archive_cold_threshold: int = 60,
    archive_keep_recent: int = 24,
    summarizer_llm: Optional[LLM] = None,
    archiver: Optional[Any] = None,
):
    # ... existing init code ...
    self.summarizer_llm = summarizer_llm
    self.archiver = archiver
```

3. Replace `_summarize_turns` method (lines 1447-1456) and update `_maybe_archive_cold_history` (lines 1369-1414) to use LLM when available:

Replace `_summarize_turns`:
```python
def _summarize_turns(self, turns: List[TurnRecord]) -> str:
    """Summarize turns using naive text truncation (no LLM)."""
    lines = ["Archived context summary:"]
    for turn in turns[-8:]:
        role = turn.message.role.value
        content = _message_to_text(turn.message)
        compact = content.replace("\n", " ").strip()
        if len(compact) > 180:
            compact = compact[:177] + "..."
        lines.append(f"- {role}: {compact}")
    return "\n".join(lines)

def _summarize_turns_with_llm(self, turns: List[TurnRecord]) -> str:
    """Summarize turns using LLM-based compression."""
    summarization_prompt = (
        "You are a conversation summarizer. Analyze the conversation history "
        "and produce a comprehensive summary covering key actions, decisions, "
        "tool usage, and outcomes. Include important context that would be "
        "needed to continue the conversation. Return plain text, not JSON."
    )
    text_parts = []
    for turn in turns:
        role = turn.message.role.value
        content = _message_to_text(turn.message)
        text_parts.append(f"{role}: {content}")

    text_history = "\n\n".join(text_parts)
    summary_request = Message(
        role=MessageRole.USER,
        content=f"Summarize the following conversation:\n\n{text_history}",
    )
    response = self.summarizer_llm.generate_content(
        system_prompt=summarization_prompt,
        conversation_history=[summary_request],
        tools=None,
    )
    if isinstance(response.content, str):
        return response.content
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)
```

In `_maybe_archive_cold_history`, replace the summarization call:
```python
# Replace: summary_text = self._summarize_turns(candidate_turns)
if self.summarizer_llm is not None:
    try:
        summary_text = self._summarize_turns_with_llm(candidate_turns)
    except Exception:
        summary_text = self._summarize_turns(candidate_turns)
else:
    summary_text = self._summarize_turns(candidate_turns)
```

After `await self.archive_store.store_summary(summary)`, add archiver support:
```python
if self.archiver is not None:
    try:
        archive_content = "\n\n".join(
            f"{t.message.role.value}: {_message_to_text(t.message)}" for t in candidate_turns
        )
        tools_used = []
        for t in candidate_turns:
            text = _message_to_text(t.message)
            if "[ToolUse " in text:
                import re as _re
                names = _re.findall(r"\[ToolUse (\S+)\]", text)
                tools_used.extend(names)
        self.archiver.write_context_file(
            filename=archive_id,
            content=archive_content,
            message_range=f"{len(candidate_turns)} turns",
            tools_used=list(set(tools_used)),
            key_files=[],
            summary=summary_text,
        )
    except Exception:
        pass  # Best-effort file archival
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_memory_engine_llm_summarizer.py -v`
Expected: PASS

**Step 5: Run existing memory engine tests for regressions**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_memory_engine_v2.py -v`
Expected: PASS (naive summarizer behavior unchanged)

**Step 6: Commit**

```bash
git add tests/test_memory_engine_llm_summarizer.py src/CAL/memory_engine.py
git commit -m "feat(memory-engine): add optional LLM summarizer and file archiver to DefaultMemoryEngine"
```

---

### Task 2: Remove Legacy Memory From Agent

**Files:**
- Modify: `src/CAL/agent.py`

**Step 1: Write the failing test — v2-only Agent constructor**

Create `tests/test_agent_v2.py`:

```python
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
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[StopTool()],
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_agent_v2.py -v`
Expected: Most tests PASS (agent already supports v2 path), but confirms v2 works standalone.

**Step 3: Strip legacy code from agent.py**

In `src/CAL/agent.py`:

1. Remove `from .memory import Memory` import (line 17 area)
2. Remove `memory` param from `__init__` signature (line 56)
3. Remove `self.memory = memory` (line 71)
4. Remove all conditionals that check `self.memory` — simplify to v2-only:
   - `__init__`: Remove `if self.memory_engine is None and self.memory is None:` guard; always create engine if not provided
   - `conversation_history`: Remove legacy branch, always return `list(self._latest_thread_messages)`
   - `_history_json`: Already uses `self.conversation_history` — just remove any legacy refs
5. Delete `_cleanup_incomplete_conversation` method entirely (lines 213-224)
6. Delete `_run_async_legacy` method entirely (lines 458-651)
7. Rename `_run_async_v2` to be the body of `run_async` (merge them):
   - Remove the routing dispatcher (lines 653-660)
   - `run_async` directly contains what was `_run_async_v2`
8. `_extract_final_output`: Already works with `self.conversation_history`— no changes needed

**Step 4: Run tests to verify they pass**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_agent_v2.py tests/test_memory_engine_v2.py tests/test_memory_engine_llm_summarizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/CAL/agent.py tests/test_agent_v2.py
git commit -m "refactor(agent): remove legacy memory path, v2 memory engine is now the only execution path"
```

---

### Task 3: Remove Legacy Memory From SubAgent

**Files:**
- Modify: `src/CAL/subagent.py`

**Step 1: Write the failing test — v2-only SubAgent**

Create `tests/test_subagent_v2.py`:

```python
import pytest

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
    sub_turns = await engine.get_thread_turns("parent_sub_delegate")
    assert parent_turns
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

    result = await agent.run_async("test")
    assert result is not None
    assert len(sub_llm.calls) == 1


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
```

**Step 2: Run tests to verify they fail/pass baseline**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_subagent_v2.py -v`
Expected: Tests should pass since agent already supports v2.

**Step 3: Strip legacy code from subagent.py**

In `src/CAL/subagent.py`:

1. Remove the entire legacy `else:` branch (lines 121-144) — the one that clones parent memory, sets archiver, sets logger on clone
2. Remove the `if self._parent_agent.memory_engine is not None:` condition — the v2 block becomes the only path
3. Remove `elif self._parent_agent.memory is not None and hasattr(self._parent_agent.memory, 'max_tokens'):` fallback for effective_max_tokens — replace with just `self._parent_agent.max_tokens` as fallback
4. Clean up imports — remove any unused imports from `.memory`

**Step 4: Run ALL tests to verify**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/test_subagent_v2.py tests/test_agent_v2.py tests/test_memory_engine_v2.py tests/test_memory_engine_llm_summarizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/CAL/subagent.py tests/test_subagent_v2.py
git commit -m "refactor(subagent): remove legacy memory clone path, use v2 memory engine only"
```

---

### Task 4: Remove Legacy Exports and Delete memory.py

**Files:**
- Modify: `src/CAL/__init__.py`
- Delete: `src/CAL/memory.py`
- Delete: `tests/test_memory.py`
- Delete: `tests/test_agent.py` (replaced by `test_agent_v2.py`)
- Delete: `tests/test_subagent.py` (replaced by `test_subagent_v2.py`)

**Step 1: Update `__init__.py` exports**

In `src/CAL/__init__.py`:
1. Remove line 11: `from .memory import Memory, FullCompressionMemory`
2. Remove from `__all__`: `'Memory'`, `'FullCompressionMemory'` (lines 69-70)

**Step 2: Delete legacy files**

```bash
rm src/CAL/memory.py
rm tests/test_memory.py
rm tests/test_agent.py
rm tests/test_subagent.py
```

**Step 3: Update conftest.py — remove TrackingMemory**

In `tests/conftest.py`:
1. Remove `from CAL.memory import FullCompressionMemory` (line 19)
2. Remove `TrackingMemory` class entirely (lines 103-111)
3. Remove `tracking_memory` fixture (lines 137-139)

**Step 4: Update test_integration.py for v2**

Rewrite `tests/test_integration.py`:

```python
import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory_engine import DefaultMemoryEngine
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
    agent = Agent(
        llm=parent_llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[sub_tool],
    )

    result = await agent.run_async("prompt")
    assert result.content[0].text == "done"


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
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=2,
        max_tokens=10,
        tools=[tool],
    )

    result = await agent.run_async("prompt")
    assert result.content[0].text == "done"


def test_agent_run_sync_end_to_end():
    llm = QueueLLM([Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")])])
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        tools=[StopTool()],
    )
    result = agent.run("prompt")
    assert result.content[0].text == "done"
```

**Step 5: Run full test suite**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/ -v`
Expected: PASS — no test imports `Memory` or `FullCompressionMemory`

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy Memory/FullCompressionMemory and associated tests"
```

---

### Task 5: Update __init__.py Exports and migration module references

**Files:**
- Modify: `src/CAL/__init__.py`
- Modify: `src/CAL/migrations/migrate_legacy_memory.py` (if it imports from memory.py)

**Step 1: Verify migration module doesn't import from deleted memory.py**

Check `src/CAL/migrations/migrate_legacy_memory.py` — it imports from `..content_blocks`, `..memory_engine`, `..message`. It does NOT import from `..memory`. No changes needed.

**Step 2: Verify __init__.py is clean**

Ensure no remaining references to `Memory` or `FullCompressionMemory` in imports or `__all__`.

**Step 3: Run full test suite**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/ -v`
Expected: PASS

**Step 4: Commit (if any changes)**

```bash
git add -A
git commit -m "chore: verify migration module and exports after legacy removal"
```

---

### Task 6: Update CLAUDE.md and README

**Files:**
- Modify: `.claude/CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Remove references to:
- "Use `FullCompressionMemory` for long-running agent tasks"
- "Custom Memory implementations MUST implement `clone()` method"
- The SubAgent section about memory cloning

Replace with v2 memory engine guidance:
- `DefaultMemoryEngine` is the memory system
- Optional `summarizer_llm` for LLM-based archive compression
- SubAgents share parent's engine with separate `thread_id`

**Step 2: Update README.md**

Remove:
- Legacy `FullCompressionMemory` backward compatibility examples
- Any references to the `memory=` parameter on Agent

Update examples to show v2-only usage.

**Step 3: Commit**

```bash
git add .claude/CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for v2-only memory engine"
```

---

### Task 7: Final Verification

**Step 1: Run full test suite**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Verify no dangling imports**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && grep -r "from.*memory import.*Memory\|from.*memory import.*FullCompression\|import FullCompression" src/ tests/`
Expected: No matches (only `memory_engine` imports remain)

**Step 3: Verify no references to deleted self.memory**

Run: `cd /Users/atarkian2/Documents/GitHub/Creevo-App/creevo-agent-library && grep -rn "self\.memory[^_]" src/CAL/agent.py src/CAL/subagent.py`
Expected: Only `self.memory_engine` references

**Step 4: Commit final state**

If any fixes were needed, commit them:
```bash
git add -A
git commit -m "chore: final cleanup after legacy memory removal"
```
