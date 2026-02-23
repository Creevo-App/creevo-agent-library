import time

import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.memory_engine import (
    ArchiveSummary,
    ContextPolicy,
    ContextRequest,
    DefaultMemoryEngine,
    InMemoryMemoryObserver,
    InMemorySemanticMemoryStore,
    MemoryScope,
    RecallQuery,
    TurnRecord,
    WorkingMemoryUpdate,
)
from CAL.message import Message, MessageRole
from CAL.subagent import SubAgentTool
from CAL.tool import StopTool
from conftest import FakeTool, QueueLLM


@pytest.mark.asyncio
async def test_context_budget_allocator_is_deterministic_and_capped():
    observer = InMemoryMemoryObserver()
    policy = ContextPolicy(
        total_token_budget=220,
        system_tokens=20,
        tool_tokens=20,
        buffer_tokens=20,
        working_tokens=40,
        recent_tokens=120,
        semantic_tokens=80,
        archive_tokens=40,
        max_recent_messages=20,
    )
    engine = DefaultMemoryEngine(context_policy=policy, observer=observer)

    run_id = "run-1"
    for i in range(10):
        msg = Message(role=MessageRole.USER, content=[TextBlock(text=f"conversation point {i} with details")])
        await engine.record_turn(
            TurnRecord(
                run_id=run_id,
                turn_id=f"t-{i}",
                thread_id="thread-1",
                resource_id="resource-1",
                message=msg,
            )
        )

    packet1 = await engine.build_context(
        ContextRequest(
            run_id=run_id,
            turn_id="ctx-1",
            agent_name="agent",
            thread_id="thread-1",
            resource_id="resource-1",
            query="details",
            policy=policy,
            system_prompt_tokens=20,
            tool_schema_tokens=20,
        )
    )
    packet2 = await engine.build_context(
        ContextRequest(
            run_id=run_id,
            turn_id="ctx-2",
            agent_name="agent",
            thread_id="thread-1",
            resource_id="resource-1",
            query="details",
            policy=policy,
            system_prompt_tokens=20,
            tool_schema_tokens=20,
        )
    )

    assert packet1.total_tokens <= policy.total_token_budget
    assert packet2.total_tokens <= policy.total_token_budget

    texts1 = [str(m.content) for m in packet1.messages]
    texts2 = [str(m.content) for m in packet2.messages]
    assert texts1 == texts2


@pytest.mark.asyncio
async def test_scope_filter_prevents_cross_resource_recall():
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(observer=observer)

    await engine.record_turn(
        TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-A",
            resource_id="resource-A",
            message=Message(role=MessageRole.USER, content=[TextBlock(text="Preferred runtime is python for service A")]),
        )
    )
    await engine.record_turn(
        TurnRecord(
            run_id="r2",
            turn_id="t2",
            thread_id="thread-B",
            resource_id="resource-B",
            message=Message(role=MessageRole.USER, content=[TextBlock(text="Preferred runtime is rust for service B")]),
        )
    )

    result = await engine.recall(
        RecallQuery(
            query="preferred runtime",
            thread_id="thread-A",
            resource_id="resource-A",
            total_limit=10,
        ),
        run_id="r3",
        turn_id="ctx-A",
    )

    assert result.items
    assert all(item.resource_id == "resource-A" for item in result.items)
    assert engine.health_snapshot().scope_leak_count == 0


@pytest.mark.asyncio
async def test_working_memory_conflict_tracking_and_versioning():
    engine = DefaultMemoryEngine()

    snap1 = await engine.update_working_memory(
        WorkingMemoryUpdate(
            scope=MemoryScope.RESOURCE,
            scope_id="resource-1",
            patch={"language": "python", "style": "pep8"},
        )
    )
    snap2 = await engine.update_working_memory(
        WorkingMemoryUpdate(
            scope=MemoryScope.RESOURCE,
            scope_id="resource-1",
            patch={"language": "typescript"},
        )
    )

    assert snap1.version == 1
    assert snap2.version == 2
    assert snap2.state["language"] == "typescript"
    assert snap2.state["style"] == "pep8"
    assert engine.health_snapshot().working_conflict_count == 1


@pytest.mark.asyncio
async def test_truncation_prefers_dropping_archive_before_semantic_and_recent():
    policy = ContextPolicy(
        total_token_budget=80,
        system_tokens=0,
        tool_tokens=0,
        buffer_tokens=0,
        working_tokens=0,
        recent_tokens=40,
        semantic_tokens=40,
        archive_tokens=40,
        max_recent_messages=10,
        max_semantic_items=4,
        max_archive_items=2,
    )
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(context_policy=policy, observer=observer)

    await engine.record_turn(
        TurnRecord(
            run_id="r1",
            turn_id="u1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(role=MessageRole.USER, content=[TextBlock(text="decision: use postgres and pgvector for memory backend")]),
        )
    )

    await engine.archive_store.store_summary(
        ArchiveSummary(
            archive_id="archive-1",
            thread_id="thread-1",
            resource_id="resource-1",
            summary="Very long archived summary that should be dropped first when memory overflows context budget.",
            turn_ids=["old1", "old2"],
            tokens_before=100,
            tokens_after=20,
            created_at=time.time(),
        )
    )

    packet = await engine.build_context(
        ContextRequest(
            run_id="r1",
            turn_id="ctx-1",
            agent_name="agent",
            thread_id="thread-1",
            resource_id="resource-1",
            query="memory backend decision",
            policy=policy,
            system_prompt_tokens=0,
            tool_schema_tokens=0,
        )
    )

    assert packet.total_tokens <= policy.total_token_budget
    if packet.truncated:
        archive_records = [
            r for r in packet.ledger.records
            if r.source_type.value == "archive" and r.source_id == "archive-1"
        ]
        assert archive_records
        assert archive_records[0].dropped_reason in {"overflow_total", "segment_budget"}


class FailingSemanticStore(InMemorySemanticMemoryStore):
    async def recall(self, query: str, scope: MemoryScope, thread_id: str, resource_id: str, limit: int):  # type: ignore[override]
        raise RuntimeError("semantic store unavailable")


@pytest.mark.asyncio
async def test_degraded_mode_fallback_when_semantic_recall_fails():
    policy = ContextPolicy(
        total_token_budget=200,
        system_tokens=0,
        tool_tokens=0,
        buffer_tokens=0,
        recent_tokens=120,
        semantic_tokens=40,
        working_tokens=20,
        archive_tokens=20,
    )
    engine = DefaultMemoryEngine(semantic_store=FailingSemanticStore(), context_policy=policy)

    await engine.record_turn(
        TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(role=MessageRole.USER, content=[TextBlock(text="hello")]),
        )
    )

    packet = await engine.build_context(
        ContextRequest(
            run_id="r1",
            turn_id="ctx-1",
            agent_name="agent",
            thread_id="thread-1",
            resource_id="resource-1",
            query="hello",
            policy=policy,
        )
    )

    assert packet.degraded_mode is True
    assert any(msg.role == MessageRole.USER for msg in packet.messages)


@pytest.mark.asyncio
async def test_v2_agent_and_subagent_isolate_threads_and_emit_ledgers():
    observer = InMemoryMemoryObserver()
    engine = DefaultMemoryEngine(observer=observer)

    sub_llm = QueueLLM([Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="sub done")])])
    sub_tool = SubAgentTool(
        name="delegate",
        description="delegate",
        system_prompt="sub system",
        tools=[],
        llm=sub_llm,
        max_calls=1,
    )

    parent_llm = QueueLLM(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=[ToolUseBlock(id="d1", name="delegate", input={"task": "do work"})],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[ToolUseBlock(id="s1", name="stop", input={})],
            ),
        ]
    )

    agent = Agent(
        llm=parent_llm,
        system_prompt="parent system",
        max_calls=4,
        max_tokens=256,
        memory_engine=engine,
        context_policy=ContextPolicy(total_token_budget=400),
        agent_name="parent",
        thread_id="thread-parent",
        resource_id="resource-main",
        tools=[sub_tool, StopTool(), FakeTool("noop")],
    )

    result = await agent.run_async("start")
    assert isinstance(result, Message)

    parent_turns = await engine.get_thread_turns("thread-parent")
    sub_turns = await engine.get_thread_turns("parent_sub_delegate")

    assert parent_turns
    assert sub_turns
    assert len(observer.context_ledgers) >= 2


@pytest.mark.asyncio
async def test_retention_cleanup_purges_raw_and_short_ttl_semantic_but_keeps_durable():
    policy = ContextPolicy(total_token_budget=200)
    engine = DefaultMemoryEngine(context_policy=policy)

    now_ts = time.time()
    await engine.record_turn(
        TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(role=MessageRole.USER, content=[TextBlock(text="We decided to use Postgres for this project")]),
            created_at=now_ts,
        )
    )

    cleanup_stats = await engine.cleanup_retention(now_ts + 31 * 24 * 60 * 60)
    assert cleanup_stats["purged_raw"] >= 1
    assert cleanup_stats["purged_semantic"] >= 1

    result = await engine.recall(
        RecallQuery(
            query="decided postgres",
            thread_id="thread-1",
            resource_id="resource-1",
            total_limit=10,
        )
    )
    assert result.items
    assert any(item.scope == MemoryScope.RESOURCE for item in result.items)
