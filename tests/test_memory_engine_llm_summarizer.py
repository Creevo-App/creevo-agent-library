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

    assert len(fake_llm.calls) >= 1, "FakeLLM should have been called for archive summarization"
    summaries = await engine.archive_store.list_summaries("thread-1", "resource-1", limit=10)
    assert summaries
    assert "[Summary]" in summaries[0].summary
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
