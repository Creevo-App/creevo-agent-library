"""Tests for memory observers (Null, Composite, ClickHouse) and processors
(Redaction, QueryExpansion, Dedupe)."""

import time

import pytest

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.memory_engine import (
    ClickHouseMemoryObserver,
    CompositeMemoryObserver,
    CompressionEvent,
    ContextLedgerEntry,
    ContextLedgerRecord,
    ContextSourceType,
    DedupeRecallProcessor,
    InMemoryMemoryObserver,
    MemoryScope,
    MemoryWriteEvent,
    NullMemoryObserver,
    QueryExpansionProcessor,
    RecallEvent,
    RecallItem,
    RedactionProcessor,
    TurnRecord,
)
from CAL.message import Message, MessageRole


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_turn(text: str, turn_id: str = "t1") -> TurnRecord:
    return TurnRecord(
        run_id="r1",
        turn_id=turn_id,
        thread_id="thread-1",
        resource_id="resource-1",
        message=Message(role=MessageRole.USER, content=[TextBlock(text=text)]),
    )


def _make_ledger() -> ContextLedgerEntry:
    return ContextLedgerEntry(
        run_id="r1",
        turn_id="t1",
        agent_name="agent",
        thread_id="thread-1",
        resource_id="resource-1",
        records=[
            ContextLedgerRecord(
                source_type=ContextSourceType.RECENT,
                source_id="src-1",
                scope="thread",
                rank=0,
                relevance_score=1.0,
                confidence=1.0,
                tokens_used=10,
                tokens_reserved=20,
            ),
        ],
        retrieval_latency_ms=1.0,
        assembly_latency_ms=1.0,
        overflowed=False,
        timestamp=time.time(),
    )


def _make_write_event() -> MemoryWriteEvent:
    return MemoryWriteEvent(
        run_id="r1",
        turn_id="t1",
        thread_id="thread-1",
        resource_id="resource-1",
        status="ok",
        memory_type="fact",
        scope="thread",
        token_count=5,
    )


def _make_recall_event() -> RecallEvent:
    return RecallEvent(
        run_id="r1",
        turn_id="t1",
        thread_id="thread-1",
        resource_id="resource-1",
        query="test",
        latency_ms=2.0,
        hit_count=1,
        degraded_mode=False,
        scope_leak_count=0,
    )


def _make_compression_event() -> CompressionEvent:
    return CompressionEvent(
        run_id="r1",
        thread_id="thread-1",
        resource_id="resource-1",
        archived_turn_count=3,
        tokens_before=100,
        tokens_after=20,
        ratio=0.2,
        fidelity_score=0.8,
        archive_id="archive-test",
    )


def _make_recall_item(text: str, item_id: str = "i1") -> RecallItem:
    return RecallItem(
        item_id=item_id,
        text=text,
        scope=MemoryScope.THREAD,
        thread_id="thread-1",
        resource_id="resource-1",
        memory_type="fact",
        confidence=1.0,
        score=0.9,
        created_at=time.time(),
        expires_at=time.time() + 3600,
    )


# ===========================================================================
# NullMemoryObserver
# ===========================================================================


class TestNullMemoryObserver:
    def test_all_methods_are_noop(self):
        obs = NullMemoryObserver()
        # Should not raise
        obs.on_context_built(_make_ledger())
        obs.on_memory_write(_make_write_event())
        obs.on_recall(_make_recall_event())
        obs.on_compression(_make_compression_event())


# ===========================================================================
# CompositeMemoryObserver
# ===========================================================================


class TestCompositeMemoryObserver:
    def test_fans_out_to_all_observers(self):
        obs_a = InMemoryMemoryObserver()
        obs_b = InMemoryMemoryObserver()
        composite = CompositeMemoryObserver([obs_a, obs_b])

        ledger = _make_ledger()
        write_event = _make_write_event()
        recall_event = _make_recall_event()
        compression_event = _make_compression_event()

        composite.on_context_built(ledger)
        composite.on_memory_write(write_event)
        composite.on_recall(recall_event)
        composite.on_compression(compression_event)

        for obs in [obs_a, obs_b]:
            assert len(obs.context_ledgers) == 1
            assert obs.context_ledgers[0] is ledger
            assert len(obs.write_events) == 1
            assert len(obs.recall_events) == 1
            assert len(obs.compression_events) == 1

    def test_empty_composite_does_not_raise(self):
        composite = CompositeMemoryObserver([])
        composite.on_context_built(_make_ledger())
        composite.on_memory_write(_make_write_event())
        composite.on_recall(_make_recall_event())
        composite.on_compression(_make_compression_event())


# ===========================================================================
# ClickHouseMemoryObserver
# ===========================================================================


class TestClickHouseMemoryObserver:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="http\\(s\\) URL"):
            ClickHouseMemoryObserver(endpoint="file:///tmp/data")

    def test_rejects_empty_netloc(self):
        with pytest.raises(ValueError, match="http\\(s\\) URL"):
            ClickHouseMemoryObserver(endpoint="http://")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="http\\(s\\) URL"):
            ClickHouseMemoryObserver(endpoint="ftp://example.com/data")

    def test_accepts_http(self):
        obs = ClickHouseMemoryObserver(endpoint="http://localhost:8123")
        assert obs.endpoint == "http://localhost:8123"

    def test_accepts_https(self):
        obs = ClickHouseMemoryObserver(endpoint="https://ch.example.com:8443/")
        assert obs.endpoint == "https://ch.example.com:8443"

    def test_strips_trailing_slash(self):
        obs = ClickHouseMemoryObserver(endpoint="http://localhost:8123///")
        assert obs.endpoint == "http://localhost:8123"

    def test_post_silently_handles_connection_errors(self):
        """_post should swallow URLError for unreachable hosts."""
        obs = ClickHouseMemoryObserver(
            endpoint="http://192.0.2.1:1",  # RFC 5737 TEST-NET, unreachable
            timeout_seconds=0.1,
        )
        # Should not raise
        obs._post("test_table", {"key": "value"})

    def test_event_methods_do_not_raise_on_connection_failure(self):
        obs = ClickHouseMemoryObserver(
            endpoint="http://192.0.2.1:1",
            timeout_seconds=0.1,
        )
        obs.on_context_built(_make_ledger())
        obs.on_memory_write(_make_write_event())
        obs.on_recall(_make_recall_event())
        obs.on_compression(_make_compression_event())


# ===========================================================================
# RedactionProcessor
# ===========================================================================


class TestRedactionProcessor:
    def test_redacts_openai_key(self):
        proc = RedactionProcessor()
        turn = _make_turn("My key is sk-abc1234567890ABCDEF")
        result = proc.process(turn)
        assert "sk-abc" not in result.message.content[0].text
        assert "[REDACTED_SECRET]" in result.message.content[0].text

    def test_redacts_google_api_key(self):
        proc = RedactionProcessor()
        turn = _make_turn("API key: AIzaSyA1234567890abcdefghij")
        result = proc.process(turn)
        assert "AIzaSy" not in result.message.content[0].text
        assert "[REDACTED_SECRET]" in result.message.content[0].text

    def test_redacts_aws_access_key(self):
        proc = RedactionProcessor()
        turn = _make_turn("Creds: AKIAIOSFODNN7EXAMPLE")
        result = proc.process(turn)
        assert "AKIA" not in result.message.content[0].text
        assert "[REDACTED_SECRET]" in result.message.content[0].text

    def test_no_secrets_returns_same_turn(self):
        proc = RedactionProcessor()
        turn = _make_turn("Just normal text, no secrets here")
        result = proc.process(turn)
        assert result is turn  # identity — no copy needed

    def test_preserves_metadata(self):
        proc = RedactionProcessor()
        turn = TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(
                role=MessageRole.USER,
                content=[TextBlock(text="key: sk-abc1234567890ABCDEF")],
                metadata={"tag": "sensitive"},
            ),
            metadata={"source": "user"},
        )
        result = proc.process(turn)
        assert result.metadata == {"source": "user"}
        assert result.message.metadata == {"tag": "sensitive"}

    def test_redacts_multiple_secrets_in_one_message(self):
        proc = RedactionProcessor()
        turn = _make_turn("OpenAI: sk-abc1234567890ABCDEF, AWS: AKIAIOSFODNN7EXAMPLE")
        result = proc.process(turn)
        text = result.message.content[0].text
        assert text.count("[REDACTED_SECRET]") == 2

    def test_preserves_tool_blocks_while_redacting_text_blocks(self):
        """RedactionProcessor must preserve ToolUseBlock/ToolResultBlock
        structure and only redact secrets inside TextBlock instances."""
        proc = RedactionProcessor()
        tool_use = ToolUseBlock(id="tu_1", name="get_key", input={"scope": "all"})
        tool_result = ToolResultBlock(
            tool_use_id="tu_1",
            content="tool output here",
        )
        turn = TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    TextBlock(text="Here is the key: sk-abc1234567890ABCDEF"),
                    tool_use,
                    tool_result,
                    TextBlock(text="And another key: AKIAIOSFODNN7EXAMPLE"),
                ],
            ),
        )
        result = proc.process(turn)

        blocks = result.message.content
        assert len(blocks) == 4

        # First block: TextBlock with redacted secret
        assert isinstance(blocks[0], TextBlock)
        assert "[REDACTED_SECRET]" in blocks[0].text
        assert "sk-abc" not in blocks[0].text

        # Second block: ToolUseBlock preserved as-is
        assert isinstance(blocks[1], ToolUseBlock)
        assert blocks[1] is tool_use
        assert blocks[1].name == "get_key"

        # Third block: ToolResultBlock preserved as-is
        assert isinstance(blocks[2], ToolResultBlock)
        assert blocks[2] is tool_result
        assert blocks[2].tool_use_id == "tu_1"

        # Fourth block: TextBlock with redacted secret
        assert isinstance(blocks[3], TextBlock)
        assert "[REDACTED_SECRET]" in blocks[3].text
        assert "AKIA" not in blocks[3].text

    def test_no_redaction_with_tool_blocks_returns_same_turn(self):
        """When no secrets are present, the original turn is returned even
        if the message contains ToolUseBlock/ToolResultBlock."""
        proc = RedactionProcessor()
        turn = TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    TextBlock(text="No secrets here"),
                    ToolUseBlock(id="tu_1", name="search", input={"q": "test"}),
                    ToolResultBlock(tool_use_id="tu_1", content="result"),
                ],
            ),
        )
        result = proc.process(turn)
        assert result is turn  # identity — no copy needed

    def test_redacts_string_content(self):
        """RedactionProcessor handles plain string content."""
        proc = RedactionProcessor()
        turn = TurnRecord(
            run_id="r1",
            turn_id="t1",
            thread_id="thread-1",
            resource_id="resource-1",
            message=Message(
                role=MessageRole.USER,
                content="My key is sk-abc1234567890ABCDEF",
            ),
        )
        result = proc.process(turn)
        assert isinstance(result.message.content, str)
        assert "[REDACTED_SECRET]" in result.message.content
        assert "sk-abc" not in result.message.content


# ===========================================================================
# QueryExpansionProcessor
# ===========================================================================


class TestQueryExpansionProcessor:
    def test_expands_known_synonym(self):
        proc = QueryExpansionProcessor()
        result = proc.process_query("found a bug")
        assert "error" in result
        assert "failure" in result
        assert "issue" in result
        assert "bug" in result

    def test_no_expansion_for_unknown_words(self):
        proc = QueryExpansionProcessor()
        result = proc.process_query("hello world")
        assert result == "hello world"

    def test_expands_decision(self):
        proc = QueryExpansionProcessor()
        result = proc.process_query("the decision was made")
        assert "choice" in result
        assert "selected" in result

    def test_expands_preference(self):
        proc = QueryExpansionProcessor()
        result = proc.process_query("user preference")
        assert "prefer" in result
        assert "settings" in result


# ===========================================================================
# DedupeRecallProcessor
# ===========================================================================


class TestDedupeRecallProcessor:
    def test_removes_exact_duplicates(self):
        proc = DedupeRecallProcessor()
        items = [
            _make_recall_item("same text", item_id="i1"),
            _make_recall_item("same text", item_id="i2"),
        ]
        result = proc.process_items(items)
        assert len(result) == 1
        assert result[0].item_id == "i1"

    def test_case_insensitive_dedup(self):
        proc = DedupeRecallProcessor()
        items = [
            _make_recall_item("Same Text", item_id="i1"),
            _make_recall_item("same text", item_id="i2"),
        ]
        result = proc.process_items(items)
        assert len(result) == 1

    def test_whitespace_insensitive_dedup(self):
        proc = DedupeRecallProcessor()
        items = [
            _make_recall_item("  same text  ", item_id="i1"),
            _make_recall_item("same text", item_id="i2"),
        ]
        result = proc.process_items(items)
        assert len(result) == 1

    def test_keeps_distinct_items(self):
        proc = DedupeRecallProcessor()
        items = [
            _make_recall_item("first item", item_id="i1"),
            _make_recall_item("second item", item_id="i2"),
        ]
        result = proc.process_items(items)
        assert len(result) == 2

    def test_empty_list(self):
        proc = DedupeRecallProcessor()
        result = proc.process_items([])
        assert result == []

    def test_preserves_order(self):
        proc = DedupeRecallProcessor()
        items = [
            _make_recall_item("alpha", item_id="i1"),
            _make_recall_item("beta", item_id="i2"),
            _make_recall_item("alpha", item_id="i3"),
            _make_recall_item("gamma", item_id="i4"),
        ]
        result = proc.process_items(items)
        assert [r.item_id for r in result] == ["i1", "i2", "i4"]
