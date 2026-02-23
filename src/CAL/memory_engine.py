"""Memory Engine v2 for CAL.

Implements multi-layer memory with explicit scopes, deterministic context assembly,
and first-class observability hooks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from opentelemetry import trace

from .content_blocks import ContentBlock, TextBlock, ToolResultBlock, ToolUseBlock
from .message import Message, MessageRole
from .llm import LLM


class MemoryScope(str, Enum):
    THREAD = "thread"
    RESOURCE = "resource"


class ContextSourceType(str, Enum):
    SYSTEM = "system"
    TOOL_SCHEMA = "tool_schema"
    RECENT = "recent"
    WORKING = "working"
    SEMANTIC = "semantic"
    ARCHIVE = "archive"
    BUFFER = "buffer"


@dataclass
class ContextPolicy:
    """Token budget and retrieval policy for context assembly."""

    total_token_budget: int = 50000
    system_tokens: int = 3000
    tool_tokens: int = 2500
    working_tokens: int = 5000
    recent_tokens: int = 18000
    semantic_tokens: int = 12000
    archive_tokens: int = 4000
    buffer_tokens: int = 2500
    max_recent_messages: int = 40
    max_semantic_items: int = 12
    max_archive_items: int = 4
    thread_recall_limit: int = 8
    resource_recall_limit: int = 8

    def normalized_budgets(self, system_tokens: int, tool_tokens: int) -> Dict[str, int]:
        """Normalize budgets to fit within total token budget."""
        fixed_system = max(system_tokens, self.system_tokens)
        fixed_tools = max(tool_tokens, self.tool_tokens)
        fixed_buffer = max(0, self.buffer_tokens)

        available = max(0, self.total_token_budget - fixed_system - fixed_tools - fixed_buffer)

        configured_memory = max(0, self.working_tokens) + max(0, self.recent_tokens) + max(0, self.semantic_tokens) + max(0, self.archive_tokens)

        if configured_memory <= 0:
            return {
                "system": fixed_system,
                "tool_schema": fixed_tools,
                "buffer": fixed_buffer,
                "working": 0,
                "recent": 0,
                "semantic": 0,
                "archive": 0,
                "memory_total": available,
            }

        if configured_memory <= available:
            return {
                "system": fixed_system,
                "tool_schema": fixed_tools,
                "buffer": fixed_buffer,
                "working": self.working_tokens,
                "recent": self.recent_tokens,
                "semantic": self.semantic_tokens,
                "archive": self.archive_tokens,
                "memory_total": available,
            }

        scale = available / float(configured_memory)
        working = int(self.working_tokens * scale)
        recent = int(self.recent_tokens * scale)
        semantic = int(self.semantic_tokens * scale)
        archive = int(self.archive_tokens * scale)
        remainder = available - (working + recent + semantic + archive)
        recent += remainder

        return {
            "system": fixed_system,
            "tool_schema": fixed_tools,
            "buffer": fixed_buffer,
            "working": max(0, working),
            "recent": max(0, recent),
            "semantic": max(0, semantic),
            "archive": max(0, archive),
            "memory_total": available,
        }


@dataclass
class RetentionPolicy:
    """Retention defaults chosen for v2."""

    raw_retention_seconds: int = 30 * 24 * 60 * 60
    durable_retention_seconds: int = 365 * 24 * 60 * 60


@dataclass
class TurnRecord:
    run_id: str
    turn_id: str
    thread_id: str
    resource_id: str
    message: Message
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemorySnapshot:
    scope: MemoryScope
    scope_id: str
    state: Dict[str, Any]
    version: int
    updated_at: float


@dataclass
class WorkingMemoryUpdate:
    scope: MemoryScope
    scope_id: str
    patch: Dict[str, Any]
    merge_strategy: str = "merge"


@dataclass
class RecallQuery:
    query: str
    thread_id: str
    resource_id: str
    thread_limit: int = 8
    resource_limit: int = 8
    total_limit: int = 16


@dataclass
class RecallItem:
    item_id: str
    text: str
    scope: MemoryScope
    thread_id: Optional[str]
    resource_id: str
    memory_type: str
    confidence: float
    score: float
    created_at: float
    expires_at: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallResult:
    items: List[RecallItem]
    degraded_mode: bool
    latency_ms: float


@dataclass
class ContextLedgerRecord:
    source_type: ContextSourceType
    source_id: str
    scope: str
    rank: int
    relevance_score: float
    confidence: float
    tokens_used: int
    tokens_reserved: int
    dropped_reason: Optional[str] = None


@dataclass
class ContextLedgerEntry:
    run_id: str
    turn_id: str
    agent_name: str
    thread_id: str
    resource_id: str
    retrieval_latency_ms: float
    assembly_latency_ms: float
    overflowed: bool
    records: List[ContextLedgerRecord]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContextPacket:
    messages: List[Message]
    token_usage_by_source: Dict[str, int]
    total_tokens: int
    truncated: bool
    degraded_mode: bool
    ledger: ContextLedgerEntry


@dataclass
class ContextRequest:
    run_id: str
    turn_id: str
    agent_name: str
    thread_id: str
    resource_id: str
    query: str
    policy: ContextPolicy
    system_prompt_tokens: int = 0
    tool_schema_tokens: int = 0


@dataclass
class MemoryHealthSnapshot:
    write_count: int
    write_errors: int
    recall_count: int
    recall_empty_count: int
    recall_hit_count: int
    scope_leak_count: int
    working_conflict_count: int
    compression_count: int
    last_error: Optional[str] = None


@dataclass
class MemoryWriteEvent:
    run_id: str
    turn_id: str
    thread_id: str
    resource_id: str
    status: str
    memory_type: str
    scope: str
    token_count: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecallEvent:
    run_id: str
    turn_id: str
    thread_id: str
    resource_id: str
    query: str
    latency_ms: float
    hit_count: int
    degraded_mode: bool
    scope_leak_count: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompressionEvent:
    run_id: str
    thread_id: str
    resource_id: str
    archived_turn_count: int
    tokens_before: int
    tokens_after: int
    ratio: float
    fidelity_score: float
    archive_id: str
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class MemoryObserver(Protocol):
    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        pass

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        pass

    def on_recall(self, event: RecallEvent) -> None:
        pass

    def on_compression(self, event: CompressionEvent) -> None:
        pass


class NullMemoryObserver:
    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        return

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        return

    def on_recall(self, event: RecallEvent) -> None:
        return

    def on_compression(self, event: CompressionEvent) -> None:
        return


class InMemoryMemoryObserver:
    """Observer used for tests and local introspection."""

    def __init__(self):
        self.context_ledgers: List[ContextLedgerEntry] = []
        self.write_events: List[MemoryWriteEvent] = []
        self.recall_events: List[RecallEvent] = []
        self.compression_events: List[CompressionEvent] = []

    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        self.context_ledgers.append(ledger)

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        self.write_events.append(event)

    def on_recall(self, event: RecallEvent) -> None:
        self.recall_events.append(event)

    def on_compression(self, event: CompressionEvent) -> None:
        self.compression_events.append(event)


class CompositeMemoryObserver:
    def __init__(self, observers: Sequence[MemoryObserver]):
        self._observers = list(observers)

    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        for observer in self._observers:
            observer.on_context_built(ledger)

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        for observer in self._observers:
            observer.on_memory_write(event)

    def on_recall(self, event: RecallEvent) -> None:
        for observer in self._observers:
            observer.on_recall(event)

    def on_compression(self, event: CompressionEvent) -> None:
        for observer in self._observers:
            observer.on_compression(event)


class LoggerMemoryObserver:
    """Bridges memory events into existing CAL logger metadata stream."""

    def __init__(self, logger: Any):
        self._logger = logger

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._logger:
            return
        try:
            self._logger.log_metadata({"memory_event_type": event_type, "memory_event": payload})
        except Exception:
            return

    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        self._emit(
            "context_built",
            {
                "run_id": ledger.run_id,
                "turn_id": ledger.turn_id,
                "thread_id": ledger.thread_id,
                "resource_id": ledger.resource_id,
                "retrieval_latency_ms": round(ledger.retrieval_latency_ms, 3),
                "assembly_latency_ms": round(ledger.assembly_latency_ms, 3),
                "overflowed": ledger.overflowed,
                "record_count": len(ledger.records),
            },
        )

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        self._emit("memory_write", event.__dict__)

    def on_recall(self, event: RecallEvent) -> None:
        self._emit("memory_recall", event.__dict__)

    def on_compression(self, event: CompressionEvent) -> None:
        self._emit("memory_compression", event.__dict__)


class OTelMemoryObserver:
    """Writes memory events into OpenTelemetry spans."""

    def __init__(self):
        self._tracer = trace.get_tracer("cal.memory")

    def _span(self, name: str, attributes: Dict[str, Any]) -> None:
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(f"cal.memory.{key}", value)

    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        self._span(
            "memory.context.build",
            {
                "run_id": ledger.run_id,
                "turn_id": ledger.turn_id,
                "thread_id": ledger.thread_id,
                "resource_id": ledger.resource_id,
                "overflowed": ledger.overflowed,
                "retrieval_latency_ms": ledger.retrieval_latency_ms,
                "assembly_latency_ms": ledger.assembly_latency_ms,
                "records": len(ledger.records),
            },
        )

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        self._span("memory.write", event.__dict__)

    def on_recall(self, event: RecallEvent) -> None:
        self._span("memory.recall", event.__dict__)

    def on_compression(self, event: CompressionEvent) -> None:
        self._span("memory.compression", event.__dict__)


class ClickHouseMemoryObserver:
    """Best-effort sink for context ledgers and memory events to ClickHouse HTTP API."""

    def __init__(self, endpoint: str, timeout_seconds: float = 1.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, table: str, payload: Dict[str, Any]) -> None:
        body = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.endpoint}/?query=INSERT%20INTO%20{table}%20FORMAT%20JSONEachRow",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds):
                return
        except (urllib.error.URLError, TimeoutError):
            return

    def on_context_built(self, ledger: ContextLedgerEntry) -> None:
        base = {
            "run_id": ledger.run_id,
            "turn_id": ledger.turn_id,
            "agent_name": ledger.agent_name,
            "thread_id": ledger.thread_id,
            "resource_id": ledger.resource_id,
            "retrieval_latency_ms": ledger.retrieval_latency_ms,
            "assembly_latency_ms": ledger.assembly_latency_ms,
            "overflowed": int(ledger.overflowed),
            "timestamp": ledger.timestamp,
        }

        for record in ledger.records:
            row = dict(base)
            row.update(
                {
                    "source_type": record.source_type.value,
                    "source_id": record.source_id,
                    "scope": record.scope,
                    "rank": record.rank,
                    "relevance_score": record.relevance_score,
                    "confidence": record.confidence,
                    "tokens_used": record.tokens_used,
                    "tokens_reserved": record.tokens_reserved,
                    "dropped_reason": record.dropped_reason or "",
                }
            )
            self._post("context_ledger", row)

    def on_memory_write(self, event: MemoryWriteEvent) -> None:
        self._post("memory_write_events", event.__dict__)

    def on_recall(self, event: RecallEvent) -> None:
        self._post("memory_recall_events", event.__dict__)

    def on_compression(self, event: CompressionEvent) -> None:
        self._post("memory_compression_events", event.__dict__)


@runtime_checkable
class ConversationStore(Protocol):
    async def append(self, turn: TurnRecord) -> None:
        pass

    async def recent(self, thread_id: str, limit: int) -> List[TurnRecord]:
        pass

    async def all(self, thread_id: str) -> List[TurnRecord]:
        pass

    async def purge_older_than(self, cutoff_ts: float) -> int:
        pass


@runtime_checkable
class WorkingMemoryStore(Protocol):
    async def get(self, scope: MemoryScope, scope_id: str) -> Optional[WorkingMemorySnapshot]:
        pass

    async def upsert(self, snapshot: WorkingMemorySnapshot) -> None:
        pass


@runtime_checkable
class SemanticMemoryStore(Protocol):
    async def upsert(self, item: RecallItem) -> None:
        pass

    async def recall(
        self,
        query: str,
        scope: MemoryScope,
        thread_id: str,
        resource_id: str,
        limit: int,
    ) -> List[RecallItem]:
        pass

    async def purge_expired(self, now_ts: float) -> int:
        pass


@dataclass
class ArchiveSummary:
    archive_id: str
    thread_id: str
    resource_id: str
    summary: str
    turn_ids: List[str]
    tokens_before: int
    tokens_after: int
    created_at: float


@runtime_checkable
class ArchiveStore(Protocol):
    async def store_summary(self, summary: ArchiveSummary) -> None:
        pass

    async def list_summaries(self, thread_id: str, resource_id: str, limit: int) -> List[ArchiveSummary]:
        pass


class InMemoryConversationStore:
    def __init__(self):
        self._turns: Dict[str, List[TurnRecord]] = {}
        self._lock = threading.Lock()

    async def append(self, turn: TurnRecord) -> None:
        with self._lock:
            self._turns.setdefault(turn.thread_id, []).append(turn)

    async def recent(self, thread_id: str, limit: int) -> List[TurnRecord]:
        with self._lock:
            turns = list(self._turns.get(thread_id, []))
        if limit <= 0:
            return []
        return turns[-limit:]

    async def all(self, thread_id: str) -> List[TurnRecord]:
        with self._lock:
            return list(self._turns.get(thread_id, []))

    async def purge_older_than(self, cutoff_ts: float) -> int:
        purged = 0
        with self._lock:
            for thread_id, turns in list(self._turns.items()):
                keep = [t for t in turns if t.created_at >= cutoff_ts]
                purged += max(0, len(turns) - len(keep))
                self._turns[thread_id] = keep
        return purged


class InMemoryWorkingMemoryStore:
    def __init__(self):
        self._data: Dict[Tuple[MemoryScope, str], WorkingMemorySnapshot] = {}
        self._lock = threading.Lock()

    async def get(self, scope: MemoryScope, scope_id: str) -> Optional[WorkingMemorySnapshot]:
        with self._lock:
            snapshot = self._data.get((scope, scope_id))
            if snapshot is None:
                return None
            return WorkingMemorySnapshot(
                scope=snapshot.scope,
                scope_id=snapshot.scope_id,
                state=dict(snapshot.state),
                version=snapshot.version,
                updated_at=snapshot.updated_at,
            )

    async def upsert(self, snapshot: WorkingMemorySnapshot) -> None:
        with self._lock:
            self._data[(snapshot.scope, snapshot.scope_id)] = snapshot


class InMemorySemanticMemoryStore:
    def __init__(self):
        self._items: List[RecallItem] = []
        self._lock = threading.Lock()

    async def upsert(self, item: RecallItem) -> None:
        with self._lock:
            self._items.append(item)

    async def recall(
        self,
        query: str,
        scope: MemoryScope,
        thread_id: str,
        resource_id: str,
        limit: int,
    ) -> List[RecallItem]:
        query_tokens = _token_set(query)
        now_ts = time.time()

        scored: List[Tuple[float, RecallItem]] = []
        with self._lock:
            items = list(self._items)

        for item in items:
            if item.scope != scope:
                continue
            if scope == MemoryScope.THREAD and item.thread_id != thread_id:
                continue
            if item.resource_id != resource_id:
                continue
            if item.expires_at is not None and item.expires_at < now_ts:
                continue

            score = _lexical_similarity(query_tokens, _token_set(item.text))
            age_seconds = max(0.0, now_ts - item.created_at)
            recency_boost = 1.0 / (1.0 + (age_seconds / 86400.0))
            final_score = score + (0.1 * recency_boost) + (0.1 * item.confidence)

            scored.append(
                (
                    final_score,
                    RecallItem(
                        item_id=item.item_id,
                        text=item.text,
                        scope=item.scope,
                        thread_id=item.thread_id,
                        resource_id=item.resource_id,
                        memory_type=item.memory_type,
                        confidence=item.confidence,
                        score=final_score,
                        created_at=item.created_at,
                        expires_at=item.expires_at,
                        metadata=dict(item.metadata),
                    ),
                )
            )

        scored.sort(key=lambda row: row[0], reverse=True)
        return [row[1] for row in scored[: max(0, limit)]]

    async def purge_expired(self, now_ts: float) -> int:
        with self._lock:
            old_len = len(self._items)
            self._items = [i for i in self._items if i.expires_at is None or i.expires_at >= now_ts]
            return old_len - len(self._items)


class InMemoryArchiveStore:
    def __init__(self):
        self._entries: List[ArchiveSummary] = []
        self._lock = threading.Lock()

    async def store_summary(self, summary: ArchiveSummary) -> None:
        with self._lock:
            self._entries.append(summary)

    async def list_summaries(self, thread_id: str, resource_id: str, limit: int) -> List[ArchiveSummary]:
        with self._lock:
            rows = [
                s
                for s in self._entries
                if s.thread_id == thread_id and s.resource_id == resource_id
            ]
        rows.sort(key=lambda entry: entry.created_at, reverse=True)
        return rows[: max(0, limit)]


@runtime_checkable
class PreWriteProcessor(Protocol):
    def process(self, turn: TurnRecord) -> TurnRecord:
        pass


@runtime_checkable
class PreRecallProcessor(Protocol):
    def process_query(self, query: str) -> str:
        pass


@runtime_checkable
class PostRecallProcessor(Protocol):
    def process_items(self, items: List[RecallItem]) -> List[RecallItem]:
        pass


class RedactionProcessor:
    """Simple pre-write redaction for common secret patterns."""

    _patterns = [
        re.compile(r"(sk-[a-zA-Z0-9]{16,})"),
        re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
        re.compile(r"(AKIA[0-9A-Z]{16})"),
    ]

    def process(self, turn: TurnRecord) -> TurnRecord:
        text = _message_to_text(turn.message)
        redacted = text
        for pattern in self._patterns:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)

        if redacted == text:
            return turn

        new_message = Message(role=turn.message.role, content=[TextBlock(text=redacted)], usage=dict(turn.message.usage), metadata=dict(turn.message.metadata))
        return TurnRecord(
            run_id=turn.run_id,
            turn_id=turn.turn_id,
            thread_id=turn.thread_id,
            resource_id=turn.resource_id,
            message=new_message,
            created_at=turn.created_at,
            metadata=dict(turn.metadata),
        )


class QueryExpansionProcessor:
    """Lightweight query expansion for better semantic recall."""

    _synonyms = {
        "bug": ["error", "failure", "issue"],
        "decision": ["choice", "selected", "decided"],
        "preference": ["prefer", "likes", "settings"],
    }

    def process_query(self, query: str) -> str:
        tokens = _token_set(query)
        expanded = [query]
        for token in tokens:
            for synonym in self._synonyms.get(token, []):
                expanded.append(synonym)
        return " ".join(expanded)


class DedupeRecallProcessor:
    def process_items(self, items: List[RecallItem]) -> List[RecallItem]:
        seen = set()
        result: List[RecallItem] = []
        for item in items:
            key = hashlib.sha1(item.text.strip().lower().encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result


@runtime_checkable
class MemoryEngine(Protocol):
    async def record_turn(self, turn: TurnRecord) -> None:
        pass

    async def build_context(self, request: ContextRequest) -> ContextPacket:
        pass

    async def update_working_memory(self, update: WorkingMemoryUpdate) -> WorkingMemorySnapshot:
        pass

    async def recall(self, query: RecallQuery, run_id: str = "", turn_id: str = "") -> RecallResult:
        pass

    async def get_thread_turns(self, thread_id: str) -> List[Message]:
        pass

    def health_snapshot(self) -> MemoryHealthSnapshot:
        pass


class DefaultMemoryEngine:
    """Default in-process implementation for v2 memory architecture."""

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
        self.conversation_store = conversation_store or InMemoryConversationStore()
        self.working_store = working_store or InMemoryWorkingMemoryStore()
        self.semantic_store = semantic_store or InMemorySemanticMemoryStore()
        self.archive_store = archive_store or InMemoryArchiveStore()

        self.context_policy = context_policy or ContextPolicy()
        self.retention_policy = retention_policy or RetentionPolicy()
        self.observer = observer or NullMemoryObserver()

        self.pre_write_processors = pre_write_processors or [RedactionProcessor()]
        self.pre_recall_processors = pre_recall_processors or [QueryExpansionProcessor()]
        self.post_recall_processors = post_recall_processors or [DedupeRecallProcessor()]

        self.archive_cold_threshold = max(4, archive_cold_threshold)
        self.archive_keep_recent = max(2, archive_keep_recent)

        self.summarizer_llm = summarizer_llm
        self.archiver = archiver
        self._archived_turn_ids: set[str] = set()
        self._lock = threading.Lock()
        self._health = MemoryHealthSnapshot(
            write_count=0,
            write_errors=0,
            recall_count=0,
            recall_empty_count=0,
            recall_hit_count=0,
            scope_leak_count=0,
            working_conflict_count=0,
            compression_count=0,
            last_error=None,
        )

    async def get_thread_turns(self, thread_id: str) -> List[Message]:
        turns = await self.conversation_store.all(thread_id)
        return [turn.message for turn in turns]

    def health_snapshot(self) -> MemoryHealthSnapshot:
        with self._lock:
            return MemoryHealthSnapshot(**self._health.__dict__)

    async def record_turn(self, turn: TurnRecord) -> None:
        try:
            for processor in self.pre_write_processors:
                turn = processor.process(turn)

            await self.conversation_store.append(turn)

            text = _message_to_text(turn.message)
            token_count = _estimate_tokens(text)
            memory_type = self._classify_memory_type(text)

            await self._upsert_semantic_memories(turn, text, memory_type)
            await self._maybe_archive_cold_history(turn.run_id, turn.thread_id, turn.resource_id)

            self._inc_health(write_count=1)
            self.observer.on_memory_write(
                MemoryWriteEvent(
                    run_id=turn.run_id,
                    turn_id=turn.turn_id,
                    thread_id=turn.thread_id,
                    resource_id=turn.resource_id,
                    status="ok",
                    memory_type=memory_type,
                    scope="thread",
                    token_count=token_count,
                )
            )
        except Exception as exc:
            self._inc_health(write_count=1, write_errors=1, last_error=str(exc))
            self.observer.on_memory_write(
                MemoryWriteEvent(
                    run_id=turn.run_id,
                    turn_id=turn.turn_id,
                    thread_id=turn.thread_id,
                    resource_id=turn.resource_id,
                    status="error",
                    memory_type="unknown",
                    scope="thread",
                    token_count=0,
                    error=str(exc),
                )
            )
            raise

    async def update_working_memory(self, update: WorkingMemoryUpdate) -> WorkingMemorySnapshot:
        existing = await self.working_store.get(update.scope, update.scope_id)
        now_ts = time.time()

        if existing is None:
            merged = dict(update.patch)
            snapshot = WorkingMemorySnapshot(
                scope=update.scope,
                scope_id=update.scope_id,
                state=merged,
                version=1,
                updated_at=now_ts,
            )
            await self.working_store.upsert(snapshot)
            return snapshot

        if update.merge_strategy == "replace":
            merged = dict(update.patch)
            conflicts = 0
        else:
            merged = dict(existing.state)
            conflicts = 0
            for key, value in update.patch.items():
                if key in merged and merged[key] != value:
                    conflicts += 1
                merged[key] = value

        if conflicts > 0:
            self._inc_health(working_conflict_count=conflicts)

        snapshot = WorkingMemorySnapshot(
            scope=update.scope,
            scope_id=update.scope_id,
            state=merged,
            version=existing.version + 1,
            updated_at=now_ts,
        )
        await self.working_store.upsert(snapshot)
        return snapshot

    async def recall(self, query: RecallQuery, run_id: str = "", turn_id: str = "") -> RecallResult:
        started = time.perf_counter()
        scope_leaks = 0

        try:
            text_query = query.query
            for processor in self.pre_recall_processors:
                text_query = processor.process_query(text_query)

            thread_items = await self.semantic_store.recall(
                text_query,
                scope=MemoryScope.THREAD,
                thread_id=query.thread_id,
                resource_id=query.resource_id,
                limit=max(0, query.thread_limit),
            )
            resource_items = await self.semantic_store.recall(
                text_query,
                scope=MemoryScope.RESOURCE,
                thread_id=query.thread_id,
                resource_id=query.resource_id,
                limit=max(0, query.resource_limit),
            )

            merged = list(thread_items) + list(resource_items)
            merged.sort(key=lambda item: item.score, reverse=True)

            filtered: List[RecallItem] = []
            for item in merged:
                if item.scope == MemoryScope.RESOURCE and item.resource_id != query.resource_id:
                    scope_leaks += 1
                    continue
                filtered.append(item)

            limited = filtered[: max(0, query.total_limit)]
            for processor in self.post_recall_processors:
                limited = processor.process_items(limited)

            latency_ms = (time.perf_counter() - started) * 1000.0
            self._inc_health(
                recall_count=1,
                recall_hit_count=len(limited),
                recall_empty_count=1 if len(limited) == 0 else 0,
                scope_leak_count=scope_leaks,
            )

            event = RecallEvent(
                run_id=run_id,
                turn_id=turn_id,
                thread_id=query.thread_id,
                resource_id=query.resource_id,
                query=query.query,
                latency_ms=latency_ms,
                hit_count=len(limited),
                degraded_mode=False,
                scope_leak_count=scope_leaks,
            )
            self.observer.on_recall(event)

            return RecallResult(items=limited, degraded_mode=False, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._inc_health(recall_count=1, recall_empty_count=1, last_error=str(exc))
            self.observer.on_recall(
                RecallEvent(
                    run_id=run_id,
                    turn_id=turn_id,
                    thread_id=query.thread_id,
                    resource_id=query.resource_id,
                    query=query.query,
                    latency_ms=latency_ms,
                    hit_count=0,
                    degraded_mode=True,
                    scope_leak_count=scope_leaks,
                    error=str(exc),
                )
            )
            return RecallResult(items=[], degraded_mode=True, latency_ms=latency_ms)

    async def build_context(self, request: ContextRequest) -> ContextPacket:
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()

        policy = request.policy or self.context_policy
        budgets = policy.normalized_budgets(request.system_prompt_tokens, request.tool_schema_tokens)

        ledger_records: List[ContextLedgerRecord] = [
            ContextLedgerRecord(
                source_type=ContextSourceType.SYSTEM,
                source_id="system_prompt",
                scope="n/a",
                rank=0,
                relevance_score=1.0,
                confidence=1.0,
                tokens_used=request.system_prompt_tokens,
                tokens_reserved=budgets["system"],
            ),
            ContextLedgerRecord(
                source_type=ContextSourceType.TOOL_SCHEMA,
                source_id="tool_schemas",
                scope="n/a",
                rank=0,
                relevance_score=1.0,
                confidence=1.0,
                tokens_used=request.tool_schema_tokens,
                tokens_reserved=budgets["tool_schema"],
            ),
        ]

        # Stage B - candidate collection
        recent_turns = await self.conversation_store.recent(request.thread_id, policy.max_recent_messages)

        working_resource = await self.working_store.get(MemoryScope.RESOURCE, request.resource_id)
        working_thread = await self.working_store.get(MemoryScope.THREAD, request.thread_id)

        recall_result = await self.recall(
            RecallQuery(
                query=request.query,
                thread_id=request.thread_id,
                resource_id=request.resource_id,
                thread_limit=policy.thread_recall_limit,
                resource_limit=policy.resource_recall_limit,
                total_limit=policy.max_semantic_items,
            ),
            run_id=request.run_id,
            turn_id=request.turn_id,
        )

        archive_summaries = await self.archive_store.list_summaries(
            request.thread_id,
            request.resource_id,
            limit=policy.max_archive_items,
        )

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Stage C/D - segment assembly
        selected_working: List[Tuple[str, Message, int]] = []
        working_tokens = 0

        merged_state: Dict[str, Any] = {}
        if working_resource:
            merged_state.update(working_resource.state)
        if working_thread:
            merged_state.update(working_thread.state)

        if merged_state:
            wm_text = json.dumps(merged_state, sort_keys=True, ensure_ascii=True)
            wm_message = Message(role=MessageRole.USER, content=[TextBlock(text=f"[Working Memory]\n{wm_text}")])
            wm_tokens = _estimate_message_tokens(wm_message)
            if wm_tokens <= budgets["working"]:
                selected_working.append(("working", wm_message, wm_tokens))
                working_tokens = wm_tokens
                ledger_records.append(
                    ContextLedgerRecord(
                        source_type=ContextSourceType.WORKING,
                        source_id=f"working::{request.resource_id}::{request.thread_id}",
                        scope="resource+thread",
                        rank=0,
                        relevance_score=1.0,
                        confidence=1.0,
                        tokens_used=wm_tokens,
                        tokens_reserved=budgets["working"],
                    )
                )
            else:
                ledger_records.append(
                    ContextLedgerRecord(
                        source_type=ContextSourceType.WORKING,
                        source_id=f"working::{request.resource_id}::{request.thread_id}",
                        scope="resource+thread",
                        rank=0,
                        relevance_score=1.0,
                        confidence=1.0,
                        tokens_used=wm_tokens,
                        tokens_reserved=budgets["working"],
                        dropped_reason="segment_budget",
                    )
                )

        selected_recent: List[Tuple[TurnRecord, int]] = []
        recent_tokens = 0
        for turn in reversed(recent_turns):
            msg_tokens = _estimate_message_tokens(turn.message)
            if recent_tokens + msg_tokens <= budgets["recent"]:
                selected_recent.append((turn, msg_tokens))
                recent_tokens += msg_tokens
            else:
                ledger_records.append(
                    ContextLedgerRecord(
                        source_type=ContextSourceType.RECENT,
                        source_id=turn.turn_id,
                        scope="thread",
                        rank=0,
                        relevance_score=0.5,
                        confidence=1.0,
                        tokens_used=msg_tokens,
                        tokens_reserved=budgets["recent"],
                        dropped_reason="segment_budget",
                    )
                )
        selected_recent.reverse()

        for idx, (turn, msg_tokens) in enumerate(selected_recent):
            ledger_records.append(
                ContextLedgerRecord(
                    source_type=ContextSourceType.RECENT,
                    source_id=turn.turn_id,
                    scope="thread",
                    rank=idx,
                    relevance_score=0.5,
                    confidence=1.0,
                    tokens_used=msg_tokens,
                    tokens_reserved=budgets["recent"],
                )
            )

        selected_semantic: List[Tuple[RecallItem, Message, int]] = []
        semantic_tokens = 0
        for idx, item in enumerate(recall_result.items):
            semantic_text = f"[Semantic {item.memory_type}] {item.text}"
            semantic_msg = Message(role=MessageRole.USER, content=[TextBlock(text=semantic_text)])
            msg_tokens = _estimate_message_tokens(semantic_msg)

            if idx >= policy.max_semantic_items or semantic_tokens + msg_tokens > budgets["semantic"]:
                ledger_records.append(
                    ContextLedgerRecord(
                        source_type=ContextSourceType.SEMANTIC,
                        source_id=item.item_id,
                        scope=item.scope.value,
                        rank=idx,
                        relevance_score=item.score,
                        confidence=item.confidence,
                        tokens_used=msg_tokens,
                        tokens_reserved=budgets["semantic"],
                        dropped_reason="segment_budget",
                    )
                )
                continue

            selected_semantic.append((item, semantic_msg, msg_tokens))
            semantic_tokens += msg_tokens
            ledger_records.append(
                ContextLedgerRecord(
                    source_type=ContextSourceType.SEMANTIC,
                    source_id=item.item_id,
                    scope=item.scope.value,
                    rank=idx,
                    relevance_score=item.score,
                    confidence=item.confidence,
                    tokens_used=msg_tokens,
                    tokens_reserved=budgets["semantic"],
                )
            )

        selected_archive: List[Tuple[ArchiveSummary, Message, int]] = []
        archive_tokens = 0
        for idx, summary in enumerate(archive_summaries):
            archive_msg = Message(
                role=MessageRole.USER,
                content=[TextBlock(text=f"[Archive Summary] {summary.summary}")],
            )
            msg_tokens = _estimate_message_tokens(archive_msg)
            if idx >= policy.max_archive_items or archive_tokens + msg_tokens > budgets["archive"]:
                ledger_records.append(
                    ContextLedgerRecord(
                        source_type=ContextSourceType.ARCHIVE,
                        source_id=summary.archive_id,
                        scope="thread",
                        rank=idx,
                        relevance_score=0.4,
                        confidence=0.6,
                        tokens_used=msg_tokens,
                        tokens_reserved=budgets["archive"],
                        dropped_reason="segment_budget",
                    )
                )
                continue

            selected_archive.append((summary, archive_msg, msg_tokens))
            archive_tokens += msg_tokens
            ledger_records.append(
                ContextLedgerRecord(
                    source_type=ContextSourceType.ARCHIVE,
                    source_id=summary.archive_id,
                    scope="thread",
                    rank=idx,
                    relevance_score=0.4,
                    confidence=0.6,
                    tokens_used=msg_tokens,
                    tokens_reserved=budgets["archive"],
                )
            )

        # Stage E - enforce overall cap with deterministic drop order
        memory_tokens = working_tokens + recent_tokens + semantic_tokens + archive_tokens
        memory_cap = budgets["memory_total"]
        truncated = False

        while memory_tokens > memory_cap:
            truncated = True

            if selected_archive:
                dropped_summary, _, dropped_tokens = selected_archive.pop(0)
                archive_tokens -= dropped_tokens
                memory_tokens -= dropped_tokens
                _mark_ledger_drop(ledger_records, ContextSourceType.ARCHIVE, dropped_summary.archive_id, "overflow_total")
                continue

            if selected_semantic:
                dropped_item, _, dropped_tokens = selected_semantic.pop(-1)
                semantic_tokens -= dropped_tokens
                memory_tokens -= dropped_tokens
                _mark_ledger_drop(ledger_records, ContextSourceType.SEMANTIC, dropped_item.item_id, "overflow_total")
                continue

            if selected_recent:
                dropped_turn, dropped_tokens = selected_recent.pop(0)
                recent_tokens -= dropped_tokens
                memory_tokens -= dropped_tokens
                _mark_ledger_drop(ledger_records, ContextSourceType.RECENT, dropped_turn.turn_id, "overflow_total")
                continue

            break

        # Final packet ordering: archive -> semantic -> working -> recent
        packet_messages: List[Message] = []
        packet_messages.extend(msg for _, msg, _ in selected_archive)
        packet_messages.extend(msg for _, msg, _ in selected_semantic)
        packet_messages.extend(msg for _, msg, _ in selected_working)
        packet_messages.extend(turn.message for turn, _ in selected_recent)

        token_usage = {
            ContextSourceType.SYSTEM.value: request.system_prompt_tokens,
            ContextSourceType.TOOL_SCHEMA.value: request.tool_schema_tokens,
            ContextSourceType.BUFFER.value: budgets["buffer"],
            ContextSourceType.RECENT.value: recent_tokens,
            ContextSourceType.WORKING.value: working_tokens,
            ContextSourceType.SEMANTIC.value: semantic_tokens,
            ContextSourceType.ARCHIVE.value: archive_tokens,
        }

        total_tokens = sum(token_usage.values())
        assembly_latency_ms = (time.perf_counter() - total_start) * 1000.0
        overflowed = truncated or (total_tokens > policy.total_token_budget)

        ledger = ContextLedgerEntry(
            run_id=request.run_id,
            turn_id=request.turn_id,
            agent_name=request.agent_name,
            thread_id=request.thread_id,
            resource_id=request.resource_id,
            retrieval_latency_ms=retrieval_latency_ms,
            assembly_latency_ms=assembly_latency_ms,
            overflowed=overflowed,
            records=ledger_records,
        )

        self.observer.on_context_built(ledger)

        return ContextPacket(
            messages=packet_messages,
            token_usage_by_source=token_usage,
            total_tokens=total_tokens,
            truncated=truncated,
            degraded_mode=recall_result.degraded_mode,
            ledger=ledger,
        )

    async def cleanup_retention(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        now_ts = now_ts or time.time()
        raw_cutoff = now_ts - self.retention_policy.raw_retention_seconds
        purged_raw = await self.conversation_store.purge_older_than(raw_cutoff)
        purged_semantic = await self.semantic_store.purge_expired(now_ts)
        return {
            "purged_raw": purged_raw,
            "purged_semantic": purged_semantic,
        }

    async def _upsert_semantic_memories(self, turn: TurnRecord, text: str, memory_type: str) -> None:
        if not text.strip():
            return

        facts = self._extract_fact_candidates(text)
        if not facts:
            return

        now_ts = time.time()
        thread_expires = now_ts + self.retention_policy.raw_retention_seconds
        durable_expires = now_ts + self.retention_policy.durable_retention_seconds

        for idx, fact_text in enumerate(facts):
            base_item_id = f"{turn.turn_id}:{idx}"

            thread_item = RecallItem(
                item_id=f"thread:{base_item_id}",
                text=fact_text,
                scope=MemoryScope.THREAD,
                thread_id=turn.thread_id,
                resource_id=turn.resource_id,
                memory_type=memory_type,
                confidence=self._confidence_for(memory_type),
                score=0.0,
                created_at=now_ts,
                expires_at=thread_expires,
                metadata={"run_id": turn.run_id, "turn_id": turn.turn_id},
            )
            await self.semantic_store.upsert(thread_item)

            if memory_type in {"fact", "decision", "preference"}:
                resource_item = RecallItem(
                    item_id=f"resource:{base_item_id}",
                    text=fact_text,
                    scope=MemoryScope.RESOURCE,
                    thread_id=None,
                    resource_id=turn.resource_id,
                    memory_type=memory_type,
                    confidence=self._confidence_for(memory_type),
                    score=0.0,
                    created_at=now_ts,
                    expires_at=durable_expires,
                    metadata={"run_id": turn.run_id, "turn_id": turn.turn_id},
                )
                await self.semantic_store.upsert(resource_item)

    async def _maybe_archive_cold_history(self, run_id: str, thread_id: str, resource_id: str) -> None:
        turns = await self.conversation_store.all(thread_id)
        if len(turns) <= self.archive_cold_threshold:
            return

        candidate_turns = [t for t in turns[:-self.archive_keep_recent] if t.turn_id not in self._archived_turn_ids]
        if len(candidate_turns) < max(4, self.archive_cold_threshold // 3):
            return

        tokens_before = sum(_estimate_message_tokens(turn.message) for turn in candidate_turns)
        if self.summarizer_llm is not None:
            try:
                summary_text = self._summarize_turns_with_llm(candidate_turns)
            except Exception:
                summary_text = self._summarize_turns(candidate_turns)
        else:
            summary_text = self._summarize_turns(candidate_turns)
        summary_message_tokens = _estimate_tokens(summary_text)

        archive_id = f"archive-{uuid.uuid4().hex[:12]}"
        summary = ArchiveSummary(
            archive_id=archive_id,
            thread_id=thread_id,
            resource_id=resource_id,
            summary=summary_text,
            turn_ids=[turn.turn_id for turn in candidate_turns],
            tokens_before=tokens_before,
            tokens_after=summary_message_tokens,
            created_at=time.time(),
        )
        await self.archive_store.store_summary(summary)

        if self.archiver is not None:
            try:
                archive_content = "\n\n".join(
                    f"{t.message.role.value}: {_message_to_text(t.message)}" for t in candidate_turns
                )
                tools_used = []
                for t in candidate_turns:
                    text = _message_to_text(t.message)
                    if "[ToolUse " in text:
                        names = re.findall(r"\[ToolUse (\S+)\]", text)
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
                pass

        for turn in candidate_turns:
            self._archived_turn_ids.add(turn.turn_id)

        ratio = summary_message_tokens / max(1, tokens_before)
        fidelity = max(0.0, min(1.0, 1.0 - ratio))

        self._inc_health(compression_count=1)
        self.observer.on_compression(
            CompressionEvent(
                run_id=run_id,
                thread_id=thread_id,
                resource_id=resource_id,
                archived_turn_count=len(candidate_turns),
                tokens_before=tokens_before,
                tokens_after=summary_message_tokens,
                ratio=ratio,
                fidelity_score=fidelity,
                archive_id=archive_id,
            )
        )

    def _classify_memory_type(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("decide", "decision", "chosen", "we will")):
            return "decision"
        if any(word in lowered for word in ("prefer", "preference", "i like", "always use")):
            return "preference"
        if any(word in lowered for word in ("todo", "next", "pending", "follow up", "task")):
            return "task"
        if any(word in lowered for word in ("error", "failed", "failure", "exception")):
            return "error_pattern"
        return "fact"

    def _confidence_for(self, memory_type: str) -> float:
        if memory_type == "decision":
            return 0.9
        if memory_type == "preference":
            return 0.85
        if memory_type == "fact":
            return 0.75
        if memory_type == "task":
            return 0.7
        return 0.6

    def _extract_fact_candidates(self, text: str) -> List[str]:
        # Split by sentence boundaries and keep informative lines only.
        parts = [part.strip() for part in re.split(r"[\n\.!?]+", text) if part.strip()]
        candidates = [part for part in parts if len(part.split()) >= 4]
        if not candidates:
            return [text[:300]]
        return candidates[:3]

    def _summarize_turns(self, turns: List[TurnRecord]) -> str:
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

    def _inc_health(
        self,
        write_count: int = 0,
        write_errors: int = 0,
        recall_count: int = 0,
        recall_empty_count: int = 0,
        recall_hit_count: int = 0,
        scope_leak_count: int = 0,
        working_conflict_count: int = 0,
        compression_count: int = 0,
        last_error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._health.write_count += write_count
            self._health.write_errors += write_errors
            self._health.recall_count += recall_count
            self._health.recall_empty_count += recall_empty_count
            self._health.recall_hit_count += recall_hit_count
            self._health.scope_leak_count += scope_leak_count
            self._health.working_conflict_count += working_conflict_count
            self._health.compression_count += compression_count
            if last_error:
                self._health.last_error = last_error


def _mark_ledger_drop(
    records: List[ContextLedgerRecord],
    source_type: ContextSourceType,
    source_id: str,
    dropped_reason: str,
) -> None:
    for record in records:
        if record.source_type == source_type and record.source_id == source_id:
            record.dropped_reason = dropped_reason
            return


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _lexical_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a.intersection(b))
    return overlap / float(len(a))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def _message_to_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content

    parts: List[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[ToolUse {block.name}] {json.dumps(block.input, default=str, ensure_ascii=True)}")
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                content_text = block.content
            elif isinstance(block.content, list):
                children = []
                for child in block.content:
                    if isinstance(child, TextBlock):
                        children.append(child.text)
                    else:
                        children.append(str(child))
                content_text = "\n".join(children)
            else:
                content_text = str(block.content)
            parts.append(f"[ToolResult {block.name or 'unknown'}] {content_text}")
        else:
            if isinstance(block, ContentBlock):
                parts.append(str(block.to_dict()))
            else:
                parts.append(str(block))
    return "\n".join(parts)


def _estimate_message_tokens(message: Message) -> int:
    if message.role == MessageRole.ASSISTANT and message.usage:
        completion_tokens = message.usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            return completion_tokens
    return _estimate_tokens(_message_to_text(message))


def estimate_tool_schema_tokens(tools: Sequence[Any]) -> int:
    total = 0
    for tool in tools:
        try:
            schema = tool.get_schema()
            total += _estimate_tokens(json.dumps(schema, default=str, ensure_ascii=True))
        except Exception:
            total += 50
    return total


def estimate_system_tokens(system_prompt: str) -> int:
    return _estimate_tokens(system_prompt or "")
