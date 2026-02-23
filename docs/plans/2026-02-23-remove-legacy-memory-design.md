# Remove Legacy Memory Backwards Compatibility

## Context

The `mikil/new-memory` branch introduced a v2 memory engine (`DefaultMemoryEngine`) alongside the legacy `Memory`/`FullCompressionMemory` system. The agent currently maintains two parallel execution paths, dual constructor params, and branching in subagent creation. This design removes all legacy memory code, making v2 the only path.

## Decisions

- **No legacy remnants**: `Memory` abstract class, `FullCompressionMemory`, and all dual-path code are removed.
- **Migration utils kept**: `migrations/migrate_legacy_memory.py` stays for one-time import of serialized legacy payloads.
- **Auto-create default**: `Agent()` with no `memory_engine` arg creates a `DefaultMemoryEngine` automatically.
- **LLM summarization ported**: `DefaultMemoryEngine` gains an optional `summarizer_llm` param. When provided, archive compression uses LLM-based summarization (ported from `FullCompressionMemory.compress()`). Falls back to naive text truncation when `None`.
- **CompressionArchiver preserved**: `compression.py` stays. `DefaultMemoryEngine` gains an optional `archiver` param for file-based archive output.

## Deletions

### Files removed
- `src/CAL/memory.py` (~800 lines)
- `tests/test_memory.py` (all legacy compression/serialization tests)

### Code removed from existing files
- `agent.py`: `_run_async_legacy()`, `_cleanup_incomplete_conversation()`, `self.memory` field, `memory` constructor param, routing logic in `run_async()`
- `subagent.py`: Legacy `else` branch (clone memory, set archiver/logger on clone)
- `__init__.py`: Exports of `Memory`, `FullCompressionMemory`
- `conftest.py`: `TrackingMemory(FullCompressionMemory)` fixture

## Changes to DefaultMemoryEngine

### New constructor params
- `summarizer_llm: Optional[LLM] = None` — when provided, `_maybe_archive_cold_history` sends cold turns to the LLM with a summarization prompt and stores the result as `ArchiveSummary.summary`
- `archiver: Optional[CompressionArchiver] = None` — when provided, full archived context is also written to disk

### Summarization flow (ported from FullCompressionMemory.compress)
1. Collect cold turns beyond `archive_keep_recent` threshold
2. Convert to text via `_message_to_text()`
3. Send to `summarizer_llm.generate_content()` with summarization system prompt
4. Store LLM response as `ArchiveSummary.summary`
5. If `archiver` is set, write full context to disk via `CompressionArchiver.write_context_file()`

### Fallback
When `summarizer_llm` is `None`, the existing `_summarize_turns()` naive text truncation is used (no behavior change for callers who don't provide an LLM).

## Agent Constructor

### Before
```python
def __init__(self, llm, system_prompt, max_calls, max_tokens,
             memory=None, agent_name="session", tools=None, logger=None,
             memory_engine=None, context_policy=None, thread_id=None, resource_id=None)
```

### After
```python
def __init__(self, llm, system_prompt, max_calls, max_tokens,
             agent_name="session", tools=None, logger=None,
             memory_engine=None, context_policy=None, thread_id=None, resource_id=None)
```

- `memory` param removed
- Single execution path (current `_run_async_v2` becomes the main `run_async` body)

## SubAgent Simplification

All subagents use:
- Shared `memory_engine` from parent
- Own `thread_id` for isolation
- Shared `resource_id` for cross-agent semantic recall
- `context_policy` derived from parent with optional `sub_max_tokens` override

The legacy clone-memory branch is removed entirely.

## Test Strategy

- Delete `tests/test_memory.py`
- Rewrite `tests/test_agent.py` to use `DefaultMemoryEngine`
- Rewrite `tests/test_subagent.py` for v2 path only
- Rewrite `tests/test_integration.py` for v2
- Update `tests/conftest.py` (remove `TrackingMemory`, update fixtures)
- Keep `tests/test_memory_engine_v2.py` and `tests/test_migration_v2.py` as-is
- Add test for LLM-based summarization in the engine
