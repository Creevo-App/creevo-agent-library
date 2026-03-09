# Product-V2 Migration Guide: Legacy Memory to V2 Memory Engine

This document outlines the steps required to migrate Product-V2 from `FullCompressionMemory` to the v2 `DefaultMemoryEngine`.

## Current State

Product-V2 uses `FullCompressionMemory` across 3 agents:

| Agent | File | Memory Persistence | Complexity |
|-------|------|--------------------|------------|
| Game Agent | `server/agents/game/agent.py` | Yes (`to_dict`/`from_json` via `__CAL_SESSION_STATE__`) | High |
| Godot Agent | `server/agents/game/Godot/agent.py` | Yes (file-based `CONVERSATION_HISTORY_FILE`) | Medium |
| Refinement Agent | `server/agents/refinement_agent.py` | No (fresh per invocation) | Low |

## Migration Steps

### 1. Update `requirements.txt`

```diff
- git+https://github.com/Creevo-App/creevo-agent-library.git@main
+ git+https://github.com/Creevo-App/creevo-agent-library.git@amirtarkian/v2-memory-engine
```

Once the PR merges, point back to `@main`.

### 2. Refinement Agent (Easiest — start here)

**File:** `server/agents/refinement_agent.py`

This agent creates fresh memory each invocation and never serializes it.

```diff
- from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory, tool
+ from CAL import Agent, GeminiLLM, StopTool, ContextPolicy, tool
```

```diff
- memory = FullCompressionMemory(
-     summarizer_llm=summarizer,
-     max_tokens=100000,
-     agent_name="refinement"
- )
-
- agent = Agent(
-     llm=gemini_llm,
-     system_prompt=SYSTEM_PROMPT,
-     max_calls=50,
-     max_tokens=10000,
-     memory=memory,
-     agent_name="refinement",
-     tools=[...],
- )
+ agent = Agent(
+     llm=gemini_llm,
+     system_prompt=SYSTEM_PROMPT,
+     max_calls=50,
+     max_tokens=10000,
+     agent_name="refinement",
+     tools=[...],
+ )
```

The v2 `Agent` auto-creates a `DefaultMemoryEngine` when none is provided. No memory parameter needed.

### 3. Game Agent (Hardest — session persistence)

**File:** `server/agents/game/agent.py`

#### 3a. Update imports

```diff
- from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory, tool, SubAgentTool
+ from CAL import Agent, GeminiLLM, StopTool, DefaultMemoryEngine, ContextPolicy, tool, SubAgentTool
+ from CAL import migrate_legacy_memory_payload
```

#### 3b. Update memory initialization

The v2 engine auto-creates when not passed. For session continuity, you need to restore turns from prior sessions.

```diff
- if conversation_state:
-     memory_dict = conversation_state.get('memory', conversation_state)
-     memory = FullCompressionMemory.from_json(
-         memory_dict, summarizer_llm=summarizer_llm, logger=logger, agent_name=session_id
-     )
-     memory.max_tokens = max_tokens
- else:
-     memory = FullCompressionMemory(
-         max_tokens=max_tokens, summarizer_llm=summarizer_llm, logger=logger, agent_name=session_id
-     )
+ engine = DefaultMemoryEngine(
+     context_policy=ContextPolicy(total_token_budget=max_tokens),
+     summarizer_llm=summarizer_llm,
+ )
+
+ # Migrate existing legacy sessions into the v2 engine
+ if conversation_state:
+     memory_payload = conversation_state.get('memory', conversation_state)
+     await migrate_legacy_memory_payload(memory_payload, engine, thread_id=session_id, resource_id=session_id)
```

#### 3c. Update Agent constructor

```diff
- agent = Agent(
-     llm=gemini_llm,
-     ...
-     memory=memory,
-     agent_name=session_id,
-     ...
- )
+ agent = Agent(
+     llm=gemini_llm,
+     ...
+     memory_engine=engine,
+     agent_name=session_id,
+     thread_id=session_id,
+     resource_id=session_id,
+     ...
+ )
```

#### 3d. Update session state emission

**This is the biggest open question.** The v2 engine's `InMemoryConversationStore` does not have a built-in `to_dict()` method. Options:

**Option A: Serialize the conversation store directly**

Add a serialization helper that exports turns from the engine's store:

```python
async def _serialize_engine_state(engine, thread_id):
    """Export v2 engine state for session persistence.

    NOTE: This accesses the public conversation_store API.
    Consider using Option B (first-class serialization) for production.
    """
    turns = await engine.conversation_store.all(thread_id)
    return {
        "version": "v2",
        "thread_id": thread_id,
        "turns": [
            {
                "turn_id": t.turn_id,
                "run_id": t.run_id,
                "thread_id": t.thread_id,
                "resource_id": t.resource_id,
                "message": {
                    "role": t.message.role.value,
                    "content": [b.to_dict() for b in t.message.content] if isinstance(t.message.content, list) else t.message.content,
                    "usage": t.message.usage,
                    "metadata": t.message.metadata,
                },
                "created_at": t.created_at,
                "metadata": t.metadata,
            }
            for t in turns
        ],
    }
```

**Option B: Add `to_dict()`/`from_dict()` to `DefaultMemoryEngine`** (recommended long-term)

This would be a follow-up PR to the CAL library itself, adding first-class serialization support to the v2 engine.

#### 3e. Update `_emit_session_state()`

```diff
  def _emit_session_state(session_id, memory):
      payload = {
          "session_id": session_id,
-         "memory": memory.to_dict(),
+         "memory": await _serialize_engine_state(memory, session_id),
      }
      print(f"{SESSION_STATE_PREFIX}{json.dumps(payload)}", flush=True)
```

### 4. Godot Agent

**File:** `server/agents/game/Godot/agent.py`

Same pattern as the game agent. Additional notes:

- **SubAgentTool simplification:** SubAgentTool shares the parent's memory engine but uses a separate `thread_id` per invocation for isolation. The `make_exploration_tool()` in `common_tools.py` no longer needs `memory.clone()` — just pass tools and the parent agent handles it.
- **File-based history:** The `CONVERSATION_HISTORY_FILE` approach needs the same migration treatment. Read the file, parse as legacy format, call `migrate_legacy_memory_payload()`.
- **Thinking level:** Already supported — `GeminiLLM(thinking_level=...)` works unchanged.

### 5. Node.js Session Store

**File:** `server/lib/agent-request.js`

The `sessionStore` Map stores memory dicts. After migration:

- The dict format changes from legacy `FullCompressionMemory.to_dict()` to the v2 serialization format
- `buildPayloadEnv()` in `helpers.js` passes `CONVERSATION_HISTORY` unchanged — it's opaque JSON either way
- The Python agent handles deserialization and migration

No changes needed in `agent-request.js` if the Python agent handles both legacy and v2 formats gracefully.

### 6. Backward Compatibility During Rollout

To support a gradual rollout where some sessions were started with legacy memory:

```python
if conversation_state:
    memory_payload = conversation_state.get('memory', conversation_state)
    if memory_payload.get('version') == 'v2':
        # Restore v2 state directly
        engine = _restore_engine_from_dict(memory_payload)
    else:
        # Legacy format — migrate
        engine = DefaultMemoryEngine(context_policy=ContextPolicy(total_token_budget=max_tokens))
        await migrate_legacy_memory_payload(memory_payload, engine, thread_id=session_id, resource_id=session_id)
```

## What Simplifies

- **No more `memory.clone()`** — SubAgentTool shares the parent engine with isolated thread IDs
- **No more `sync_token_count_from_llm_usage()`** — `ContextPolicy` defines token budgets applied during `build_context()` assembly
- **No more `_cleanup_incomplete_conversation()`** — v2 turn recording uses structured error handling with automatic health tracking
- **Automatic context assembly** — `ContextPolicy` handles token budgets, truncation ordering, and degraded mode fallback

## Estimated Effort

| Task | Effort | Risk |
|------|--------|------|
| Refinement agent | ~30 min | Low |
| Game agent (minus serialization) | ~1 hour | Low |
| Session serialization (Option A) | ~2 hours | Medium |
| Session serialization (Option B — CAL PR) | ~3 hours | Low |
| Godot agent | ~1 hour | Low |
| Node.js changes | ~30 min | Low |
| Testing with live sessions | ~2 hours | Medium |
| Legacy session migration testing | ~1 hour | Medium |

**Total: ~1-2 days of focused work**, depending on serialization approach.

## Recommended Order

1. Refinement agent (validate v2 works end-to-end)
2. Add serialization to `DefaultMemoryEngine` (CAL library PR)
3. Game agent (with new serialization)
4. Godot agent
5. Integration testing with live K8s pods
