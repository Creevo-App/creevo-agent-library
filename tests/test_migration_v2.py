import pytest

from CAL.memory_engine import DefaultMemoryEngine
from CAL.migrations import migrate_legacy_memory_json, migrate_legacy_memory_payload


@pytest.mark.asyncio
async def test_migrate_legacy_memory_payload_into_v2_engine():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {},
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "world"}],
                "usage": {"completion_tokens": 2},
            },
        ]
    }

    engine = DefaultMemoryEngine()
    migrated = await migrate_legacy_memory_payload(
        payload=payload,
        memory_engine=engine,
        thread_id="thread-1",
        resource_id="resource-1",
        run_id="migration",
    )

    assert migrated == 2
    turns = await engine.get_thread_turns("thread-1")
    assert len(turns) == 2
    assert turns[0].role.value == "user"
    assert turns[1].role.value == "assistant"


@pytest.mark.asyncio
async def test_migrate_legacy_memory_json_empty_is_noop():
    engine = DefaultMemoryEngine()
    migrated = await migrate_legacy_memory_json(
        data="",
        memory_engine=engine,
        thread_id="thread-1",
        resource_id="resource-1",
    )
    assert migrated == 0
