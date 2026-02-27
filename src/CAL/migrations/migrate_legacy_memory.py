"""One-time migration helpers from legacy FullCompressionMemory payloads to v2 MemoryEngine.

TODO: Remove this module (and its exports in __init__.py) once all legacy memory
payloads have been migrated to the v2 DefaultMemoryEngine.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from ..content_blocks import ContentBlock
from ..memory_engine import MemoryEngine, TurnRecord
from ..message import Message, MessageRole


def _role_from_str(value: str) -> MessageRole:
    if value == MessageRole.ASSISTANT.value:
        return MessageRole.ASSISTANT
    if value == MessageRole.TOOL_RESPONSE.value:
        return MessageRole.TOOL_RESPONSE
    if value == MessageRole.SYSTEM.value:
        return MessageRole.SYSTEM
    return MessageRole.USER


def _content_from_serializable(content: Union[str, List[Dict[str, Any]], None]):
    if content is None:
        return []
    if isinstance(content, str):
        return content

    blocks = []
    for item in content:
        block = ContentBlock.from_dict(item)
        if block:
            blocks.append(block)
    return blocks


async def migrate_legacy_memory_payload(
    payload: Dict[str, Any],
    memory_engine: MemoryEngine,
    thread_id: str,
    resource_id: str,
    run_id: str = "migration",
) -> int:
    """Migrate serialized legacy memory payload into a v2 memory engine.

    Returns the number of migrated messages.
    """
    messages_data = payload.get("messages", [])
    migrated = 0

    for idx, raw in enumerate(messages_data):
        role = _role_from_str(raw.get("role", MessageRole.USER.value))
        message = Message(
            role=role,
            content=_content_from_serializable(raw.get("content")),
            usage=raw.get("usage") or {},
            metadata=raw.get("metadata") or {},
        )

        turn = TurnRecord(
            run_id=run_id,
            turn_id=f"{run_id}:{idx:04d}",
            thread_id=thread_id,
            resource_id=resource_id,
            message=message,
            metadata={"source": "legacy_memory"},
        )
        await memory_engine.record_turn(turn)
        migrated += 1

    return migrated


async def migrate_legacy_memory_json(
    data: str,
    memory_engine: MemoryEngine,
    thread_id: str,
    resource_id: str,
    run_id: str = "migration",
) -> int:
    """Migrate a legacy memory JSON string into a v2 memory engine."""
    if not data:
        return 0
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(payload).__name__}: {data[:120]!r}"
        )
    return await migrate_legacy_memory_payload(
        payload=payload,
        memory_engine=memory_engine,
        thread_id=thread_id,
        resource_id=resource_id,
        run_id=run_id,
    )
