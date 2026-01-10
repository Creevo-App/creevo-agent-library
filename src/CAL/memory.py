"""
Conversation memory management for CAL agents.
"""
import base64
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .message import Message, MessageRole
from .content_blocks import (
    ContentBlock,
    ImageBlock,
    ImageSource,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class Memory(ABC):
    """
    Abstract base class for conversation memory implementations.
    """

    @abstractmethod
    def add_message(self, message: Message):
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_history(self) -> List[Message]:
        """Return the current conversation history."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all stored messages."""
        pass

    @abstractmethod
    def compress(self):
        """Compress memory to reduce size."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory into a dictionary."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Serialize the memory into a JSON string."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Memory":
        """Construct memory from a serialized dictionary payload."""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Optional[str]) -> "Memory":
        """Construct memory from a JSON string."""
        pass


class FullCompressionMemory(Memory):
    """
    Memory for long-running agentic tasks. Keeps the initial user prompt,
    summarizes middle turns, and keeps recent messages.
    """

    def __init__(self, max_items: int = 50, messages: Optional[List[Message]] = None):
        self.max_items = max_items
        self._messages: List[Message] = []
        if messages:
            for message in messages:
                self._messages.append(message)

    def add_message(self, message: Message):
        """Add a message to memory. Compress when capacity exceeded."""
        self._messages.append(message)
        if len(self._messages) > self.max_items:
            self.compress()

    def compress(self):
        """
        Compress memory: keep initial user message, summarize middle,
        keep most recent N messages.
        """
        if len(self._messages) <= 3:
            return
        
        keep_recent = self.max_items // 2
        
        initial = self._messages[0]
        to_compress = self._messages[1:-keep_recent]
        recent = self._messages[-keep_recent:]
        
        if not to_compress:
            return
        
        # Build summary
        summary_parts = [f"[Summary of {len(to_compress)} previous turns:]"]
        for msg in to_compress:
            summary_parts.append(f"- {msg.role.value.upper()}: {self._summarize_content(msg.content)}")
        
        summary_message = Message(
            role=MessageRole.USER,
            content="\n".join(summary_parts),
            metadata={"compressed": True}
        )
        
        self._messages = [initial, summary_message] + recent

    def _summarize_content(self, content: Union[str, List[ContentBlock]]) -> str:
        """Create a brief summary of message content."""
        if isinstance(content, str):
            return content[:150] + "..." if len(content) > 150 else content
        
        parts = []
        for block in content:
            if isinstance(block, TextBlock):
                text = block.text[:100] + "..." if len(block.text) > 100 else block.text
                parts.append(text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[Called {block.name}]")
            elif isinstance(block, ToolResultBlock):
                parts.append("[Tool result]")
            elif isinstance(block, ImageBlock):
                parts.append("[Image]")
        
        return " ".join(parts) if parts else "[empty]"

    def get_history(self) -> List[Message]:
        """Return a shallow copy of the current conversation history."""
        return list(self._messages)

    def clear(self):
        """Clear all stored messages."""
        self._messages.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory into a dictionary."""
        return {
            "max_items": self.max_items,
            "messages": [self._message_to_dict(message) for message in self._messages],
        }

    def to_json(self) -> str:
        """Serialize the memory into a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FullCompressionMemory":
        """Construct memory from a serialized dictionary payload."""
        if not payload:
            return cls()
        max_items = payload.get("max_items", 50)
        messages_data = payload.get("messages", [])
        messages = [cls._message_from_dict(item) for item in messages_data]
        return cls(max_items=max_items, messages=messages)

    @classmethod
    def from_json(cls, data: Optional[str]) -> "FullCompressionMemory":
        """Construct memory from a JSON string."""
        if not data:
            return cls()
        return cls.from_dict(json.loads(data))

    @staticmethod
    def _message_to_dict(message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": message.role.value,
            "content": FullCompressionMemory._content_to_serializable(message.content),
            "usage": message.usage,
        }
        if message.metadata:
            payload["metadata"] = message.metadata
        return payload

    @staticmethod
    def _message_from_dict(data: Dict[str, Any]) -> Message:
        role_value = data.get("role", MessageRole.USER.value)
        role = MessageRole(role_value)
        content = FullCompressionMemory._content_from_serializable(data.get("content"))
        return Message(role=role, content=content, usage=data.get("usage") or {}, metadata=data.get("metadata") or {})

    @staticmethod
    def _content_to_serializable(content: Union[str, List[ContentBlock], List[ToolResultBlock]]) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(content, str):
            return content
        return [block.to_dict() for block in content]

    @staticmethod
    def _content_from_serializable(content: Union[str, List[Dict[str, Any]], None]) -> Union[str, List[ContentBlock]]:
        if content is None:
            return []
        if isinstance(content, str):
            return content
        blocks: List[ContentBlock] = []
        for block_data in content:
            block = ContentBlock.from_dict(block_data)
            if block:
                blocks.append(block)
        return blocks
