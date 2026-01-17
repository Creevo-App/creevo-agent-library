"""
Message categorization utilities for conversation memory.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .message import Message
from .content_blocks import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@dataclass
class CategorizedMessages:
    """Result of message categorization for compression."""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # Paired tool use + result
    file_reads: List[Dict[str, Any]] = field(default_factory=list)  # File read tool results
    conversations: List[Message] = field(default_factory=list)       # Text-only exchanges
    images: List[Dict[str, Any]] = field(default_factory=list)       # Image blocks


class MessageCategorizer:
    """
    Categorizes messages for intelligent compression.
    
    Separates messages into:
    - Tool calls with their results
    - File read operations (preserved with more detail)
    - Regular conversation text
    - Images (reference only)
    """
    
    @classmethod
    def categorize(cls, messages: List[Message]) -> CategorizedMessages:
        """
        Categorize a list of messages.
        
        Args:
            messages: List of messages to categorize
            
        Returns:
            CategorizedMessages with separated content types
        """
        result = CategorizedMessages()
        
        # Track tool uses waiting for results
        pending_tool_uses: Dict[str, ToolUseBlock] = {}
        
        for msg in messages:
            if isinstance(msg.content, str):
                result.conversations.append(msg)
                continue
            
            has_text_only = True
            
            for block in msg.content:
                if isinstance(block, TextBlock):
                    # Will be included in conversation if no tool blocks
                    pass
                elif isinstance(block, ToolUseBlock):
                    has_text_only = False
                    pending_tool_uses[block.id] = block
                elif isinstance(block, ToolResultBlock):
                    has_text_only = False
                    tool_use = pending_tool_uses.pop(block.tool_use_id, None)
                    
                    tool_name = block.name or (tool_use.name if tool_use else "")
                    is_read_tool = block.metadata.get("is_read_tool", False) if block.metadata else False
                    
                    entry = {
                        "tool_use": tool_use,
                        "tool_result": block,
                        "tool_name": tool_name,
                    }
                    
                    if is_read_tool:
                        result.file_reads.append(entry)
                    else:
                        result.tool_calls.append(entry)
                        
                elif isinstance(block, ImageBlock):
                    has_text_only = False
                    result.images.append({
                        "block": block,
                        "message_role": msg.role,
                    })
            
            # If message only has text blocks, treat as conversation
            if has_text_only and msg.content:
                result.conversations.append(msg)
        
        return result
    
    @classmethod
    def extract_tools_and_files(
        cls, 
        categorized: CategorizedMessages
    ) -> tuple[List[str], List[str]]:
        """
        Extract unique tool names and key file paths from categorized messages.
        
        Returns:
            Tuple of (tool_names, file_paths)
        """
        tools_used = set()
        key_files = set()
        
        for entry in categorized.tool_calls + categorized.file_reads:
            if entry.get("tool_name"):
                tools_used.add(entry["tool_name"])
            tool_use = entry.get("tool_use")
            if tool_use and hasattr(tool_use, "input"):
                for key in ["path", "file_path", "filename", "target_file"]:
                    if key in tool_use.input:
                        key_files.add(str(tool_use.input[key]))
        
        return list(tools_used), list(key_files)
