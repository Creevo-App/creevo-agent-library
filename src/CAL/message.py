"""
Message classes for CAL
"""
from enum import Enum
from typing import Union, List, TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from .content_blocks import ContentBlock


class MessageRole(Enum):
    """Enumeration for message roles"""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL_RESPONSE = "tool response"


class Message:
    """Message class containing role and content"""
    
    def __init__(
        self,
        role: MessageRole,
        content: Union[str, List['ContentBlock']],
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a message.
        
        Args:
            role: The role of the message sender (USER or ASSISTANT)
            content: The content of the message (str or list of ContentBlock)
            usage: Optional token usage information (prompt_tokens, completion_tokens, total_tokens)
        """
        self.role = role
        self.content = content
        self.usage = usage or {}
        self.metadata = metadata or {}
    
    def __repr__(self):
        return (
            f"Message(role={self.role}, content={self.content}, "
            f"usage={self.usage}, metadata={self.metadata})"
        )

