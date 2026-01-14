"""
CAL (Creevo Agent Library)
A library for managing agents, LLMs, and tools.
"""

from .agent import Agent
from .llm import LLM, AnthropicVertexLLM, GeminiLLM
from .tool import Tool, tool, StopTool
from .subagent import SubAgentTool, subagent
from .message import Message, MessageRole
from .memory import Memory, FullCompressionMemory
from .content_blocks import (
    ContentBlock,
    TextBlock,
    ImageBlock,
    ImageSource,
    ToolUseBlock,
    ToolResultBlock
)

__all__ = [
    'Agent',
    'LLM',
    'AnthropicVertexLLM',
    'GeminiLLM',
    'Tool',
    'tool',
    'StopTool',
    'SubAgentTool',
    'subagent',
    'Message',
    'MessageRole',
    'ContentBlock',
    'TextBlock',
    'ImageBlock',
    'ImageSource',
    'ToolUseBlock',
    'ToolResultBlock',
    'Memory',
    'FullCompressionMemory',
]
