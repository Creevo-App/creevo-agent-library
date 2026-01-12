"""
CAL (Creevo Agent Library)
A library for managing agents, LLMs, and tools.
"""

from .agent import Agent
from .llm import LLM, AnthropicVertexLLM, GeminiLLM
from .tool import Tool, tool, StopTool
from .message import Message, MessageRole
from .memory import Memory, FullCompressionMemory
from .compression import CompressionConfig, CompressionArchiver
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
    'CompressionConfig',
    'CompressionArchiver',
]
