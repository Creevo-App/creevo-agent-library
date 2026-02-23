"""
CAL (Creevo Agent Library)
A library for managing agents, LLMs, and tools.
"""

from .agent import Agent
from .llm import LLM, AnthropicVertexLLM, GeminiLLM
from .tool import Tool, tool, StopTool
from .subagent import SubAgentTool, subagent
from .message import Message, MessageRole
from .memory_engine import (
    ArchiveStore,
    ContextLedgerEntry,
    ContextPolicy,
    ContextRequest,
    ContextSourceType,
    DefaultMemoryEngine,
    InMemoryArchiveStore,
    InMemoryConversationStore,
    InMemoryMemoryObserver,
    InMemorySemanticMemoryStore,
    InMemoryWorkingMemoryStore,
    LoggerMemoryObserver,
    MemoryEngine,
    MemoryHealthSnapshot,
    MemoryObserver,
    MemoryScope,
    OTelMemoryObserver,
    RecallItem,
    RecallQuery,
    RecallResult,
    RetentionPolicy,
    TurnRecord,
    WorkingMemorySnapshot,
    WorkingMemoryUpdate,
    estimate_system_tokens,
    estimate_tool_schema_tokens,
)
from .migrations import migrate_legacy_memory_json, migrate_legacy_memory_payload
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
    'MemoryEngine',
    'DefaultMemoryEngine',
    'MemoryObserver',
    'InMemoryMemoryObserver',
    'LoggerMemoryObserver',
    'OTelMemoryObserver',
    'ContextPolicy',
    'ContextRequest',
    'ContextSourceType',
    'ContextLedgerEntry',
    'MemoryScope',
    'TurnRecord',
    'WorkingMemorySnapshot',
    'WorkingMemoryUpdate',
    'RecallQuery',
    'RecallItem',
    'RecallResult',
    'MemoryHealthSnapshot',
    'RetentionPolicy',
    'ArchiveStore',
    'InMemoryConversationStore',
    'InMemoryWorkingMemoryStore',
    'InMemorySemanticMemoryStore',
    'InMemoryArchiveStore',
    'estimate_system_tokens',
    'estimate_tool_schema_tokens',
    'migrate_legacy_memory_json',
    'migrate_legacy_memory_payload',
    'CompressionConfig',
    'CompressionArchiver',
]

# Optional MCP support — only available when the `mcp` package is installed.
try:
    from .mcp import MCPTool, MCPToolList, MCPServerConnection, connect_mcp_server, disconnect_mcp_tools
    __all__ += ['MCPTool', 'MCPToolList', 'MCPServerConnection', 'connect_mcp_server', 'disconnect_mcp_tools']
except ImportError:
    pass
