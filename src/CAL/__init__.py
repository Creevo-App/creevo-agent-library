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
    ArchiveSummary,
    ArchiveStore,
    CompositeMemoryObserver,
    ContextLedgerEntry,
    ContextPacket,
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
    NullMemoryObserver,
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
from .compression import CompressionConfig, CompressionArchiver
from .logger import Logger, LangSmithLogger, MaximLogger, SentryLogger
from .content_blocks import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
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
    'ThinkingBlock',
    'MemoryEngine',
    'DefaultMemoryEngine',
    'MemoryObserver',
    'NullMemoryObserver',
    'CompositeMemoryObserver',
    'InMemoryMemoryObserver',
    'LoggerMemoryObserver',
    'OTelMemoryObserver',
    'ContextPolicy',
    'ContextPacket',
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
    'ArchiveSummary',
    'ArchiveStore',
    'InMemoryConversationStore',
    'InMemoryWorkingMemoryStore',
    'InMemorySemanticMemoryStore',
    'InMemoryArchiveStore',
    'estimate_system_tokens',
    'estimate_tool_schema_tokens',
    'CompressionConfig',
    'CompressionArchiver',
    'Logger',
    'MaximLogger',
    'SentryLogger',
    'LangSmithLogger',
]

# Optional MCP support — only available when the `mcp` package is installed.
try:
    from .mcp import MCPTool, MCPToolList, MCPServerConnection, connect_mcp_server, disconnect_mcp_tools
    __all__ += ['MCPTool', 'MCPToolList', 'MCPServerConnection', 'connect_mcp_server', 'disconnect_mcp_tools']
except ImportError:
    pass
