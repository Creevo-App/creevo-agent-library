# CAL (Creevo Agent Library)

CAL is a Python library for building agentic AI applications with LLM abstraction, tool management, and scoped memory/context orchestration.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/Creevo-App/creevo-agent-library.git
```

Or for a specific version:

```bash
pip install git+https://github.com/Creevo-App/creevo-agent-library.git@v0.1.0
```

## Quick Start

```python
from CAL import (
    Agent,
    ContextPolicy,
    DefaultMemoryEngine,
    GeminiLLM,
    StopTool,
)
from CAL.logger import MaximLogger

# Initialize LLM
llm = GeminiLLM(model='gemini-3-pro-preview', api_key=None, max_tokens=4096)

# Initialize v2 memory engine + context policy
memory_engine = DefaultMemoryEngine()
context_policy = ContextPolicy(
    total_token_budget=50000,
    recent_tokens=18000,
    semantic_tokens=12000,
    working_tokens=5000,
)

# Initialize logger
logger = MaximLogger(agent_name="my-session")

# Create agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    max_calls=250,
    max_tokens=4096,
    memory_engine=memory_engine,
    context_policy=context_policy,
    thread_id="thread-123",
    resource_id="user-42",
    agent_name="my-session",
    logger=logger,
    tools=[StopTool()],
)

# Run agent
result = agent.run("Hello, how can you help me?")
```

## Architecture

CAL provides:

- **Agent**: Agentic loop implementation with tool execution
- **LLM**: Abstract base class with Gemini and Anthropic implementations
- **Memory Engine (v2)**: Scoped thread/resource memory with working memory, semantic recall, archive summaries, and deterministic context assembly
- **Tool**: Tool system with `@tool` decorator for easy tool creation
- **Message**: Message and content block types for conversation handling
- **Observability**: OpenTelemetry + logger metadata + optional ClickHouse event sink

## Components

### Agent

The `Agent` class implements the agentic loop:
1. Build a bounded context packet from `MemoryEngine`
2. Call the LLM with context
3. Persist turns and memory signals
4. Execute tools
5. Persist tool results and iterate until completion

### Memory Engine (v2)

`DefaultMemoryEngine` includes:
1. **Conversation Store**: Immutable turn log by `thread_id`
2. **Working Memory Store**: Structured JSON state by scope (`thread`, `resource`)
3. **Semantic Memory Store**: Distilled facts for recall with TTL
4. **Archive Store**: Cold history summaries for long-running threads
5. **Context Ledger**: Per-turn explainability for token usage and truncation decisions

### Observability

Observers:
- `InMemoryMemoryObserver`
- `LoggerMemoryObserver`
- `OTelMemoryObserver`
- `ClickHouseMemoryObserver`

Artifacts:
- `build/observability/clickhouse_schema.sql`
- `build/observability/alerts.md`

### LLM Providers

- `GeminiLLM`: Google Gemini API integration
- `AnthropicVertexLLM`: Anthropic Claude via Vertex AI (stub)

### Tools

CAL provides three ways to give an agent tools:

**`@tool` decorator** — wrap an async function as a tool:

```python
from CAL import tool

@tool
async def my_tool(param1: str, param2: int):
    """Tool description"""
    return {
        "content": [{"type": "text", "text": f"Result: {param1}, {param2}"}],
        "metadata": {}
    }
```

**`@subagent` decorator** — delegate to a specialized sub-agent (see `src/CAL/subagent.py`).

**MCP servers** — connect to any [Model Context Protocol](https://modelcontextprotocol.io/) server
and use its tools as regular CAL tools (install with `pip install "creevo-agent-library[mcp]"`):

```python
from CAL.mcp import connect_mcp_server, disconnect_mcp_tools

# Returns a list of MCPTool instances — same interface as @tool and @subagent
mcp_tools = await connect_mcp_server(command="npx", args=["-y", "@upstash/context7-mcp"])

agent = Agent(
    llm=GeminiLLM(model="gemini-3-flash-preview", api_key="...", max_tokens=4096),
    system_prompt="You are a helpful assistant.",
    max_calls=10,
    max_tokens=4096,
    agent_name="mcp-agent",
    tools=[StopTool(), *mcp_tools],  # mix with any other CAL tools
)

result = await agent.run_async("What does React useEffect do?")
await disconnect_mcp_tools(mcp_tools)
```

See [`examples/mcp_context7_agent.ipynb`](examples/mcp_context7_agent.ipynb) for a full walkthrough.

## Migration

For one-time migration from legacy serialized memory payloads:

- `src/CAL/migrations/migrate_legacy_memory.py`
- `migrate_legacy_memory_json(...)`
- `migrate_legacy_memory_payload(...)`

## Documentation

For API details, see source modules under `src/CAL`.

## License

MIT License
