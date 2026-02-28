# CAL (Creevo Agent Library)

CAL is a Python library for building agentic AI applications with LLM abstraction, tool management, and conversation memory.

## Installation

```bash
pip install creevo-agent-library
```

With MCP support:

```bash
pip install "creevo-agent-library[mcp]"
```

## Quick Start

```python
import os
from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory

# Initialize LLM (uses GEMINI_API_KEY env var when api_key is None)
llm = GeminiLLM(model="gemini-2.5-flash", api_key=None, max_tokens=4096)

# Initialize memory with a summarizer LLM (required for compression)
summarizer_llm = GeminiLLM(model="gemini-2.0-flash", api_key=None, max_tokens=2048)
memory = FullCompressionMemory(summarizer_llm=summarizer_llm, max_tokens=50000)

# Create agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    max_calls=10,
    max_tokens=4096,
    memory=memory,
    agent_name="my-agent",
    tools=[StopTool()]
)

# Run agent
result = agent.run("Hello, how can you help me?")
print(result.content)
```

## Architecture

CAL provides:

- **Agent**: Agentic loop implementation with tool execution
- **LLM**: Abstract base class with Gemini and Anthropic implementations
- **Memory**: Conversation memory management with compression support
- **Tool**: Tool system with `@tool` decorator for easy tool creation
- **Message**: Message and content block types for conversation handling
- **Logger**: OpenTelemetry and Maxim AI logging support

## Components

### Agent

The `Agent` class implements the agentic loop:
1. LLM generates response (may include tool calls)
2. Parse tool uses from response
3. Execute tools in parallel
4. Add tool results to memory
5. Repeat until completion or max iterations

### LLM Providers

- `GeminiLLM`: Google Gemini API integration
- `AnthropicVertexLLM`: Anthropic Claude via Vertex AI (stub)

### Memory

- `FullCompressionMemory`: LLM-based compression that keeps initial prompt, summarizes middle turns using an LLM, and keeps recent messages. Requires a `summarizer_llm` for compression.

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
    memory=FullCompressionMemory(
        summarizer_llm=GeminiLLM(model="gemini-3-flash-preview", api_key="...", max_tokens=2048),
        max_tokens=50000,
    ),
    agent_name="mcp-agent",
    tools=[StopTool(), *mcp_tools],  # mix with any other CAL tools
)

result = await agent.run_async("What does React useEffect do?")
await disconnect_mcp_tools(mcp_tools)
```

See [`examples/mcp_context7_agent.ipynb`](examples/mcp_context7_agent.ipynb) for a full walkthrough.

## Documentation

For detailed API documentation, see the [wiki](https://github.com/Creevo-App/creevo-agent-library/wiki).

## License

MIT License
