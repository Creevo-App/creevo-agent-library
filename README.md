# CAL (Creevo Agent Library)

CAL is a Python library for building agentic AI applications with LLM abstraction, tool management, and conversation memory.

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
from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory
from CAL.logger import MaximLogger

# Initialize LLM
llm = GeminiLLM(model='gemini-3-pro-preview', api_key=None, max_tokens=4096)

# Initialize memory with a summarizer LLM (required for compression)
summarizer_llm = GeminiLLM(model='gemini-3-flash-preview', api_key=None, max_tokens=2048)
memory = FullCompressionMemory(summarizer_llm=summarizer_llm, max_tokens=50000)

# Initialize logger
logger = MaximLogger(agent_name="my-session")

# Create agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    max_calls=250,
    max_tokens=4096,
    memory=memory,
    agent_name="my-session",
    logger=logger,
    tools=[StopTool()]
)

# Run agent
result = agent.run("Hello, how can you help me?")
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

Use the `@tool` decorator to create tools:

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

### MCP (Model Context Protocol)

Connect to external MCP servers and use their tools inside a CAL agent:

```python
from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory
from CAL.mcp import connect_mcp_server, disconnect_mcp_tools

async def main():
    # Connect to an MCP server (e.g. Context7)
    mcp_tools = await connect_mcp_server(command="npx", args=["-y", "@upstash/context7-mcp"])

    agent = Agent(
        llm=GeminiLLM(model="gemini-2.5-flash", api_key="...", max_tokens=4096),
        system_prompt="You are a helpful assistant.",
        max_calls=10,
        max_tokens=4096,
        memory=FullCompressionMemory(
            summarizer_llm=GeminiLLM(model="gemini-2.5-flash", api_key="...", max_tokens=2048),
            max_tokens=50000,
        ),
        agent_name="mcp-agent",
        tools=[StopTool(), *mcp_tools],
    )

    result = await agent.run_async("What does React useEffect do?")

    # Clean up MCP server connections
    await disconnect_mcp_tools(mcp_tools)
```

Install with MCP support:

```bash
pip install "creevo-agent-library[mcp] @ git+https://github.com/Creevo-App/creevo-agent-library.git"
```

## Documentation

For detailed API documentation, see the source code or contact the Creevo team.

## License

MIT License
