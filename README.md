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

# Initialize memory
memory = FullCompressionMemory(max_items=50)

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

- `FullCompressionMemory`: Keeps initial prompt, summarizes middle turns, keeps recent messages

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

## Documentation

For detailed API documentation, see the source code or contact the Creevo team.

## License

MIT License
