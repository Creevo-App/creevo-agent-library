# Research Assistant Agent

A CAL agent example that demonstrates building a practical research tool with web search, note-taking, and report generation capabilities.

## Features

- **Web Search**: Search the internet for information on any topic using Tavily API
- **Note Taking**: Save and organize research findings with titles and tags
- **Note Management**: List and read saved research notes
- **Report Generation**: Compile research into formatted reports

## Setup

1. Create a `.env` file with your API keys:

```bash
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API=your_tavily_api_key
```

2. Install dependencies:

```bash
pip install CAL python-dotenv httpx
```

3. Run the agent:

```bash
python agent.py
```

## Example Usage

```
You: Research the current state of quantum computing

# Agent will:
# 1. Search the web for quantum computing information
# 2. Save key findings as notes
# 3. Present a summary

You: Save a note about the key companies in this space

You: Generate a report on quantum computing
```

## File Structure

```
research_assistant/
├── agent.py       # Main agent setup and run loop
├── tools.py       # Custom tool definitions (@tool decorator)
├── prompt.py      # System prompts and guidelines
└── README.md      # This file
```

## Output Directories

- `research_notes/` - Saved research notes (Markdown files)
- `research_reports/` - Generated research reports (Markdown files)

## Key CAL Concepts Demonstrated

### 1. Custom Tools with `@tool` Decorator

```python
@tool
async def web_search(query: str):
    """Tool description for the LLM"""
    # Tool implementation
    return {
        "content": [{"type": "text", "text": "Result"}],
        "metadata": {"key": "value"}
    }
```

### 2. Memory Management with ContextPolicy

```python
memory_engine = DefaultMemoryEngine()

context_policy = ContextPolicy(
    total_token_budget=50000,
    recent_tokens=18000,
    semantic_tokens=12000,
    working_tokens=5000,
)
```

### 3. Structured System Prompts

Prompts are organized into role, guidelines, and tool usage sections for clarity.

### 4. Agent Configuration

```python
agent = Agent(
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
    max_calls=50,
    max_tokens=4096,
    memory_engine=memory_engine,
    context_policy=context_policy,
    thread_id="research-thread",
    resource_id="research-user",
    agent_name="research-assistant",
    tools=[StopTool(), web_search, ...]
)
```
