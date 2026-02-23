# CAL Framework Guidelines

## Architecture Overview

- CAL is a Python library for building agentic AI applications
- Main components: `Agent`, `LLM` (GeminiLLM), `DefaultMemoryEngine`, `Tool`, `Message`, `SubAgentTool`
- Package: `creevo-agent-library` (installed from GitHub: https://github.com/Creevo-App/creevo-agent-library)

## Memory Management

- `DefaultMemoryEngine` is the only memory system — scoped threads, semantic recall, working memory, archive summaries
- Agent auto-creates a `DefaultMemoryEngine` if none is provided
- Optional `summarizer_llm` param on `DefaultMemoryEngine` enables LLM-based archive compression (falls back to naive text truncation)
- Optional `archiver` param writes full archived context to disk via `CompressionArchiver`
- SubAgents share the parent's engine with separate `thread_id` and shared `resource_id`
- Migration utils (`migrate_legacy_memory_json`, `migrate_legacy_memory_payload`) available for one-time import of legacy payloads

## SubAgent Architecture

SubAgents enable multi-agent delegation where a main agent spawns specialized sub-agents via tool calls.

### Defining SubAgents

Use the `@subagent` decorator to define a subagent as its own tool:

```python
from CAL import subagent, tool
from CAL import GeminiLLM

@tool
async def review_file(path: str):
    """Review a file."""
    return {"content": [{"type": "text", "text": "..."}], "metadata": {}}

@subagent(
    system_prompt="You are a code reviewer...",
    tools=[review_file],
    llm=GeminiLLM(api_key="...", model="gemini-2.0-flash-exp", max_tokens=8192),
    max_calls=5
)
async def code_reviewer(task: str):
    """Delegate code review to a specialized sub-agent."""
    pass  # Body unused - decorator handles execution
```

### Key Behaviors

- SubAgents share the parent's `memory_engine` with a separate `thread_id` for isolation and shared `resource_id` for cross-agent semantic recall
- Each subagent is its own distinct tool with predefined `system_prompt`, `tools`, and `llm` (model configuration)
- Subagents use their own LLM instance, allowing different models for different subagents
- Nesting is supported (subagents can include other subagents in their tools list)
- `Agent.__init__` automatically binds SubAgentTools to the parent agent

### Module Structure

- `src/CAL/subagent.py` contains `SubAgentTool` and `@subagent` decorator
- Kept separate from `src/CAL/tool.py` to avoid circular imports with `src/CAL/agent.py`

## Circular Import Pattern

When modules have circular dependencies:

1. **Preferred**: Move dependent code to a separate module that can import both (e.g., `subagent.py` imports from both `agent.py` and `tool.py`)
2. **Fallback**: Use local imports inside methods when circular imports at module load time are unavoidable

Example from `Agent.__init__`:
```python
# Local import to avoid circular import at module load time
from .subagent import SubAgentTool
for tool in self.tools:
    if isinstance(tool, SubAgentTool):
        tool.bind_parent(self)
```

## Gemini API Requirements

### Conversation History Structure

- Gemini requires function calls to immediately follow a user turn or function response turn
- Tool responses map to role `"user"` (not a separate role)
- Assistant messages map to role `"model"`
- Messages with same role are merged (parts combined)

### Common Errors

- `400 INVALID_ARGUMENT: function call turn comes immediately after a user turn or after a function response turn`
  - Cause: Conversation history has invalid sequence (e.g., consecutive model messages)
  - Fix: Ensure history ends with user message before LLM call; clean up incomplete tool call sequences

## Tool Implementation

- Use `@tool` decorator for regular tool functions
- Use `@subagent` decorator for delegating to specialized sub-agents
- Tools are async functions returning dict with `content` and `metadata` keys
- Content should be a list of content blocks: `[{"type": "text", "text": "..."}]`
- `ToolResultBlock.content` accepts either a string or `List[ContentBlock]`

## Debugging LLM Issues

1. **Add visibility first**: Before fixing API errors, add debug logging to see exact request structure
2. **Check conversation sequence**: Print role sequence being sent (`user -> model -> user -> ...`)
3. **Avoid over-engineering**: Don't add complex validation layers without understanding the root cause
4. **Minimal fixes**: Make targeted changes, test, iterate
