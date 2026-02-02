"""SubAgent classes for CAL - enables multi-agent delegation."""

import inspect
from typing import Callable, List

from google.genai import types

from .agent import Agent
from .tool import Tool
from .content_blocks import ToolResultBlock, TextBlock
from .llm import LLM


class SubAgentTool(Tool):
    """A tool that spawns a sub-agent to handle delegated tasks."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: List[Tool],
        llm: LLM,
        max_calls: int = 10,
        max_tokens: int = None,
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.sub_tools = tools
        self.sub_llm = llm
        self.sub_max_calls = max_calls
        self.sub_max_tokens = max_tokens
        self._parent_agent: Agent = None
        self.input_schema = {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "The task to delegate to the sub-agent"}},
            "required": ["task"]
        }

    def bind_parent(self, parent_agent: Agent):
        """Bind to parent agent for context access."""
        self._parent_agent = parent_agent

    def get_schema(self) -> dict:
        """Return schema in Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }

    def gemini_input_form(self):
        """Convert tool schema to Gemini format."""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=self.input_schema
                )
            ]
        )

    async def execute(self, **kwargs) -> ToolResultBlock:
        """Execute by spawning a sub-agent with the parent's context."""
        tool_use_id = kwargs.pop('tool_use_id', 'stub_tool_use_id')
        task = kwargs.get('task', '')

        if self._parent_agent is None:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Error: SubAgentTool '{self.name}' not bound to parent agent",
                is_error=True,
                name=self.name
            )

        # Clone parent's memory to give sub-agent full context
        sub_memory = self._parent_agent.memory.clone()

        # Apply sub-agent's max_tokens to the cloned memory for correct compression threshold.
        # Priority: 1) explicit sub_max_tokens, 2) parent memory's max_tokens (compression threshold),
        # 3) fallback to parent agent's max_tokens (for custom Memory implementations without max_tokens).
        # Note: The abstract Memory base class doesn't define max_tokens—only FullCompressionMemory does.
        # Custom Memory implementations may omit it, so we check with hasattr before accessing.
        if self.sub_max_tokens is not None:
            effective_max_tokens = self.sub_max_tokens
        elif hasattr(self._parent_agent.memory, 'max_tokens'):
            effective_max_tokens = self._parent_agent.memory.max_tokens
        else:
            effective_max_tokens = self._parent_agent.max_tokens
        if hasattr(sub_memory, 'max_tokens'):
            sub_memory.max_tokens = effective_max_tokens

        # Update agent_name and archiver to use sub-agent's identity (not parent's)
        sub_agent_name = f"{self._parent_agent.agent_name}_sub_{self.name}"
        if hasattr(sub_memory, 'agent_name'):
            sub_memory.agent_name = sub_agent_name
        if hasattr(sub_memory, 'archiver'):
            from .compression import CompressionArchiver
            sub_memory.archiver = CompressionArchiver(agent_name=sub_agent_name)

        # Create child logger for nested span logging
        child_logger = None
        if self._parent_agent.logger:
            child_logger = self._parent_agent.logger.create_child_logger(self.name)

        # Update sub_memory's logger so compression events are logged under the subagent,
        # not the parent agent. The cloned memory inherits the parent's logger by default.
        if hasattr(sub_memory, 'logger'):
            sub_memory.logger = child_logger

        # Create sub-agent with its own LLM configuration
        sub_agent = Agent(
            llm=self.sub_llm,
            system_prompt=self.system_prompt,
            max_calls=self.sub_max_calls,
            max_tokens=effective_max_tokens,
            memory=sub_memory,
            agent_name=sub_agent_name,
            tools=list(self.sub_tools),
            logger=child_logger,
        )

        # Run the sub-agent
        try:
            result_message = await sub_agent.run_async(task)
        finally:
            # End the child logger's wrapper span
            if child_logger:
                child_logger.end_child()

        # Pass through the content directly (ToolResultBlock accepts str or List[ContentBlock])
        content = result_message.content
        if isinstance(content, list):
            # Filter to only TextBlocks, there could be a ToolUseBlock if the agent hit the max call limit.
            content = [b for b in content if isinstance(b, TextBlock)] or [TextBlock(text="[No text response from sub-agent]")]

        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=False,
            name=self.name
        )

    def __repr__(self):
        return f"SubAgentTool(name={self.name})"


def subagent(
    system_prompt: str,
    tools: List[Tool],
    llm: LLM,
    max_calls: int = 10,
    max_tokens: int = None,
):
    """
    Decorator to define a sub-agent as a tool.

    Example:
        @subagent(
            system_prompt="You are a code reviewer...",
            tools=[review_tool, lint_tool],
            llm=GeminiLLM(api_key="...", model="gemini-3-pro-preview", max_tokens=8192),
            max_tokens=50000
        )
        async def code_reviewer(task: str):
            '''Reviews code for issues.'''
            pass

    Args:
        system_prompt: System prompt for the sub-agent
        tools: List of tools available to the sub-agent
        llm: LLM instance to use for the sub-agent
        max_calls: Maximum tool calls for the sub-agent (default 10)
        max_tokens: Maximum tokens for sub-agent memory (default: inherit from parent)

    Returns:
        SubAgentTool instance
    """
    def decorator(func: Callable) -> SubAgentTool:
        return SubAgentTool(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            system_prompt=system_prompt,
            tools=tools,
            llm=llm,
            max_calls=max_calls,
            max_tokens=max_tokens,
        )
    return decorator
