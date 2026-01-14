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
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.sub_tools = tools
        self.sub_llm = llm
        self.sub_max_calls = max_calls
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

        # Create sub-agent with its own LLM configuration
        sub_agent = Agent(
            llm=self.sub_llm,
            system_prompt=self.system_prompt,
            max_calls=self.sub_max_calls,
            max_tokens=self._parent_agent.max_tokens,
            memory=sub_memory,
            session_id=f"{self._parent_agent.session_id}_sub_{self.name}",
            tools=list(self.sub_tools),
            logger=None,
        )

        # Run the sub-agent
        result_message = await sub_agent.run_async(task)

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
    max_calls: int = 10
):
    """
    Decorator to define a sub-agent as a tool.

    Example:
        @subagent(
            system_prompt="You are a code reviewer...",
            tools=[review_tool, lint_tool],
            llm=GeminiLLM(api_key="...", model="gemini-3-pro-preview", max_tokens=8192)
        )
        async def code_reviewer(task: str):
            '''Reviews code for issues.'''
            pass

    Args:
        system_prompt: System prompt for the sub-agent
        tools: List of tools available to the sub-agent
        llm: LLM instance to use for the sub-agent
        max_calls: Maximum tool calls for the sub-agent (default 10)

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
        )
    return decorator
