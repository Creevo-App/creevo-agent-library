"""SubAgent classes for CAL - enables multi-agent delegation."""

import inspect
import uuid
from typing import Callable, List

from google.genai import types

from .agent import Agent
from .tool import Tool
from .content_blocks import ToolResultBlock, TextBlock
from .llm import LLM
from .memory_engine import ContextPolicy, TurnRecord


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

        sub_agent_name = f"{self._parent_agent.agent_name}_sub_{self.name}"

        # Apply sub-agent max_tokens preference when provided.
        if self.sub_max_tokens is not None:
            effective_max_tokens = self.sub_max_tokens
        else:
            effective_max_tokens = self._parent_agent.max_tokens

        # Create child logger for nested span logging
        # Pass sub_agent_name so logging metadata reflects the subagent, not the parent
        child_logger = None
        if self._parent_agent.logger:
            child_logger = self._parent_agent.logger.create_child_logger(
                self.name,
                agent_name=sub_agent_name
            )

        # Create sub-agent with its own LLM configuration.
        parent_policy = self._parent_agent.context_policy or ContextPolicy()
        sub_policy = ContextPolicy(**parent_policy.__dict__)
        if self.sub_max_tokens is not None:
            sub_policy.total_token_budget = max(1024, int(self.sub_max_tokens))

        # Each invocation gets a unique thread_id so that conversation history
        # from previous invocations of the same subagent tool doesn't leak in.
        invocation_thread_id = f"{sub_agent_name}:{uuid.uuid4().hex[:12]}"

        sub_agent = Agent(
            llm=self.sub_llm,
            system_prompt=self.system_prompt,
            max_calls=self.sub_max_calls,
            max_tokens=effective_max_tokens,
            memory_engine=self._parent_agent.memory_engine,
            context_policy=sub_policy,
            agent_name=sub_agent_name,
            thread_id=invocation_thread_id,
            resource_id=self._parent_agent.resource_id,
            tools=list(self.sub_tools),
            logger=child_logger,
        )

        # Seed the child thread with the parent's recent conversation history
        # so the sub-agent has context about what the parent has been doing.
        # Limit to the most recent turns to avoid triggering expensive archival
        # operations when the sub-agent records its first real turn.
        parent_turns = await self._parent_agent.memory_engine.conversation_store.all(
            self._parent_agent.thread_id
        )
        max_seed_turns = max(4, getattr(self._parent_agent.memory_engine, 'archive_cold_threshold', 60) // 2)
        parent_turns = parent_turns[-max_seed_turns:]
        for turn in parent_turns:
            seeded_turn = TurnRecord(
                run_id=turn.run_id,
                turn_id=f"{invocation_thread_id}:seed:{turn.turn_id}",
                thread_id=invocation_thread_id,
                resource_id=turn.resource_id,
                message=turn.message,
                metadata={**turn.metadata, "seeded_from": self._parent_agent.thread_id},
            )
            await self._parent_agent.memory_engine.conversation_store.append(seeded_turn)

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
