"""
Agent classes for CAL with OpenTelemetry tracing
"""
import os
import time
import sys
import json
import asyncio
from typing import List, Optional

from .logger import Logger
from .llm import LLM
from .tool import Tool
from .message import Message, MessageRole
from .content_blocks import ToolResultBlock, TextBlock, ToolUseBlock
from .memory import Memory

PROGRESS_PREFIX = "__AGENT_PROGRESS__"

def emit_progress(session_id: str, event: str, message: str, detail: dict = None):
    """Emit a progress event to stdout for Node.js to capture and forward to frontend."""
    payload = {
        "session_id": session_id,
        "event": event,
        "message": message,
        "detail": detail or {},
        "timestamp": time.time(),
    }
    print(f"{PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


class Agent:
    """Agent class with custom tracing"""
    
    def __init__(
        self,
        llm: LLM,
        system_prompt: str,
        max_calls: int,
        max_tokens: int,
        memory: Memory,
        session_id: str,
        tools: Optional[List[Tool]] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            llm: The LLM instance to use
            system_prompt: System prompt for the agent
            max_calls: Maximum number of tool calls allowed
            max_tokens: Maximum tokens for generation
            tools: Optional list of tools to register with the agent
        """
        self.llm = llm
        self.tools: List[Tool] = tools if tools is not None else []
        self.system_prompt = system_prompt
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.memory = memory
        self.session_id = session_id
        self.logger = logger

        # Initialize logger metadata
        if self.logger:
            self.logger.log_metadata({
                "system_prompt": self.system_prompt,
                "session_id": self.session_id
            })
        
    def register_tool(self, tool: Tool):
        """
        Register a tool with the agent.
        
        Args:
            tool: The tool to register
        """
        self.tools.append(tool)
    
    @property
    def conversation_history(self) -> List[Message]:
        """Expose current conversation history."""
        return self.memory.get_history()
    
    def get_token_usage(self) -> dict:
        """
        Get cumulative token usage from all LLM responses in the conversation history.
        
        Returns:
            Dictionary with 'prompt_tokens', 'completion_tokens', and 'total_tokens'
            representing the cumulative usage across all assistant messages.
        """
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        
        for message in self.memory.get_history():
            if message.role == MessageRole.ASSISTANT and message.usage:
                usage = message.usage
                total_prompt += usage.get('prompt_tokens', 0)
                total_completion += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
        
        return {
            'prompt_tokens': total_prompt,
            'completion_tokens': total_completion,
            'total_tokens': total_tokens
        }
    
    def _history_json(self) -> str:
        """Return the serialized conversation history."""
        return self.memory.to_json()
    
    def _parse_tool_uses(self, message: Message) -> List[ToolUseBlock]:
        """
        Extract ToolUseBlocks from message content.
        
        Args:
            message: The message to parse
            
        Returns:
            List of ToolUseBlock objects
        """
        tool_uses = []
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tool_uses.append(block)
        return tool_uses
    
    def _find_tool(self, name: str) -> Optional[Tool]:
        """
        Find a tool by name in the registry.
        Normalizes names by replacing hyphens with underscores for matching.
        
        Args:
            name: Name of the tool to find
            
        Returns:
            Tool instance or None if not found
        """
        normalized_name = name.replace('-', '_')
        for tool in self.tools:
            if tool.name.replace('-', '_') == normalized_name:
                return tool
        return None
    
    async def _execute_tools(self, tool_uses: List[ToolUseBlock]) -> List[ToolResultBlock]:
        """
        Execute tools and return results.
        Captures start/end times for each tool execution.
        """
        results = []
        for tool_use in tool_uses:
            tool = self._find_tool(tool_use.name)
            
            start_time = time.time_ns()
            
            if tool is None:
                result = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=f"Error: Tool '{tool_use.name}' not found",
                    is_error=True,
                    name=tool_use.name
                )
            else:
                result = await tool.execute(tool_use_id=tool_use.id, **tool_use.input)
            
            end_time = time.time_ns()
            
            # Log tool response
            if self.logger:
                self.logger.log_tool_response(
                    tool_use, 
                    result,
                    start_time=start_time,
                    end_time=end_time
                )
                
            results.append(result)
        
        return results

    def _extract_final_output(self) -> str:
        """Helper to find the last meaningful output for the trace result. If no output is found, return an empty string."""
        # Check reverse history for the most recent relevant result
        for message in reversed(self.memory.get_history()):
            if message.role == MessageRole.USER and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock) and block.metadata:
                        if "full_span_output" in block.metadata:
                            return block.metadata["full_span_output"]
                    
        return ""
    
    def _has_tool_calls(self, message: Message) -> bool:
        """Check if a message contains tool calls."""
        if message.role != MessageRole.ASSISTANT:
            return False
        if isinstance(message.content, str):
            return False
        return any(isinstance(block, ToolUseBlock) for block in message.content)
    
    def _cleanup_incomplete_conversation(self):
        """
        Remove incomplete conversation sequences from memory.
        If the last message is an assistant message with tool calls,
        remove it since it's incomplete (no tool responses followed).
        """
        history = self.memory.get_history()
        if not history:
            return
        
        last_message = history[-1]
        if self._has_tool_calls(last_message):
            # Remove the incomplete assistant message with tool calls
            # This can happen if conversation history was saved mid-execution
            self.memory._messages.pop()
    
    
    async def run_async(self, user_prompt: str) -> Message:
        """
        Run the agent with the given user prompt (async version).
        Implements agentic loop: LLM -> Tool -> LLM -> ...
        """
        assert self.max_calls > 0, "max_calls must be positive"
        
        # Clean up any incomplete conversation sequences (e.g., from mid-execution saves)
        self._cleanup_incomplete_conversation()
        
        emit_progress(self.session_id, "start", "Got your request, analyzing your game idea...")
        
        # Start Trace
        if self.logger:
            self.logger.start_trace("agent.run", user_prompt)

        # Add user message to conversation history
        user_message = Message(role=MessageRole.USER, content=[TextBlock(text=user_prompt)])
        self.memory.add_message(user_message)
        
        iteration = 0
        last_agent_message = None
        workflow_status = "completed_success"
        
        try:
            while iteration < self.max_calls:
                
                # Step 1: Generate LLM response with Timing
                emit_progress(
                    self.session_id,
                    "llm_start",
                    f"Step {iteration + 1}: Thinking...",
                    {"iteration": iteration}
                )
                llm_start_time = time.time_ns()
                
                agent_message = self.llm.generate_content(
                    self.system_prompt,
                    self.memory.get_history(),
                    self.tools,
                )
                
                llm_end_time = time.time_ns()
                last_agent_message = agent_message
                
                emit_progress(
                    self.session_id,
                    "llm_end",
                    f"Step {iteration + 1}: Planning complete.",
                    {"iteration": iteration}
                )
                
                # Log LLM response
                if self.logger:
                    self.logger.log_llm_response(
                        agent_message, 
                        iteration, 
                        model=self.llm.name, 
                        provider=self.llm.provider,
                        start_time=llm_start_time,
                        end_time=llm_end_time
                    )

                # Step 2: Add agent message to conversation history
                self.memory.add_message(agent_message)
                
                # Check MAX_TOKENS
                if hasattr(agent_message, 'metadata') and agent_message.metadata:
                    finish_reason = agent_message.metadata.get('finish_reason')
                    if finish_reason and 'MAX_TOKENS' in finish_reason:
                        workflow_status = "completed_max_tokens"
                        print("Hit MAX_TOKENS, stopping agent loop", file=sys.stderr)
                        break
                
                # Step 3: Parse tool uses
                tool_uses = self._parse_tool_uses(agent_message)
                
                if not tool_uses:
                    workflow_status = "completed_no_tools"
                    break
                
                emit_progress(
                    self.session_id,
                    "tool_start",
                    f"Step {iteration + 1}: Running tools...",
                    {
                        "iteration": iteration,
                        "tools": [tu.name for tu in tool_uses],
                    }
                )
                
                # Step 4: Execute tools (Internal logic handles timing logging)
                tool_results = await self._execute_tools(tool_uses)
                
                emit_progress(
                    self.session_id,
                    "tool_end",
                    f"Step {iteration + 1}: Tools complete.",
                    {
                        "iteration": iteration,
                        "tools": [tu.name for tu in tool_uses],
                    }
                )
                
                # Step 5: Check if stop was called
                stop_called = any(tu.name == "stop" for tu in tool_uses)
                
                # Step 6: Add results to history
                tool_message = Message(role=MessageRole.USER, content=tool_results)
                self.memory.add_message(tool_message)
                
                if stop_called:
                    workflow_status = "completed_stop"
                    break
                
                iteration += 1
                
            if iteration >= self.max_calls:
                workflow_status = "completed_max_iterations"
                
        except Exception as e:
            workflow_status = f"error: {str(e)}"
            raise e
        finally:
            status_msg = {
                "completed_success": "Game ready soon – finalizing details...",
                "completed_no_tools": "Finished reasoning about your game.",
                "completed_max_tokens": "Reached internal limit, finishing up...",
                "completed_max_iterations": "Finished maximum number of planning steps.",
                "completed_stop": "Completed building your game!",
            }.get(workflow_status, "Wrapping up...")

            emit_progress(
                self.session_id,
                "complete",
                status_msg,
                {"workflow_status": workflow_status}
            )
            
            # End Trace
            if self.logger:
                final_output = self._extract_final_output()
                
                # Fallback: if extractor failed but we have a message, try to stringify safely
                if not final_output and last_agent_message:
                     content = last_agent_message.content
                     if isinstance(content, list):
                         # Robust string extraction from blocks
                         parts = []
                         for x in content:
                             if hasattr(x, 'text'):
                                 parts.append(x.text)
                             else:
                                 parts.append(str(x))
                         final_output = " ".join(parts)
                     else:
                         final_output = str(content)

                self.logger.end_trace(
                    output=str(final_output),
                    metadata={
                        "status": workflow_status,
                        "total_iterations": iteration
                    }
                )

        # last_agent_message is guaranteed to be set since max_calls > 0
        return last_agent_message


    def run(self, user_prompt: str) -> Message:
        """
        Sync wrapper for run_async.
        Safely handles existing event loops to prevent RuntimeError.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Agent.run() was called from a running event loop. "
                "Use 'await agent.run_async()' instead."
            )
        
        try:
            return asyncio.run(self.run_async(user_prompt))
        except Exception as e:
            raise RuntimeError(f"Agent execution failed: {e}") from e