"""
Agent classes for CAL with OpenTelemetry tracing.
"""
import asyncio
import json
import queue
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from .content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from .llm import LLM
from .logger import Logger
from .memory_engine import (
    CompositeMemoryObserver,
    ContextPolicy,
    ContextRequest,
    DefaultMemoryEngine,
    LoggerMemoryObserver,
    MemoryEngine,
    OTelMemoryObserver,
    TurnRecord,
    estimate_system_tokens,
    estimate_tool_schema_tokens,
)
from .message import Message, MessageRole
from .tool import Tool

PROGRESS_PREFIX = "__AGENT_PROGRESS__"


def emit_progress(agent_name: str, event: str, message: str, detail: dict = None):
    """Emit a progress event to stdout for Node.js to capture and forward to frontend."""
    payload = {
        "agent_name": agent_name,
        "event": event,
        "message": message,
        "detail": detail or {},
        "timestamp": time.time(),
    }
    print(f"{PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


class Agent:
    """Agent class using DefaultMemoryEngine for conversation management."""

    def __init__(
        self,
        llm: LLM,
        system_prompt: str,
        max_calls: int,
        max_tokens: int,
        agent_name: str = "session",
        tools: Optional[List[Tool]] = None,
        logger: Optional[Logger] = None,
        memory_engine: Optional[MemoryEngine] = None,
        context_policy: Optional[ContextPolicy] = None,
        thread_id: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        """Initialize the agent."""
        self.llm = llm
        self.tools: List[Tool] = tools if tools is not None else []
        self.system_prompt = system_prompt
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.agent_name = agent_name
        self.logger = logger
        self._context_queue = queue.Queue()

        self.context_policy = context_policy or ContextPolicy(total_token_budget=max(1024, max_tokens * 12))
        self.thread_id = thread_id or agent_name
        self.resource_id = resource_id or agent_name

        self.memory_engine = memory_engine
        if self.memory_engine is None:
            self.memory_engine = DefaultMemoryEngine(context_policy=self.context_policy)
            # Install default observers only on engines we created ourselves.
            # When a memory_engine is passed in (e.g. shared from a parent agent),
            # we must not mutate its observers — the owner is responsible for setup.
            observer_chain = []
            if self.logger is not None:
                observer_chain.append(LoggerMemoryObserver(self.logger))
            observer_chain.append(OTelMemoryObserver())
            self.memory_engine.observer = CompositeMemoryObserver(observer_chain)

        self._latest_thread_messages: List[Message] = []

        # Bind SubAgentTools to this agent for context access
        from .subagent import SubAgentTool

        for tool in self.tools:
            if isinstance(tool, SubAgentTool):
                tool.bind_parent(self)

        if self.logger:
            self.logger.log_metadata(
                {
                    "system_prompt": self.system_prompt,
                    "agent_name": self.agent_name,
                    "thread_id": self.thread_id,
                    "resource_id": self.resource_id,
                    "memory_mode": "v2",
                }
            )

    def register_tool(self, tool: Tool):
        """Register a tool with the agent."""
        self.tools.append(tool)

    @property
    def conversation_history(self) -> List[Message]:
        """Expose current conversation history."""
        return list(self._latest_thread_messages)

    def _history_json(self) -> str:
        """Return serialized conversation history."""
        history = self.conversation_history
        payload = [
            {
                "role": msg.role.value,
                "content": str(msg.content),
                "usage": msg.usage,
                "metadata": msg.metadata,
            }
            for msg in history
        ]
        return json.dumps(payload, ensure_ascii=True)

    def _parse_tool_uses(self, message: Message) -> List[ToolUseBlock]:
        """Extract ToolUseBlocks from message content."""
        tool_uses = []
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tool_uses.append(block)
        return tool_uses

    def _find_tool(self, name: str) -> Optional[Tool]:
        """Find a tool by name in the registry."""
        normalized_name = name.replace("-", "_")
        for tool in self.tools:
            if tool.name.replace("-", "_") == normalized_name:
                return tool
        return None

    async def _execute_tools(self, tool_uses: List[ToolUseBlock]) -> List[ToolResultBlock]:
        """Execute tools and return results."""
        results = []
        for tool_use in tool_uses:
            tool = self._find_tool(tool_use.name)
            start_time = time.time_ns()

            if tool is None:
                result = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=f"Error: Tool '{tool_use.name}' not found",
                    is_error=True,
                    name=tool_use.name,
                )
            else:
                result = await tool.execute(tool_use_id=tool_use.id, **tool_use.input)

            end_time = time.time_ns()
            if self.logger:
                self.logger.log_tool_response(tool_use, result, start_time=start_time, end_time=end_time)
            results.append(result)
        return results

    def _extract_final_output(self) -> str:
        """Find the most recent tool output metadata marker or fallback."""
        history = self.conversation_history
        for message in reversed(history):
            if message.role == MessageRole.USER and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock) and block.metadata and "full_span_output" in block.metadata:
                        return block.metadata["full_span_output"]
        return ""

    def _has_tool_calls(self, message: Message) -> bool:
        """Check if a message contains tool calls."""
        if message.role != MessageRole.ASSISTANT:
            return False
        if isinstance(message.content, str):
            return False
        return any(isinstance(block, ToolUseBlock) for block in message.content)

    def push_context(self, context: str) -> None:
        """Push additional context into the agent. Thread-safe."""
        self._context_queue.put(context)

    def _drain_context_queue(self) -> List[str]:
        pushed: List[str] = []
        while True:
            try:
                pushed.append(self._context_queue.get_nowait())
            except queue.Empty:
                break
        return pushed

    async def _record_v2_turn(
        self,
        run_id: str,
        seq: int,
        message: Message,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        turn_id = f"{run_id}:{seq:04d}"
        turn = TurnRecord(
            run_id=run_id,
            turn_id=turn_id,
            thread_id=self.thread_id,
            resource_id=self.resource_id,
            message=message,
            metadata=metadata or {},
        )
        await self.memory_engine.record_turn(turn)
        self._latest_thread_messages = await self.memory_engine.get_thread_turns(self.thread_id)
        return seq + 1

    async def run_async(self, user_prompt: str) -> Message:
        """Run the agent asynchronously using the memory engine."""
        assert self.max_calls > 0, "max_calls must be positive"
        run_id = uuid.uuid4().hex
        max_retries = 10
        retries = 0
        error_streak = 0  # consecutive API errors, used for backoff delay calculation

        emit_progress(self.agent_name, "start", "Got your request, analyzing...")


        if self.logger:
            self.logger.start_trace("agent.run", user_prompt)

        iteration = 0
        turn_seq = 0
        last_agent_message: Optional[Message] = None
        workflow_status = "completed_success"

        user_message = Message(role=MessageRole.USER, content=[TextBlock(text=user_prompt)])
        turn_seq = await self._record_v2_turn(run_id, turn_seq, user_message, metadata={"event": "initial_prompt"})

        try:
            while iteration < self.max_calls:
                pushed_contexts = self._drain_context_queue()
                for context_text in pushed_contexts:
                    context_message = Message(role=MessageRole.USER, content=[TextBlock(text=context_text)])
                    turn_seq = await self._record_v2_turn(
                        run_id,
                        turn_seq,
                        context_message,
                        metadata={"event": "pushed_context"},
                    )

                emit_progress(
                    self.agent_name,
                    "llm_start",
                    f"Step {iteration + 1}: Thinking...",
                    {"iteration": iteration},
                )
                llm_start_time = time.time_ns()

                context_query = user_prompt
                if pushed_contexts:
                    context_query = f"{context_query}\n\nExternal context:\n" + "\n".join(pushed_contexts)

                context_request = ContextRequest(
                    run_id=run_id,
                    turn_id=f"{run_id}:ctx:{iteration:04d}",
                    agent_name=self.agent_name,
                    thread_id=self.thread_id,
                    resource_id=self.resource_id,
                    query=context_query,
                    policy=self.context_policy,
                    system_prompt_tokens=estimate_system_tokens(self.system_prompt),
                    tool_schema_tokens=estimate_tool_schema_tokens(self.tools),
                )
                context_packet = await self.memory_engine.build_context(context_request)

                try:
                    agent_message = self.llm.generate_content(
                        self.system_prompt,
                        context_packet.messages,
                        self.tools,
                    )
                except Exception as exc:
                    retries += 1
                    error_streak += 1
                    if retries < max_retries:
                        delay = min(2 ** (error_streak - 1), 60)
                        print(f"LLM error: {exc}, retry {retries}/{max_retries} after {delay}s", file=sys.stderr)
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"LLM call failed after {max_retries} retries: {exc}") from exc

                llm_end_time = time.time_ns()
                last_agent_message = agent_message

                emit_progress(
                    self.agent_name,
                    "llm_end",
                    f"Step {iteration + 1}: Planning complete.",
                    {
                        "iteration": iteration,
                        "context_tokens": context_packet.total_tokens,
                        "context_truncated": context_packet.truncated,
                        "context_degraded_mode": context_packet.degraded_mode,
                    },
                )

                if self.logger:
                    self.logger.log_llm_response(
                        agent_message,
                        iteration,
                        model=self.llm.name,
                        provider=self.llm.provider,
                        start_time=llm_start_time,
                        end_time=llm_end_time,
                    )

                # Check for error/limit conditions before persisting.
                # MAX_TOKENS: persist the (partial) response, then stop.
                # MALFORMED_FUNCTION_CALL: skip persistence on retries to avoid
                # polluting the conversation store; only persist on final failure.
                if hasattr(agent_message, "metadata") and agent_message.metadata:
                    finish_reason = agent_message.metadata.get("finish_reason")
                    if finish_reason and "MAX_TOKENS" in str(finish_reason):
                        workflow_status = "completed_max_tokens"
                        print("Hit MAX_TOKENS, stopping agent loop", file=sys.stderr)
                        turn_seq = await self._record_v2_turn(
                            run_id, turn_seq, agent_message,
                            metadata={"iteration": iteration, "event": "max_tokens"},
                        )
                        break
                    if finish_reason and "MALFORMED_FUNCTION_CALL" in str(finish_reason):
                        retries += 1
                        error_streak += 1
                        if retries < max_retries:
                            delay = min(2 ** (error_streak - 1), 60)
                            print(f"MALFORMED_FUNCTION_CALL detected, retry {retries}/{max_retries} after {delay}s", file=sys.stderr)
                            await asyncio.sleep(delay)
                            continue
                        # Final failure — persist so the response is visible for debugging
                        turn_seq = await self._record_v2_turn(
                            run_id, turn_seq, agent_message,
                            metadata={"iteration": iteration, "event": "malformed_function_call"},
                        )
                        workflow_status = "error_malformed_function_call"
                        break

                # Successful LLM response — reset backoff streak and persist
                error_streak = 0

                turn_seq = await self._record_v2_turn(
                    run_id,
                    turn_seq,
                    agent_message,
                    metadata={
                        "iteration": iteration,
                        "context_tokens": context_packet.total_tokens,
                        "token_usage_by_source": context_packet.token_usage_by_source,
                    },
                )


                tool_uses = self._parse_tool_uses(agent_message)
                if not tool_uses:
                    workflow_status = "completed_no_tools"
                    break

                retries = 0
                emit_progress(
                    self.agent_name,
                    "tool_start",
                    f"Step {iteration + 1}: Running tools...",
                    {"iteration": iteration, "tools": [tu.name for tu in tool_uses]},
                )

                tool_results = await self._execute_tools(tool_uses)

                emit_progress(
                    self.agent_name,
                    "tool_end",
                    f"Step {iteration + 1}: Tools complete.",
                    {"iteration": iteration, "tools": [tu.name for tu in tool_uses]},
                )

                tool_message = Message(role=MessageRole.USER, content=tool_results)
                turn_seq = await self._record_v2_turn(
                    run_id,
                    turn_seq,
                    tool_message,
                    metadata={"iteration": iteration, "event": "tool_results"},
                )

                if any(tu.name == "stop" for tu in tool_uses):
                    workflow_status = "completed_stop"
                    break

                iteration += 1

            if iteration >= self.max_calls:
                workflow_status = "completed_max_iterations"

        except Exception as exc:
            workflow_status = f"error: {str(exc)}"
            raise
        finally:
            status_msg = {
                "completed_success": "Almost done – finalizing...",
                "completed_no_tools": "Finished reasoning.",
                "completed_max_tokens": "Reached internal limit, finishing up...",
                "completed_max_iterations": "Finished maximum number of planning steps.",
                "completed_stop": "Task completed.",
                "error_malformed_function_call": "Encountered an error, finishing up...",
            }.get(workflow_status, "Wrapping up...")

            emit_progress(
                self.agent_name,
                "complete",
                status_msg,
                {"workflow_status": workflow_status},
            )

            if self.logger:
                final_output = self._extract_final_output()
                if not final_output and last_agent_message:
                    content = last_agent_message.content
                    if isinstance(content, list):
                        parts = []
                        for block in content:
                            if hasattr(block, "text"):
                                parts.append(block.text)
                            else:
                                parts.append(str(block))
                        final_output = " ".join(parts)
                    else:
                        final_output = str(content)

                self.logger.end_trace(
                    output=str(final_output),
                    metadata={
                        "status": workflow_status,
                        "total_iterations": iteration,
                        "thread_id": self.thread_id,
                        "resource_id": self.resource_id,
                        "memory_mode": "v2",
                    },
                )

            while not self._context_queue.empty():
                self._context_queue.get_nowait()

        if last_agent_message is None:
            return Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="")])
        return last_agent_message

    def run(self, user_prompt: str) -> Message:
        """Sync wrapper for run_async."""
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
        except Exception as exc:
            raise RuntimeError(f"Agent execution failed: {exc}") from exc
