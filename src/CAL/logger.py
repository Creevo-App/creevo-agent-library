"""
Abstract Logger interface for Maxim AI and LangSmith
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .message import Message
from .content_blocks import ThinkingBlock, ToolUseBlock, ToolResultBlock

import os
import sys
import time
import json
from datetime import datetime, timezone
from uuid import uuid4

class Logger(ABC):
    """Abstract base class for logging implementations"""

    @abstractmethod
    def log_llm_response(
        self,
        message: Message,
        iteration: int,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> None:
        """
        Log LLM response to the tracing system.

        Args:
            message: The message containing the LLM response
            iteration: The iteration number of this response
            model: Optional model name
            provider: Optional provider name
            start_time: Start time in nanoseconds
            end_time: End time in nanoseconds
        """
        pass

    @abstractmethod
    def log_tool_response(
        self,
        tool_use: ToolUseBlock,
        result: ToolResultBlock,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> None:
        """
        Log tool call and result to the tracing system.

        Args:
            tool_use: The tool use block containing tool call information
            result: The tool result block containing the result
            start_time: Start time in nanoseconds
            end_time: End time in nanoseconds
        """
        pass

    @abstractmethod
    def start_trace(self, name: str, user_prompt: str) -> Optional[str]:
        """
        Start a new trace.

        Args:
            name: Name of the trace
            user_prompt: The user's initial prompt

        Returns:
            Trace ID if available, None otherwise
        """
        pass

    @abstractmethod
    def end_trace(self, output: str, metadata: Dict[str, Any]) -> None:
        """
        End the current trace.

        Args:
            output: The final output of the trace
            metadata: Additional metadata to attach to the trace
        """
        pass

    @abstractmethod
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log metadata to the current trace.

        Args:
            metadata: Dictionary of metadata to log
        """
        pass

    @abstractmethod
    def flush(self, timeout_millis: int = 5000) -> None:
        """
        Flush pending traces to the tracing system.

        Args:
            timeout_millis: Maximum time to wait for flush (milliseconds)
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the logger and flush any pending traces"""
        pass

    @abstractmethod
    def create_child_logger(self, name: str, agent_name: Optional[str] = None) -> "Logger":
        """
        Create a child logger for sub-agent execution with nested span context.

        Args:
            name: Name identifier for the child logger (typically sub-agent name)
            agent_name: Optional agent name for the child logger. If not provided,
                       inherits from parent. Subagents should pass their own agent_name
                       so logging metadata reflects the subagent, not the parent.

        Returns:
            A new Logger instance that logs under a nested span context
        """
        pass

    @abstractmethod
    def end_child(self) -> None:
        """
        End the child logger's wrapper span. Call this when the sub-agent finishes.
        Only applicable for child loggers created via create_child_logger().
        """
        pass


def _ns_to_datetime(ns: Optional[int]) -> Optional[datetime]:
    """Convert nanosecond timestamp to timezone-aware datetime."""
    if ns is None:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _format_content(message: Message) -> tuple:
    """Extract thinking and content parts from a message.

    Returns:
        (content_text, thinking_text) tuple where thinking_text is None
        if no thinking blocks are present.
    """
    thinking_parts = []
    content_parts = []
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                thinking_parts.append(block.text)
            elif hasattr(block, 'text'):
                content_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                content_parts.append(f"[Tool Use: {block.name}({json.dumps(block.input)})]")
    else:
        content_parts.append(str(message.content))
    return " ".join(content_parts), "\n".join(thinking_parts) if thinking_parts else None


class LangSmithLogger(Logger):
    """
    Implementation of Logger using the LangSmith Python SDK (RunTree API).

    Requires the ``langsmith`` package: ``pip install langsmith``

    Environment variables:
        LANGSMITH_API_KEY: Required. Your LangSmith API key.
        LANGSMITH_ENDPOINT: Optional. Defaults to https://api.smith.langchain.com
    """

    def __init__(
        self,
        project_name: str = "default",
        agent_name: Optional[str] = None,
        _parent_run=None,
    ):
        try:
            from langsmith.run_trees import RunTree
        except ImportError:
            raise ImportError(
                "LangSmithLogger requires the 'langsmith' package. "
                "Install it with: pip install langsmith"
            )

        self._RunTree = RunTree
        self._is_child = _parent_run is not None
        self._parent_run = _parent_run
        self.project_name = project_name
        self.agent_name: str = agent_name or "unknown"
        self.system_prompt: str = ""
        self.user_prompt: str = ""

        if not self._is_child:
            api_key = os.getenv("LANGSMITH_API_KEY")
            if not api_key:
                raise EnvironmentError("LANGSMITH_API_KEY not set in environment")

        self.root_run = None  # RunTree for the agent trace

    def start_trace(self, name: str, user_prompt: str) -> Optional[str]:
        if self._is_child:
            self.user_prompt = user_prompt
            return None

        if self.root_run:
            print(
                "Warning: start_trace called with existing active run. "
                "Ending previous run before starting new one.",
                file=sys.stderr,
            )
            try:
                self.root_run.end()
                self.root_run.patch()
            except Exception as e:
                print(f"Warning: Failed to end previous LangSmith run: {e}", file=sys.stderr)

        self.user_prompt = user_prompt

        inputs = {"prompt": user_prompt}
        if self.system_prompt:
            inputs["system_prompt"] = self.system_prompt

        self.root_run = self._RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
            project_name=self.project_name,
            extra={
                "metadata": {
                    "agent_name": self.agent_name,
                    "service": "cal-agent",
                }
            },
        )
        self.root_run.post()

        return str(self.root_run.id)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        if "agent_name" in metadata:
            self.agent_name = metadata["agent_name"]
        elif "session_id" in metadata:
            self.agent_name = metadata["session_id"]
        if "system_prompt" in metadata:
            self.system_prompt = metadata["system_prompt"]

        run = self._parent_run if self._is_child else self.root_run
        if run:
            if run.extra is None:
                run.extra = {}
            run.extra.setdefault("metadata", {}).update(metadata)
            try:
                run.patch()
            except Exception as e:
                print(f"Warning: Failed to persist metadata to LangSmith: {e}", file=sys.stderr)

    def log_llm_response(
        self,
        message: Message,
        iteration: int,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> None:
        parent = self._parent_run if self._is_child else self.root_run
        if not parent:
            return

        try:
            content_text, thinking_text = _format_content(message)

            usage = message.usage or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            inputs = {
                "messages": [
                    {"role": "system", "content": self.system_prompt or ""},
                    {"role": "user", "content": self.user_prompt or ""},
                ],
            }

            outputs = {
                "choices": [
                    {
                        "message": {
                            "role": message.role.value,
                            "content": content_text,
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            extra_metadata = {
                "iteration": iteration,
                "agent_name": self.agent_name,
                "provider": provider or "unknown",
            }
            if thinking_text:
                extra_metadata["thinking"] = thinking_text

            start_dt = _ns_to_datetime(start_time)

            llm_run = parent.create_child(
                name=f"llm.generate_content (iter {iteration})",
                run_type="llm",
                inputs=inputs,
                start_time=start_dt,
                extra={
                    "metadata": extra_metadata,
                    "invocation_params": {
                        "model": model or "unknown",
                        "provider": provider or "unknown",
                    },
                },
            )
            llm_run.post()

            end_dt = _ns_to_datetime(end_time) or datetime.now(tz=timezone.utc)
            llm_run.end(outputs=outputs, end_time=end_dt)
            llm_run.patch()

        except Exception as e:
            print(f"Warning: Failed to log LLM response to LangSmith: {e}", file=sys.stderr)

    def log_tool_response(
        self,
        tool_use: ToolUseBlock,
        result: ToolResultBlock,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> None:
        parent = self._parent_run if self._is_child else self.root_run
        if not parent:
            return

        try:
            input_data = tool_use.input if isinstance(tool_use.input, dict) else {"raw": str(tool_use.input)}
            output_content = result.content if isinstance(result.content, str) else str(result.content)

            start_dt = _ns_to_datetime(start_time)

            tool_run = parent.create_child(
                name=f"tool.{tool_use.name}",
                run_type="tool",
                inputs=input_data,
                start_time=start_dt,
                extra={
                    "metadata": {
                        "agent_name": self.agent_name,
                        "tool_use_id": tool_use.id,
                    }
                },
            )
            tool_run.post()

            outputs = {"output": output_content}
            if result.metadata:
                outputs["metadata"] = result.metadata

            end_dt = _ns_to_datetime(end_time) or datetime.now(tz=timezone.utc)

            if result.is_error:
                tool_run.end(
                    outputs=outputs,
                    end_time=end_dt,
                    error=output_content,
                )
            else:
                tool_run.end(outputs=outputs, end_time=end_dt)
            tool_run.patch()

        except Exception as e:
            print(f"Warning: Failed to log tool response to LangSmith: {e}", file=sys.stderr)

    def end_trace(self, output: str, metadata: Dict[str, Any]) -> None:
        if self._is_child:
            return

        if not self.root_run:
            print(
                "Warning: end_trace called with no active run. Was start_trace called?",
                file=sys.stderr,
            )
            return

        try:
            outputs = {"output": output, "metadata": metadata}
            self.root_run.end(outputs=outputs)
            self.root_run.patch()
        except Exception as e:
            print(f"Warning: Error ending LangSmith trace: {e}", file=sys.stderr)
        finally:
            self.root_run = None

    def flush(self, timeout_millis: int = 5000) -> None:
        if self._is_child:
            return
        # RunTree uses synchronous HTTP calls (post/patch) so there is
        # nothing to flush. This is kept for interface compatibility.

    def shutdown(self) -> None:
        if self._is_child:
            self.end_child()
            return

        if self.root_run:
            try:
                self.root_run.end(outputs={"output": "shutdown"})
                self.root_run.patch()
            except Exception as e:
                print(f"Warning: Error ending run during shutdown: {e}", file=sys.stderr)
            finally:
                self.root_run = None

    def create_child_logger(self, name: str, agent_name: Optional[str] = None) -> "LangSmithLogger":
        """Create a child logger for sub-agent execution with nested run context."""
        parent = self._parent_run if self._is_child else self.root_run
        if not parent:
            raise RuntimeError("Cannot create child logger: no active trace or parent run")

        child_agent_name = agent_name if agent_name is not None else self.agent_name

        wrapper_run = parent.create_child(
            name=f"subagent_{name}",
            run_type="chain",
            inputs={"task": "subagent delegation"},
            extra={
                "metadata": {
                    "type": "subagent",
                    "subagent_name": name,
                    "agent_name": child_agent_name,
                }
            },
        )
        wrapper_run.post()

        child = LangSmithLogger(
            project_name=self.project_name,
            agent_name=child_agent_name,
            _parent_run=wrapper_run,
        )
        child.system_prompt = self.system_prompt
        child.user_prompt = self.user_prompt
        return child

    def end_child(self) -> None:
        """End the wrapper run for this child logger."""
        if not self._is_child:
            raise RuntimeError("Cannot end child logger: not a child logger")

        if self._parent_run is None:
            return

        try:
            self._parent_run.end(outputs={"output": "subagent complete"})
            self._parent_run.patch()
        except Exception as e:
            print(f"Warning: Error ending child LangSmith run: {e}", file=sys.stderr)
        finally:
            self._parent_run = None


class MaximLogger(Logger):
    """
    Implementation of Logger for Maxim AI using the updated SDK v0.1.3+.
    """

    def __init__(self, agent_name: Optional[str] = None, _parent_span=None):
        self._parent_span = _parent_span
        self._is_child = _parent_span is not None

        if self._is_child:
            self.maxim_client = None
            self.logger_instance = None
            self.root_trace = None
            self.maxim_session = None
        else:
            self.maxim_client = None  # Keep reference to client for proper cleanup
            self.logger_instance = self._setup_maxim()
            self.root_trace = None
            self.maxim_session = None  # Maxim session object for proper session linking

        self.agent_name: str = agent_name or "unknown"
        self.system_prompt: str = ""
        self.user_prompt: str = ""
        self._is_shutdown = False  # Track shutdown state to avoid duplicate cleanup

    def _setup_maxim(self):
        """Initialize Maxim client. Returns None if initialization fails."""
        maxim_api_key = os.getenv("MAXIM_API_KEY")
        maxim_log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")

        if not maxim_api_key:
            print("MaximLogger disabled: MAXIM_API_KEY not set in environment.", file=sys.stderr)
            return None
        if not maxim_log_repo_id:
            print("MaximLogger disabled: MAXIM_LOG_REPO_ID not set in environment.", file=sys.stderr)
            return None

        try:
            from maxim import Maxim
            from maxim.logger import LoggerConfigDict

            self.maxim_client = Maxim({"api_key": maxim_api_key})
            return self.maxim_client.logger(LoggerConfigDict(id=maxim_log_repo_id))

        except ImportError:
            print("MaximLogger disabled: 'maxim-py' package not found. Please `pip install maxim-py`.", file=sys.stderr)
            return None
        except Exception as e:
            error_msg = str(e)
            if "Invalid log repository" in error_msg:
                print("MaximLogger disabled: MAXIM_LOG_REPO_ID appears invalid. Check the Maxim dashboard for the correct log repository ID. Note this is different from the repository name.", file=sys.stderr)
            else:
                print(f"MaximLogger disabled: Failed to initialize - {e}", file=sys.stderr)
            return None

    def create_child_logger(self, name: str, agent_name: Optional[str] = None) -> "MaximLogger":
        """Create a child logger that logs under a nested sub-agent span.

        Args:
            name: Name identifier for the child span (typically sub-agent tool name)
            agent_name: Agent name for the child logger. If not provided, inherits
                       from parent. Subagents should pass their own agent_name so
                       logging metadata reflects the subagent, not the parent.
        """
        span_parent = self._parent_span if self._is_child else self.root_trace

        if not span_parent:
            raise RuntimeError("Cannot create child logger: no active trace or parent span")

        # Use provided agent_name or fall back to parent's
        child_agent_name = agent_name if agent_name is not None else self.agent_name

        # Create wrapper span for sub-agent execution
        wrapper_span = span_parent.span({
            "id": str(uuid4()),
            "name": f"subagent_{name}",
            "tags": {
                "type": "subagent",
                "subagent_name": name,
                "agent_name": child_agent_name
            }
        })

        # Create child logger that will log under this wrapper span
        child = MaximLogger(agent_name=child_agent_name, _parent_span=wrapper_span)
        child.system_prompt = self.system_prompt
        child.user_prompt = self.user_prompt
        return child

    def end_child(self) -> None:
        """End the wrapper span for this child logger."""
        if not self._is_child:
            raise RuntimeError("Cannot end child logger: not a child logger, this was called improperly.")

        if self._parent_span is None:
            return

        self._parent_span.end()
        self._parent_span = None

    def start_trace(self, name: str, user_prompt: str) -> Optional[str]:
        # Child loggers don't manage their own trace
        if self._is_child:
            self.user_prompt = user_prompt
            return None

        if not self.logger_instance:
            return None

        # Check for existing trace
        if self.root_trace:
            print("Warning: start_trace called with existing active trace. Ending previous trace before starting new one.", file=sys.stderr)
            try:
                self.root_trace.end()
            except Exception as e:
                print(f"Warning: Failed to end previous Maxim trace: {e}", file=sys.stderr)

        try:
            # Store user prompt for logging
            self.user_prompt = user_prompt

            # Create or get Maxim session for proper session linking in UI
            if not self.maxim_session:
                self.maxim_session = self.logger_instance.session({
                    "id": str(uuid4()),
                    "name": f"Session-{self.agent_name}"
                })

            trace_id = str(uuid4())
            # Create trace within the session context (not directly on logger)
            self.root_trace = self.maxim_session.trace({
                "id": trace_id,
                "name": name,
                "tags": {
                    "service": "cal-agent"
                }
            })

            input_value = f"User Prompt: {user_prompt}"
            if self.system_prompt:
                 input_value = f"{input_value}\n\nSystem Prompt: {self.system_prompt}"

            self.root_trace.set_input(input_value)
            return trace_id

        except Exception as e:
            print(f"Warning: Failed to start Maxim trace: {e}", file=sys.stderr)
            return None

    def log_llm_response(
        self,
        message: Message,
        iteration: int,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> None:
        # Determine span parent: use _parent_span for child loggers, root_trace otherwise
        span_parent = self._parent_span if self._is_child else self.root_trace
        if not span_parent:
            return

        try:
            # Create a span for the LLM interaction
            span_id = str(uuid4())
            span = span_parent.span({
                "id": span_id,
                "name": f"llm_generate_iter_{iteration}",
                "tags": {
                    "iteration": iteration,
                    "model": model or "unknown",
                    "agent_name": self.agent_name
                }
            })

            # Prepare usage data
            usage = message.usage or {}
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            tokens = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get('total_tokens', prompt_tokens + completion_tokens)
            }

            # Setup Generation Log - user prompt first, then system prompt
            messages = [
                {"role": "user", "content": self.user_prompt or "No user prompt"},
                {"role": "system", "content": self.system_prompt or "No system prompt"},
            ]
            generation = span.generation({
                "id": str(uuid4()),
                "name": "llm_inference",
                "provider": provider or "unknown",
                "model": model or "unknown",
                "messages": messages,
                "model_parameters": {},
            })

            # Use shared helper to extract content
            content_text, thinking_text = _format_content(message)

            if thinking_text:
                span.add_tag("thinking", thinking_text)
            created_time = int(end_time / 1e9) if end_time else int(time.time())

            generation.result({
                "id": str(uuid4()),
                "object": "chat.completion",
                "created": created_time,
                "model": model or "unknown",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": message.role.value,
                        "content": content_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": tokens
            })

            span.end()

        except Exception as e:
            print(f"Warning: Failed to log LLM response to Maxim: {e}", file=sys.stderr)

    def log_tool_response(
        self,
        tool_use: ToolUseBlock,
        result: ToolResultBlock,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> None:
        span_parent = self._parent_span if self._is_child else self.root_trace
        if not span_parent:
            return

        try:
            # Create a span for the tool execution
            span_id = str(uuid4())
            span = span_parent.span({
                "id": span_id,
                "name": f"tool_execute_{tool_use.name}",
                "tags": {
                    "tool_name": tool_use.name,
                    "agent_name": self.agent_name,
                    "is_error": result.is_error
                }
            })

            input_str = json.dumps(tool_use.input) if isinstance(tool_use.input, dict) else str(tool_use.input)
            span.add_tag("input", input_str)

            output_content = result.content if isinstance(result.content, str) else str(result.content)
            span.add_tag("output", output_content)

            if result.metadata:
                for k, v in result.metadata.items():
                    span.add_tag(k, str(v))

            span.end()

        except Exception as e:
            print(f"Warning: Failed to log Tool response to Maxim: {e}", file=sys.stderr)

    def end_trace(self, output: str, metadata: Dict[str, Any]) -> None:
        # Child loggers don't manage their own trace
        if self._is_child:

            return

        if not self.root_trace:
            print("Warning: end_trace called with no active trace. Was start_trace called?", file=sys.stderr)
            return

        try:
            self.root_trace.set_output(output)
            if metadata:
                for k, v in metadata.items():
                    self.root_trace.add_tag(k, str(v))

            # End the trace - this may trigger async flushing
            self.root_trace.end()
        except Exception as e:
            print(f"Warning: Error ending Maxim trace: {e}", file=sys.stderr)
        finally:
            self.root_trace = None

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        if "agent_name" in metadata:
            self.agent_name = metadata["agent_name"]
        elif "session_id" in metadata:  # Backward compatibility
            self.agent_name = metadata["session_id"]
        if "system_prompt" in metadata:
            self.system_prompt = metadata["system_prompt"]
        if self._is_child:
            if self._parent_span:
                for k, v in metadata.items():
                    self._parent_span.add_tag(k, str(v))
            return

        if self.root_trace:
            for k, v in metadata.items():
                self.root_trace.add_tag(k, str(v))

    def flush(self, timeout_millis: int = 5000) -> None:
        # Child loggers don't manage flush
        if self._is_child:
            return

        # Skip if already shutdown
        if self._is_shutdown:
            return

        # Maxim SDK handles flushing internally usually, or via end()
        # If there's an active trace, ensure it's ended before flushing
        if self.root_trace:
            try:
                self.root_trace.end()
            except Exception as e:
                if "cannot schedule new futures after shutdown" not in str(e):
                    print(f"Warning: Error ending trace during flush: {e}", file=sys.stderr)
            finally:
                self.root_trace = None

        # Note: Don't end session on flush - session persists across multiple traces/requests

        # Use the logger's flush method if available
        if self.logger_instance:
            try:
                if hasattr(self.logger_instance, 'flush'):
                    self.logger_instance.flush()
                else:
                    # Fallback: give SDK time to process
                    time.sleep(0.1)
            except Exception as e:
                if "cannot schedule new futures after shutdown" not in str(e):
                    print(f"Warning: Error flushing Maxim logger: {e}", file=sys.stderr)

    def shutdown(self) -> None:
        # Child loggers just end their span, they don't manage client lifecycle
        if self._is_child:
            self.end_child()
            return

        # Prevent duplicate shutdown calls
        if self._is_shutdown:
            return
        self._is_shutdown = True

        # Ensure trace is ended before shutdown
        if self.root_trace:
            try:
                self.root_trace.end()
            except Exception as e:
                if "cannot schedule new futures after shutdown" not in str(e):
                    print(f"Warning: Error ending trace during shutdown: {e}", file=sys.stderr)
            finally:
                self.root_trace = None

        # End the Maxim session
        if self.maxim_session:
            try:
                self.maxim_session.end()
            except Exception as e:
                if "cannot schedule new futures after shutdown" not in str(e):
                    print(f"Warning: Error ending Maxim session: {e}", file=sys.stderr)
            finally:
                self.maxim_session = None

        # Properly cleanup the Maxim client
        if self.maxim_client:
            try:
                # Try to cleanup the client - this ensures worker threads are properly terminated
                if hasattr(self.maxim_client, 'cleanup'):
                    self.maxim_client.cleanup()
                elif hasattr(self.maxim_client, 'close'):
                    self.maxim_client.close()
                elif hasattr(self.maxim_client, 'shutdown'):
                    self.maxim_client.shutdown()
            except Exception as e:
                # Silently ignore shutdown errors - they're expected during process exit
                if "cannot schedule new futures after shutdown" not in str(e):
                    print(f"Warning: Error cleaning up Maxim client: {e}", file=sys.stderr)
            finally:
                self.maxim_client = None
                self.logger_instance = None
