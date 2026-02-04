"""
Conversation memory management for CAL agents.
"""
import json
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from uuid import uuid4

from .message import Message, MessageRole
from .content_blocks import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .categorization import CategorizedMessages, MessageCategorizer
from .compression import (
    CompressionConfig,
    CompressionArchiver,
)

if TYPE_CHECKING:
    from .llm import LLM
    from .logger import Logger


class Memory(ABC):
    """
    Abstract base class for conversation memory implementations.
    """

    @abstractmethod
    def add_message(self, message: Message):
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_history(self) -> List[Message]:
        """Return the current conversation history."""
        pass

    @abstractmethod
    def compress(self):
        """Compress memory to reduce size."""
        pass

    @abstractmethod
    def clone(self) -> "Memory":
        """Create a copy of this memory with the same history."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory into a dictionary."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Serialize the memory into a JSON string."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: Dict[str, Any], **kwargs) -> "Memory":
        """Construct memory from a serialized dictionary payload.
        
        Args:
            payload: Serialized memory data
            **kwargs: Implementation-specific arguments (e.g., summarizer_llm, logger)
        """
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Optional[str], **kwargs) -> "Memory":
        """Construct memory from a JSON string.

        Args:
            data: JSON string of serialized memory
            **kwargs: Implementation-specific arguments (e.g., summarizer_llm, logger)
        """
        pass

    def sync_token_count_from_llm_usage(self, prompt_tokens: int) -> None:
        """Sync internal token count with actual LLM usage.

        This method allows the memory to use actual token counts from the LLM
        instead of estimates. Called by the Agent after each LLM response.

        The prompt_tokens represents the total tokens sent to the LLM, which includes:
        - System prompt tokens
        - Tool schema tokens
        - All conversation history tokens

        This gives an accurate picture of context size for compression decisions.

        Args:
            prompt_tokens: The actual input token count from the LLM response usage data.

        Note: Default implementation is a no-op. Subclasses that track token counts
        (like FullCompressionMemory) should override this method.
        """
        pass  # Default no-op for Memory implementations that don't track tokens


class FullCompressionMemory(Memory):
    """
    Memory for long-running agentic tasks. Keeps the initial user prompt,
    summarizes middle turns using LLM-based compression, and keeps recent messages.
    
    Requires a summarizer_llm for intelligent compression. For non-LLM compression,
    use a different Memory implementation.
    
    When file-based archival is enabled (archiver provided), tool calls and
    context are stored in files with a history.md index for agent discovery.
    """

    def __init__(
        self,
        summarizer_llm: "LLM",
        max_tokens: int = 50000,
        messages: Optional[List[Message]] = None,
        compression_config: Optional[CompressionConfig] = None,
        logger: Optional["Logger"] = None,
        agent_name: Optional[str] = None,
        archiver: Optional[CompressionArchiver] = None,
    ):
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._total_tokens: int = 0  # Track running token count
        self._system_overhead: int = 0  # System prompt + tool schema tokens (calculated by sync)
        self.summarizer_llm = summarizer_llm
        self.compression_config = compression_config or CompressionConfig()
        self.logger = logger
        self.agent_name = agent_name
        
        if archiver:
            self.archiver = archiver
        else:
            if not self.agent_name:
                self.agent_name = f"agent_{uuid4().hex[:8]}"
            self.archiver = CompressionArchiver(agent_name=self.agent_name)
        
        if messages:
            for message in messages:
                self._messages.append(message)
                self._total_tokens += self._estimate_message_tokens(message)

    def add_message(self, message: Message):
        """Add a message to memory. Compress when token capacity exceeded."""
        msg_tokens = self._estimate_message_tokens(message)
        self._messages.append(message)
        self._total_tokens += msg_tokens

        if self._total_tokens > self.max_tokens:
            self.compress()

    def sync_token_count_from_llm_usage(self, prompt_tokens: int) -> None:
        """Sync internal token count with actual LLM usage.

        Updates _total_tokens to match the actual prompt tokens from the LLM,
        which includes system prompt, tool schemas, and conversation history.
        This ensures compression triggers based on real context size, not estimates.

        Also calculates and stores system overhead (system prompt + tool schema tokens)
        so that compression can account for it when deciding how much to keep.

        Args:
            prompt_tokens: The actual input token count from the LLM response.
        """
        old_total = self._total_tokens
        self._total_tokens = prompt_tokens

        # Calculate system overhead = actual tokens - estimated message tokens
        # This represents the fixed cost of system prompt + tool schemas
        estimated_message_tokens = sum(self._estimate_message_tokens(m) for m in self._messages)
        self._system_overhead = max(0, prompt_tokens - estimated_message_tokens)

        if old_total != prompt_tokens:
            print(
                f"[Memory] Synced token count: estimated {old_total} -> actual {prompt_tokens} "
                f"(diff: {prompt_tokens - old_total:+d}, system_overhead: {self._system_overhead})",
                file=sys.stderr
            )

    def _estimate_message_tokens(self, message: Message) -> int:
        """
        Get or estimate the token count for a message.

        Used for:
        - Adding completion_tokens after sync_token_count_from_llm_usage()
        - Fallback estimation when no LLM usage data is available
        - Compression breakpoint calculation in compress()

        For assistant messages with usage data, returns completion_tokens only
        (since prompt_tokens are already accounted for via sync).
        For other messages, estimates from content (~4 chars/token).
        """
        # For assistant messages, use completion_tokens (the new output tokens only)
        if message.role == MessageRole.ASSISTANT and message.usage:
            if 'completion_tokens' in message.usage:
                return message.usage['completion_tokens']
        
        token_count = 0
        
        def _estimate_tokens(text: str) -> int:
            """Estimate tokens in text using ~4 characters per token."""
            return len(text) // 4
        
        # Estimate tokens from content
        if isinstance(message.content, str):
            token_count = _estimate_tokens(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextBlock):
                    token_count += _estimate_tokens(block.text)
                elif isinstance(block, ToolUseBlock):
                    # Estimate tokens for tool use: name + input serialization
                    tool_text = f"{block.name}({str(block.input)})"
                    if block.thought:
                        tool_text += block.thought
                    token_count += _estimate_tokens(tool_text)
                elif isinstance(block, ToolResultBlock):
                    # Estimate tokens for tool result
                    if isinstance(block.content, str):
                        token_count += _estimate_tokens(block.content)
                    elif isinstance(block.content, list):
                        for sub_block in block.content:
                            if isinstance(sub_block, TextBlock):
                                token_count += _estimate_tokens(sub_block.text)
                            else:
                                token_count += _estimate_tokens(str(sub_block))
                    else:
                        token_count += _estimate_tokens(str(block.content))
                elif isinstance(block, ImageBlock):
                    # Images consume significant tokens, estimate conservatively
                    token_count += 100  # Rough estimate for image tokens
                else:
                    # Fallback for unknown block types
                    token_count += _estimate_tokens(str(block))
        
        # Ensure at least 1 token
        return max(1, token_count)

    def compress(self):
        """
        Compress memory: keep initial user message, summarize middle,
        keep most recent messages up to token limit (keep_recent_tokens).

        Uses LLM-based summarization to generate a summary, then archives
        full context to files for agent discovery via history.md index.
        """
        if len(self._messages) <= 3:
            return

        # TODO: DELETE THIS DEBUG BLOCK - compression debugging
        # ============ DEBUG START: compression state ============
        _debug_state = {
            "_total_tokens": self._total_tokens,
            "_system_overhead": self._system_overhead,
            "max_tokens": self.max_tokens,
            "config_keep_recent_tokens": self.compression_config.keep_recent_tokens,
            "num_messages": len(self._messages),
        }
        print(f"[DEBUG compress] state: {_debug_state}", file=sys.stderr)
        if self.logger:
            self.logger.log_metadata({"debug_compress_state": _debug_state})
        # ============ DEBUG END: compression state ============

        # Calculate effective token budget for messages after accounting for system overhead
        # Leave a buffer for: initial message (~500), summary (~1000), new responses (~2000)
        response_buffer = 3500
        available_for_messages = self.max_tokens - self._system_overhead - response_buffer

        # Use the smaller of configured keep_recent_tokens or available budget
        # This ensures we don't keep more than we have room for
        keep_recent_tokens = min(
            self.compression_config.keep_recent_tokens,
            max(1000, available_for_messages)  # Keep at least 1000 tokens
        )

        # TODO: DELETE THIS DEBUG BLOCK - compression debugging
        # ============ DEBUG START: budget calculations ============
        _debug_budget = {
            "available_for_messages": available_for_messages,
            "effective_keep_recent_tokens": keep_recent_tokens,
        }
        print(f"[DEBUG compress] budget: {_debug_budget}", file=sys.stderr)
        if self.logger:
            self.logger.log_metadata({"debug_compress_budget": _debug_budget})
        # ============ DEBUG END: budget calculations ============

        if keep_recent_tokens < self.compression_config.keep_recent_tokens:
            print(
                f"[Memory] Adjusted keep_recent_tokens: {self.compression_config.keep_recent_tokens} -> {keep_recent_tokens} "
                f"(max_tokens={self.max_tokens}, system_overhead={self._system_overhead})",
                file=sys.stderr
            )
        
        # Find breakpoint by iterating backwards, accumulating tokens
        initial = self._messages[0]
        token_count = 0
        breakpoint_idx = 1  # Default: include all messages after initial
        
        for i in range(len(self._messages) - 1, 0, -1):  # Skip index 0 (initial message)
            msg = self._messages[i]
            msg_tokens = self._estimate_message_tokens(msg)
            
            if token_count + msg_tokens > keep_recent_tokens:
                breakpoint_idx = i + 1  # Start from the next message (which fits)
                break
            
            token_count += msg_tokens
        
        recent_messages = self._messages[breakpoint_idx:]
        
        # Ensure recent_messages doesn't start with a tool_result, as this breaks
        # Anthropic's requirement that tool_results must follow tool_use.
        def _has_tool_result(msg: Message) -> bool:
            return isinstance(msg.content, list) and any(isinstance(b, ToolResultBlock) for b in msg.content)
        
        while breakpoint_idx > 1 and breakpoint_idx < len(self._messages) and _has_tool_result(self._messages[breakpoint_idx]):
            breakpoint_idx -= 1
            recent_messages = self._messages[breakpoint_idx:]
        
        # Ensure we keep at least 1 recent message if possible
        if not recent_messages and len(self._messages) > 1:
            print(f"[Memory] Compress fallback: keeping last message (exceeded keep_recent_tokens={keep_recent_tokens})", file=sys.stderr)
            if self.logger:
                self.logger.log_metadata({"memory_compress_fallback": "single_message", "keep_recent_tokens": keep_recent_tokens})
            recent_messages = [self._messages[-1]]
        
        if len(recent_messages) >= len(self._messages) - 1:
            # All messages except initial would be kept, nothing to compress
            print(f"[Memory] Skipping compression: keep_recent_tokens ({keep_recent_tokens}) >= max_tokens ({self.max_tokens}) or all messages fit", file=sys.stderr)
            if self.logger:
                self.logger.log_metadata({"memory_compress_skipped": "all_messages_fit", "keep_recent_tokens": keep_recent_tokens, "max_tokens": self.max_tokens})
            return
        
        to_compress = self._messages[1:len(self._messages) - len(recent_messages)]
        recent = recent_messages

        # TODO: DELETE THIS DEBUG BLOCK - compression debugging
        # ============ DEBUG START: compression breakdown ============
        _debug_recent_tokens = sum(self._estimate_message_tokens(m) for m in recent_messages)
        _debug_compress_tokens = sum(self._estimate_message_tokens(m) for m in to_compress)
        _debug_breakdown = {
            "breakpoint_idx": breakpoint_idx,
            "recent_messages_count": len(recent_messages),
            "to_compress_count": len(to_compress),
            "recent_tokens": _debug_recent_tokens,
            "compress_tokens": _debug_compress_tokens,
        }
        print(f"[DEBUG compress] breakdown: {_debug_breakdown}", file=sys.stderr)
        if self.logger:
            self.logger.log_metadata({"debug_compress_breakdown": _debug_breakdown})
        # ============ DEBUG END: compression breakdown ============

        if not to_compress:
            return
        
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        print(f"[Memory] Starting compression: {len(to_compress)} messages", file=sys.stderr)
        
        # Step 1: Categorize messages for intelligent compression
        categorized = MessageCategorizer.categorize(to_compress)
        tools_used, key_files = MessageCategorizer.extract_tools_and_files(categorized)
        
        # Step 2: Call the summarizer LLM with messages to compress
        # The LLM is expected to return JSON with: filename, summary, detailed_summary
        summarization_prompt = """You are a conversation summarizer. Analyze the conversation history and produce a JSON summary.

Your task is to:
1. Identify the main topic or task being discussed
2. Capture key decisions, actions taken, and important context
3. Note any files or resources that were referenced

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "filename": "short_descriptive_name",
  "summary": "A brief 1-2 sentence summary of what happened",
  "detailed_summary": "A comprehensive summary covering key actions, decisions, tool usage, and outcomes. Include important context that would be needed to continue the conversation."
}

Rules for the fields:
- filename: lowercase, underscores, no spaces, max 50 chars (e.g., "setup_database_schema", "debug_auth_flow")
- summary: Brief overview, 1-2 sentences
- detailed_summary: Thorough recap including tool calls made, files modified, decisions reached, and any pending items"""

        # Convert messages to text-only format to avoid Gemini sequence issues
        text_history = self._messages_to_text_for_summarization(to_compress)
        summary_request = Message(
            role=MessageRole.USER,
            content=f"Summarize the following conversation:\n\n{text_history}"
        )

        response = self.summarizer_llm.generate_content(
            system_prompt=summarization_prompt,
            conversation_history=[summary_request],
            tools=None
        )
        
        # Extract response text
        if isinstance(response.content, str):
            response_text = response.content
        else:
            text_parts = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            response_text = "\n".join(text_parts)
        
        # Step 3: Parse the JSON response
        # parsed = self._parse_llm_json_response(response_text)
        # filename = parsed.get("filename", "context")
        # summary = parsed.get("summary", "Conversation context")
        # detailed_summary = parsed.get("detailed_summary", summary)
        filename = "context"
        summary = response_text
        detailed_summary = response_text
        
        # Step 4: Format and archive the content
        archive_content = self._format_archive_content(categorized, detailed_summary)
        message_range = f"{len(to_compress)} messages"
        
        archive_path = self.archiver.write_context_file(
            filename=filename,
            content=archive_content,
            message_range=message_range,
            tools_used=tools_used,
            key_files=key_files,
            summary=summary,
        )
        
        summary_text = f"[Previous context archived to: {archive_path}]\n\n{summary}"
        archive_file = self.archiver._entries[-1].filename
        
        elapsed_time = time.time() - start_time
        end_time_ns = time.time_ns()
        
        summary_message = Message(
            role=MessageRole.USER,
            content=summary_text,
            metadata={
                "compressed": True,
                "original_count": len(to_compress),
                "archive_file": archive_file,
            }
        )
        
        self._messages = [initial, summary_message] + recent

        # Recalculate total token count after compression
        # Include system overhead so the next sync doesn't cause immediate re-compression
        estimated_message_tokens = sum(self._estimate_message_tokens(m) for m in self._messages)
        self._total_tokens = self._system_overhead + estimated_message_tokens

        # Log compression complete
        print(f"[Memory] Compression complete: {len(to_compress)} messages -> 1 summary in {elapsed_time:.2f}s "
              f"(message_tokens: {estimated_message_tokens}, system_overhead: {self._system_overhead}, total: {self._total_tokens})",
              file=sys.stderr)

        # TODO: DELETE THIS DEBUG BLOCK - compression debugging
        # ============ DEBUG START: final state ============
        _debug_final = {
            "final_num_messages": len(self._messages),
            "final_total_tokens": self._total_tokens,
            "final_message_tokens": estimated_message_tokens,
            "final_system_overhead": self._system_overhead,
        }
        print(f"[DEBUG compress] final: {_debug_final}", file=sys.stderr)
        if self.logger:
            self.logger.log_metadata({"debug_compress_final": _debug_final})
        # ============ DEBUG END: final state ============
        
        # Log as tool call to tracing system
        if self.logger:
            tool_id = str(uuid4())
            tool_use = ToolUseBlock(
                id=tool_id,
                name="memory_compression",
                input={
                    "messages_to_compress": len(to_compress),
                    "agent_name": self.agent_name or "unknown",
                }
            )
            
            result_content = f"Compressed {len(to_compress)} messages into 1 summary ({len(summary_text)} chars) in {elapsed_time:.2f}s"
            result_content += f"\nArchived to: {archive_file}"
            
            tool_result = ToolResultBlock(
                tool_use_id=tool_id,
                content=result_content,
                is_error=False,
                name="memory_compression",
                metadata={
                    "messages_compressed": len(to_compress),
                    "compression_duration_seconds": round(elapsed_time, 3),
                    "summary_length": len(summary_text),
                    "archive_file": archive_file,
                    "archive_dir": str(self.archiver.session_dir),
                }
            )
            
            self.logger.log_tool_response(
                tool_use=tool_use,
                result=tool_result,
                start_time=start_time_ns,
                end_time=end_time_ns,
            )

    def _format_archive_content(
        self,
        categorized: CategorizedMessages,
        detailed_summary: str,
    ) -> str:
        """Format the full content to be written to an archive file."""
        lines = [
            "# Archived Conversation Context",
            "",
            "## Summary",
            detailed_summary,
            "",
        ]
        
        # Add tool calls section
        if categorized.tool_calls:
            lines.extend(["## Tool Calls", ""])
            for entry in categorized.tool_calls:
                tool_use = entry.get("tool_use")
                tool_result = entry.get("tool_result")
                tool_name = entry.get("tool_name", "unknown")
                
                lines.append(f"### {tool_name}")
                if tool_use and hasattr(tool_use, "input"):
                    # Store full input (no truncation - archive files contain complete context)
                    input_str = json.dumps(tool_use.input, default=str, indent=2)
                    lines.append(f"**Input:**\n```json\n{input_str}\n```")
                
                if tool_result:
                    result_content = tool_result.content
                    if isinstance(result_content, str):
                        result_str = result_content
                    elif isinstance(result_content, list):
                        parts = []
                        for child in result_content:
                            if isinstance(child, TextBlock):
                                parts.append(child.text)
                            else:
                                parts.append(str(child))
                        result_str = "\n".join(parts)
                    else:
                        result_str = str(result_content)
                    
                    # Store full result (no truncation - archive files contain complete context)
                    error_marker = " (ERROR)" if tool_result.is_error else ""
                    lines.append(f"**Result{error_marker}:**\n```\n{result_str}\n```")
                lines.append("")
        
        if categorized.file_reads:
            lines.extend(["## File Reads", ""])
            for entry in categorized.file_reads:
                tool_use = entry.get("tool_use")
                tool_result = entry.get("tool_result")
                
                file_path = "unknown"
                if tool_use and hasattr(tool_use, "input") and "file_path" in tool_use.input:
                    file_path = tool_use.input["file_path"]
                
                lines.append(f"### {file_path}")
                if tool_result:
                    result_content = tool_result.content
                    if isinstance(result_content, str):
                        result_str = result_content
                    elif isinstance(result_content, list):
                        parts = []
                        for child in result_content:
                            if isinstance(child, TextBlock):
                                parts.append(child.text)
                        result_str = "\n".join(parts)
                    else:
                        result_str = str(result_content)
                    
                    # Store full file content (no truncation - archive files contain complete context)
                    lines.append(f"```\n{result_str}\n```")
                lines.append("")
        
        return "\n".join(lines)

    def _messages_to_text_for_summarization(self, messages: List[Message]) -> str:
        """
        Convert messages to a text representation for summarization.
        
        Strips tool call structure and creates a readable text format
        that the summarizer LLM can process without Gemini's strict
        function call sequence requirements.
        
        Uses clear delimiters to preserve structure while avoiding
        function_call/function_response parts that trigger sequence validation.
        """
        lines = []
        for msg in messages:
            role = msg.role.value.upper()
            
            if isinstance(msg.content, str):
                lines.append(f"[{role}]: {msg.content}")
                continue
            
            parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    input_str = json.dumps(block.input, indent=2, default=str)
                    parts.append(f"\n--- Tool Call: {block.name} ---\n{input_str}\n---")
                elif isinstance(block, ToolResultBlock):
                    if isinstance(block.content, str):
                        result_text = block.content
                    elif isinstance(block.content, list):
                        text_parts = []
                        for child in block.content:
                            if isinstance(child, TextBlock):
                                text_parts.append(child.text)
                        result_text = "\n".join(text_parts)
                    else:
                        result_text = str(block.content)
                    error_marker = " [ERROR]" if block.is_error else ""
                    tool_name = block.name or "unknown"
                    parts.append(f"\n--- Tool Result ({tool_name}){error_marker} ---\n{result_text}\n---")
            
            if parts:
                lines.append(f"[{role}]: {''.join(parts)}")
        
        return "\n\n".join(lines)

    def get_history(self) -> List[Message]:
        """
        Return the current conversation history.
        
        If archived context exists, injects a reference to the history.md
        file so the agent knows about available context files.
        """
        history = list(self._messages)
        
        # If we have archived context, inject reference into the first message
        if self.archiver.has_archived_context():
            history_ref = self.archiver.get_history_reference()
            
            # Find the first user message (after initial) that has compressed metadata
            # and inject the history reference there
            for i, msg in enumerate(history):
                if msg.metadata and msg.metadata.get("compressed"):
                    # Prepend history reference to the compressed summary
                    if isinstance(msg.content, str):
                        new_content = f"{history_ref}\n\n{msg.content}"
                    else:
                        # If content is blocks, prepend as text block
                        history_block = TextBlock(text=history_ref)
                        new_content = [history_block] + list(msg.content)
                    
                    # Create updated message with injected reference
                    history[i] = Message(
                        role=msg.role,
                        content=new_content,
                        usage=msg.usage,
                        metadata={**msg.metadata, "history_injected": True},
                    )
                    break
        
        return history

    def clone(self) -> 'FullCompressionMemory':
        """Create a deep copy of this memory with the same history.

        All Message objects are deep copied to ensure complete isolation
        between parent and subagent memory instances.

        Note: The archiver is NOT shared - each clone gets its own fresh archiver
        to prevent state pollution between parent and cloned memory instances.

        Note: _total_tokens is re-estimated from messages, not copied directly.
        The synced token count includes system prompt + tool schemas which may
        differ for the cloned memory's agent. The subagent will call
        sync_token_count_from_llm_usage() after its first LLM call to get the
        accurate count for its own configuration.
        """
        cloned_messages = [msg.clone() for msg in self._messages]

        return FullCompressionMemory(
            summarizer_llm=self.summarizer_llm,
            max_tokens=self.max_tokens,
            messages=cloned_messages,
            compression_config=self.compression_config,
            logger=self.logger,
            agent_name=self.agent_name,
            archiver=None,  # Don't share archiver - let clone create its own
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory into a dictionary.

        Note: summarizer_llm and logger are not serialized and must be
        re-injected when deserializing via from_dict/from_json.

        Note: _total_tokens is intentionally NOT serialized. The synced token
        count from LLM usage includes system prompt + tool schemas, which are
        not part of the serialized memory. When restored, token count is
        re-estimated from messages only. The Agent will call
        sync_token_count_from_llm_usage() after the first LLM call to restore
        the accurate count including the new system prompt and tools.
        """
        result = {
            "max_tokens": self.max_tokens,
            "messages": [self._message_to_dict(message) for message in self._messages],
            "compression_config": {
                "keep_recent_tokens": self.compression_config.keep_recent_tokens,
                "max_summary_tokens": self.compression_config.max_summary_tokens,
                "preserve_tool_names": self.compression_config.preserve_tool_names,
                "archive_tool_results": self.compression_config.archive_tool_results,
                "archive_file_reads": self.compression_config.archive_file_reads,
                "summary_style": self.compression_config.summary_style,
                "compression_ratio": self.compression_config.compression_ratio,
            },
            "agent_name": self.agent_name,
            "archiver": self.archiver.to_dict(),
        }
        
        return result

    def to_json(self) -> str:
        """Serialize the memory into a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=True)

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        summarizer_llm: "LLM",
        logger: Optional["Logger"] = None,
    ) -> "FullCompressionMemory":
        """Construct memory from a serialized dictionary payload.

        Args:
            payload: Serialized memory data
            summarizer_llm: LLM instance for compression (required)
            logger: Optional logger for compression events

        Note: Token count is re-estimated from messages during construction.
        The serialized data does not include _total_tokens because the synced
        count includes system prompt + tool schemas which may change. The Agent
        should call sync_token_count_from_llm_usage() after the first LLM call
        to restore accurate token tracking.
        """
        if not payload:
            return cls(summarizer_llm=summarizer_llm, logger=logger)
        
        # Handle backward compatibility: convert old max_items to max_tokens
        # Rough estimate: 50 messages * ~2000 tokens per message = ~100k tokens
        if "max_tokens" in payload:
            max_tokens = payload.get("max_tokens", 100000)
        elif "max_items" in payload:
            max_tokens = payload.get("max_items", 50) * 2000
        else:
            max_tokens = 100000
        
        messages_data = payload.get("messages", [])
        messages = [cls._message_from_dict(item) for item in messages_data]
        agent_name = payload.get("agent_name", payload.get("session_id"))
        
        # Deserialize compression config
        config_data = payload.get("compression_config", {})
        # Handle backward compatibility: convert old keep_recent (message count) to tokens
        # Rough estimate: 25 messages * ~400 tokens per message = ~10k tokens
        if "keep_recent" in config_data and "keep_recent_tokens" not in config_data:
            # Convert old message-based config to token-based (estimate 400 tokens per message)
            keep_recent_tokens = config_data.get("keep_recent", 25) * 400
        else:
            keep_recent_tokens = config_data.get("keep_recent_tokens", 15000)
        compression_config = CompressionConfig(
            keep_recent_tokens=keep_recent_tokens,
            max_summary_tokens=config_data.get("max_summary_tokens", 8000),
            preserve_tool_names=config_data.get("preserve_tool_names", True),
            archive_tool_results=config_data.get("archive_tool_results", True),
            archive_file_reads=config_data.get("archive_file_reads", True),
            summary_style=config_data.get("summary_style", "narrative"),
            compression_ratio=config_data.get("compression_ratio", 0.4),
        )
        
        # Deserialize archiver if present
        archiver = None
        archiver_data = payload.get("archiver")
        if archiver_data:
            archiver = CompressionArchiver.from_dict(archiver_data)
        
        return cls(
            max_tokens=max_tokens,
            messages=messages,
            summarizer_llm=summarizer_llm,
            compression_config=compression_config,
            logger=logger,
            agent_name=agent_name,
            archiver=archiver,
        )

    @classmethod
    def from_json(
        cls,
        data: Optional[str],
        summarizer_llm: "LLM",
        logger: Optional["Logger"] = None,
        agent_name: Optional[str] = None,
    ) -> "FullCompressionMemory":
        """Construct memory from a JSON string.

        Args:
            data: JSON string of serialized memory
            summarizer_llm: LLM instance for compression (required)
            logger: Optional logger for compression events
            agent_name: Optional agent name (used if not present in serialized data)

        Note: Token count is re-estimated from messages. See from_dict docstring
        for details on why _total_tokens is not serialized.
        """
        if not data:
            return cls(summarizer_llm=summarizer_llm, logger=logger, agent_name=agent_name)
        
        payload = json.loads(data)
        # Use provided agent_name if not in payload (backward compatibility with session_id)
        if agent_name and not payload.get("agent_name") and not payload.get("session_id"):
            payload["agent_name"] = agent_name
        elif not payload.get("agent_name") and payload.get("session_id"):
            payload["agent_name"] = payload.get("session_id")
        
        return cls.from_dict(payload, summarizer_llm=summarizer_llm, logger=logger)

    @staticmethod
    def _message_to_dict(message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": message.role.value,
            "content": FullCompressionMemory._content_to_serializable(message.content),
            "usage": message.usage,
        }
        if message.metadata:
            payload["metadata"] = message.metadata
        return payload

    @staticmethod
    def _message_from_dict(data: Dict[str, Any]) -> Message:
        role_value = data.get("role", MessageRole.USER.value)
        role = MessageRole(role_value)
        content = FullCompressionMemory._content_from_serializable(data.get("content"))
        return Message(role=role, content=content, usage=data.get("usage") or {}, metadata=data.get("metadata") or {})

    @staticmethod
    def _content_to_serializable(content: Union[str, List[ContentBlock], List[ToolResultBlock]]) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(content, str):
            return content
        return [block.to_dict() for block in content]

    @staticmethod
    def _content_from_serializable(content: Union[str, List[Dict[str, Any]], None]) -> Union[str, List[ContentBlock]]:
        if content is None:
            return []
        if isinstance(content, str):
            return content
        blocks: List[ContentBlock] = []
        for block_data in content:
            block = ContentBlock.from_dict(block_data)
            if block:
                blocks.append(block)
        return blocks
