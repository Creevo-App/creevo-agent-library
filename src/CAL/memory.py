"""
Conversation memory management for CAL agents.
"""
import json
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

from .message import Message, MessageRole
from .content_blocks import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .compression import (
    CompressionConfig,
    CompressionArchiver,
    CategorizedMessages,
    MessageCategorizer,
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
    def clear(self):
        """Clear all stored messages."""
        pass

    @abstractmethod
    def compress(self):
        """Compress memory to reduce size."""
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
    def from_dict(cls, payload: Dict[str, Any]) -> "Memory":
        """Construct memory from a serialized dictionary payload."""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Optional[str]) -> "Memory":
        """Construct memory from a JSON string."""
        pass


class FullCompressionMemory(Memory):
    """
    Memory for long-running agentic tasks. Keeps the initial user prompt,
    summarizes middle turns, and keeps recent messages.
    
    Supports LLM-based intelligent summarization when a summarizer_llm is provided,
    otherwise falls back to simple truncation-based compression.
    
    When file-based archival is enabled (archiver provided), tool calls and
    context are stored in files with a history.md index for agent discovery.
    """
    
    # Lazy-loaded tiktoken encoding (cl100k_base is a good general-purpose encoding)
    _encoding = None
    
    @classmethod
    def _get_encoding(cls):
        """Get or initialize tiktoken encoding."""
        if cls._encoding is None and _TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (used by GPT-4, good general-purpose)
                cls._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                cls._encoding = False  # Mark as unavailable
        return cls._encoding if cls._encoding else None

    def __init__(
        self,
        max_tokens: int = 100000,
        messages: Optional[List[Message]] = None,
        summarizer_llm: Optional["LLM"] = None,
        compression_config: Optional[CompressionConfig] = None,
        logger: Optional["Logger"] = None,
        session_id: Optional[str] = None,
        archiver: Optional[CompressionArchiver] = None,
    ):
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._total_tokens: int = 0  # Track running token count
        self.summarizer_llm = summarizer_llm
        self.compression_config = compression_config or CompressionConfig()
        self.logger = logger
        self.session_id = session_id
        
        # Initialize archiver for file-based compression
        if archiver:
            self.archiver = archiver
        elif session_id:
            self.archiver = CompressionArchiver(session_id=session_id)
        else:
            self.archiver = None
        
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

    def _estimate_message_tokens(self, message: Message) -> int:
        """
        Get the token count for a message, using actual usage data when available.
        
        Priority order:
        1. Use actual token count from message.usage if available (most accurate)
           - Assistant messages from LLM responses have this data
        2. Use tiktoken for text content if available (accurate)
        3. Fall back to character-based estimation (~4 chars per token)
        
        Note: Only assistant messages (LLM responses) have usage data.
        User messages and tool responses are estimated.
        """
        # If message has usage metadata with token count, use that (most accurate)
        # This applies to assistant messages from LLM responses
        if message.usage and 'total_tokens' in message.usage:
            return message.usage['total_tokens']
        
        token_count = 0
        
        # Try to use tiktoken if available
        encoding = self._get_encoding()
        use_tiktoken = encoding is not None
        
        def _count_tokens(text: str) -> int:
            """Count tokens in text using tiktoken or fallback to estimation."""
            if use_tiktoken:
                try:
                    return len(encoding.encode(text))
                except Exception:
                    # If encoding fails, fall back to estimation
                    return len(text) // 4
            else:
                # Fallback: ~4 characters per token
                return len(text) // 4
        
        # Estimate tokens from content
        if isinstance(message.content, str):
            token_count = _count_tokens(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextBlock):
                    token_count += _count_tokens(block.text)
                elif isinstance(block, ToolUseBlock):
                    # Estimate tokens for tool use: name + input serialization
                    tool_text = f"{block.name}({str(block.input)})"
                    if block.thought:
                        tool_text += block.thought
                    token_count += _count_tokens(tool_text)
                elif isinstance(block, ToolResultBlock):
                    # Estimate tokens for tool result
                    if isinstance(block.content, str):
                        token_count += _count_tokens(block.content)
                    elif isinstance(block.content, list):
                        for sub_block in block.content:
                            if isinstance(sub_block, TextBlock):
                                token_count += _count_tokens(sub_block.text)
                            else:
                                token_count += _count_tokens(str(sub_block))
                    else:
                        token_count += _count_tokens(str(block.content))
                elif isinstance(block, ImageBlock):
                    # Images consume significant tokens, estimate conservatively
                    token_count += 100  # Rough estimate for image tokens
                else:
                    # Fallback for unknown block types
                    token_count += _count_tokens(str(block))
        
        # Ensure at least 1 token
        return max(1, token_count)

    def compress(self):
        """
        Compress memory: keep initial user message, summarize middle,
        keep most recent messages up to token limit (keep_recent_tokens).
        
        Uses LLM-based summarization if summarizer_llm is available,
        otherwise falls back to truncation-based compression.
        """
        if len(self._messages) <= 3:
            return
        
        keep_recent_tokens = self.compression_config.keep_recent_tokens
        
        # Calculate how many recent messages to keep based on token count
        initial = self._messages[0]
        token_count = 0
        recent_messages = []
        
        # Start from the end and work backwards, accumulating tokens
        for i in range(len(self._messages) - 1, 0, -1):  # Skip index 0 (initial message)
            msg = self._messages[i]
            msg_tokens = self._estimate_message_tokens(msg)
            
            if token_count + msg_tokens > keep_recent_tokens:
                break
            
            recent_messages.insert(0, msg)
            token_count += msg_tokens
        
        # Ensure we keep at least 1 recent message if possible
        if not recent_messages and len(self._messages) > 1:
            recent_messages = [self._messages[-1]]
        
        if len(recent_messages) >= len(self._messages) - 1:
            # All messages except initial would be kept, nothing to compress
            return
        
        to_compress = self._messages[1:len(self._messages) - len(recent_messages)]
        recent = recent_messages
        
        if not to_compress:
            return
        
        # Determine compression method
        use_file_based = self.summarizer_llm and self.archiver
        if use_file_based:
            compression_method = "llm_file_based"
        elif self.summarizer_llm:
            compression_method = "llm_inline"
        else:
            compression_method = "truncation"
        
        # Log compression start
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        print(f"[Memory] Starting compression: {len(to_compress)} messages using {compression_method}", file=sys.stderr)
        
        # Log to Maxim if logger available
        if self.logger:
            self.logger.log_metadata({
                "compression_started": True,
                "compression_method": compression_method,
                "messages_to_compress": len(to_compress),
                "session_id": self.session_id or "unknown",
            })
        
        # Track archive file for logging
        archive_file = None
        
        # Generate summary using appropriate method
        try:
            if self.summarizer_llm:
                summary_text = self._compress_with_llm(to_compress)
                # Check if file was created (for file-based compression)
                if self.archiver and self.archiver.has_archived_context():
                    entries = self.archiver._entries
                    if entries:
                        archive_file = entries[-1].filename
            else:
                summary_text = self._compress_with_truncation(to_compress)
        except Exception as e:
            # If LLM compression fails, fall back to truncation
            print(f"[Memory] LLM compression failed, falling back to truncation: {e}", file=sys.stderr)
            summary_text = self._compress_with_truncation(to_compress)
            compression_method = "truncation_fallback"
        
        elapsed_time = time.time() - start_time
        end_time_ns = time.time_ns()
        
        # Create summary message
        summary_message = Message(
            role=MessageRole.USER,
            content=summary_text,
            metadata={
                "compressed": True,
                "original_count": len(to_compress),
                "compression_method": compression_method,
                "archive_file": archive_file,
            }
        )
        
        self._messages = [initial, summary_message] + recent
        
        # Recalculate total token count after compression
        self._total_tokens = sum(self._estimate_message_tokens(m) for m in self._messages)
        
        # Log compression complete
        print(f"[Memory] Compression complete: {len(to_compress)} messages -> 1 summary in {elapsed_time:.2f}s", file=sys.stderr)
        
        # Log completion to Maxim
        if self.logger:
            compression_metadata = {
                "compression_completed": True,
                "compression_method": compression_method,
                "messages_compressed": len(to_compress),
                "compression_duration_seconds": round(elapsed_time, 3),
                "summary_length": len(summary_text),
            }
            if archive_file:
                compression_metadata["archive_file"] = archive_file
                compression_metadata["archive_dir"] = str(self.archiver.session_dir)
            
            self.logger.log_metadata(compression_metadata)

    def _summarize_content(self, content: Union[str, List[ContentBlock]]) -> str:
        """Create a brief summary of message content (truncation-based fallback)."""
        if isinstance(content, str):
            return content[:150] + "..." if len(content) > 150 else content
        
        parts = []
        for block in content:
            if isinstance(block, TextBlock):
                text = block.text[:100] + "..." if len(block.text) > 100 else block.text
                parts.append(text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[Called {block.name}]")
            elif isinstance(block, ToolResultBlock):
                parts.append("[Tool result]")
            elif isinstance(block, ImageBlock):
                parts.append("[Image]")
        
        return " ".join(parts) if parts else "[empty]"

    def _format_message_for_prompt(self, msg: Message) -> str:
        """Format a single message for inclusion in the compression prompt."""
        role_label = msg.role.value.upper()
        
        if isinstance(msg.content, str):
            return f"[{role_label}]: {msg.content}"
        
        parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append(f"Text: {block.text}")
            elif isinstance(block, ToolUseBlock):
                tool_info = f"Tool Call: {block.name}"
                if self.compression_config.preserve_tool_names and block.input:
                    # Include key input parameters (truncate large values)
                    input_summary = {}
                    for k, v in block.input.items():
                        if isinstance(v, str) and len(v) > 200:
                            input_summary[k] = v[:200] + "..."
                        else:
                            input_summary[k] = v
                    tool_info += f" with inputs: {json.dumps(input_summary, default=str)}"
                parts.append(tool_info)
            elif isinstance(block, ToolResultBlock):
                result_preview = ""
                if isinstance(block.content, str):
                    result_preview = block.content[:500] + "..." if len(block.content) > 500 else block.content
                elif isinstance(block.content, list):
                    text_parts = []
                    for child in block.content:
                        if isinstance(child, TextBlock):
                            text_parts.append(child.text[:300])
                        elif isinstance(child, ImageBlock):
                            text_parts.append("[Image]")
                    result_preview = " ".join(text_parts)[:500]
                tool_name = block.name or "unknown"
                parts.append(f"Tool Result ({tool_name}): {result_preview}")
            elif isinstance(block, ImageBlock):
                parts.append("[Image content]")
        
        return f"[{role_label}]:\n" + "\n".join(parts)

    def _build_compression_prompt(self, messages_to_compress: List[Message]) -> str:
        """
        Build a structured prompt for the summarizer LLM.
        
        The prompt instructs the LLM to:
        - Preserve key decisions and outcomes from tool calls
        - Retain important file content that may be referenced later
        - Summarize conversation flow coherently
        """
        formatted_messages = []
        for i, msg in enumerate(messages_to_compress, 1):
            formatted_messages.append(f"--- Message {i} ---\n{self._format_message_for_prompt(msg)}")
        
        conversation_text = "\n\n".join(formatted_messages)
        
        prompt = f"""You are a conversation summarizer for an AI coding assistant. Your task is to compress the following conversation history while preserving all important context.

CONVERSATION TO SUMMARIZE:
{conversation_text}

INSTRUCTIONS:
1. Identify and preserve:
   - The user's goals and requests
   - Key decisions made by the assistant
   - Important file contents, code snippets, or data that may be referenced later
   - Tool calls and their outcomes (what was done and what resulted)
   - Any errors or issues encountered

2. You may omit:
   - Redundant information
   - Verbose tool outputs that have already been processed
   - Intermediate reasoning that led to a final decision

3. Format your summary as a coherent narrative that could serve as context for continuing the conversation. Write it as if briefing someone who needs to continue this task.

4. Keep the summary concise but complete. Aim for a summary that captures the essence of {len(messages_to_compress)} messages in a readable format.

OUTPUT YOUR SUMMARY:"""
        
        return prompt

    def _build_file_compression_prompt(
        self, 
        messages_to_compress: List[Message],
        categorized: CategorizedMessages,
        tools_used: List[str],
        key_files: List[str],
    ) -> str:
        """
        Build a prompt for file-based compression that requests structured JSON output.
        
        The LLM will provide:
        - A semantic filename for the archive
        - A brief summary for the history index
        - A detailed summary for the context file
        """
        formatted_messages = []
        for i, msg in enumerate(messages_to_compress, 1):
            formatted_messages.append(f"--- Message {i} ---\n{self._format_message_for_prompt(msg)}")
        
        conversation_text = "\n\n".join(formatted_messages)
        
        prompt = f"""You are a conversation archiver for an AI coding assistant. Analyze the following conversation and provide structured metadata for archiving.

CONVERSATION TO ARCHIVE:
{conversation_text}

CONTEXT:
- Tools used: {', '.join(tools_used) if tools_used else 'None'}
- Key files: {', '.join(key_files) if key_files else 'None'}
- Number of tool calls: {len(categorized.tool_calls)}
- Number of file reads: {len(categorized.file_reads)}

INSTRUCTIONS:
Analyze this conversation and output a JSON object with the following fields:

1. "filename": A descriptive snake_case filename (without .md extension) that captures what this conversation segment is about. Examples: "user_authentication_implementation", "database_schema_exploration", "api_debugging_session", "file_structure_analysis"

2. "summary": A brief 1-2 sentence summary suitable for an index/table of contents. This should quickly convey what happened in this segment.

3. "detailed_summary": A comprehensive summary (2-4 paragraphs) that preserves:
   - The user's goals and what was accomplished
   - Key decisions made and their rationale
   - Important outcomes from tool calls
   - Any errors or issues that were encountered
   - Context needed to continue the conversation

OUTPUT ONLY VALID JSON (no markdown code blocks, no extra text):"""
        
        return prompt

    def _format_archive_content(
        self,
        messages_to_compress: List[Message],
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
        
        # Add file reads section (with more detail preserved)
        if categorized.file_reads:
            lines.extend(["## File Reads", ""])
            for entry in categorized.file_reads:
                tool_use = entry.get("tool_use")
                tool_result = entry.get("tool_result")
                
                file_path = "unknown"
                if tool_use and hasattr(tool_use, "input"):
                    for key in ["path", "file_path", "filename", "target_file"]:
                        if key in tool_use.input:
                            file_path = tool_use.input[key]
                            break
                
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

    def _parse_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return defaults if parsing fails
            return {
                "filename": "context",
                "summary": "Conversation context",
                "detailed_summary": response_text,
            }

    def _compress_with_llm(self, messages_to_compress: List[Message]) -> str:
        """
        Use the summarizer LLM to generate an intelligent summary.
        
        If an archiver is available, uses file-based compression:
        1. Categorizes messages
        2. Asks LLM for semantic filename and summaries
        3. Archives full content to file
        4. Returns brief summary with file reference
        
        Otherwise falls back to inline summarization.
        """
        # If no archiver, use simple inline summarization
        if not self.archiver:
            return self._compress_with_llm_inline(messages_to_compress)
        
        # File-based compression
        # Step 1: Categorize messages
        categorized = MessageCategorizer.categorize(messages_to_compress)
        tools_used, key_files = MessageCategorizer.extract_tools_and_files(categorized)
        
        # Step 2: Get semantic filename and summaries from LLM
        prompt = self._build_file_compression_prompt(
            messages_to_compress, 
            categorized, 
            tools_used, 
            key_files
        )
        
        summary_request = Message(
            role=MessageRole.USER,
            content=[TextBlock(text=prompt)]
        )
        
        response = self.summarizer_llm.generate_content(
            system_prompt="You are a helpful assistant that analyzes conversations and outputs structured JSON.",
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
        
        # Parse the JSON response
        parsed = self._parse_llm_json_response(response_text)
        filename = parsed.get("filename", "context")
        summary = parsed.get("summary", "Conversation context")
        detailed_summary = parsed.get("detailed_summary", summary)
        
        # Step 3: Format and archive the content
        archive_content = self._format_archive_content(
            messages_to_compress,
            categorized,
            detailed_summary,
        )
        
        # Calculate message range
        # We don't have message indices directly, so use count
        message_range = f"{len(messages_to_compress)} messages"
        
        # Write to archive file
        archive_path = self.archiver.write_context_file(
            filename=filename,
            content=archive_content,
            message_range=message_range,
            tools_used=tools_used,
            key_files=key_files,
            summary=summary,
        )
        
        # Step 4: Return summary with file reference
        return f"[Previous context archived to: {archive_path}]\n\n{summary}"

    def _compress_with_llm_inline(self, messages_to_compress: List[Message]) -> str:
        """
        Fallback: Use the summarizer LLM for inline summarization (no file archival).
        """
        prompt = self._build_compression_prompt(messages_to_compress)
        
        # Create a temporary conversation for the summarizer
        summary_request = Message(
            role=MessageRole.USER,
            content=[TextBlock(text=prompt)]
        )
        
        # Call the summarizer LLM
        response = self.summarizer_llm.generate_content(
            system_prompt="You are a helpful assistant that summarizes conversations accurately and concisely.",
            conversation_history=[summary_request],
            tools=None
        )
        
        # Extract text from response
        if isinstance(response.content, str):
            return response.content
        
        text_parts = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        
        return "\n".join(text_parts) if text_parts else "[Summary generation failed]"

    def _compress_with_truncation(self, messages_to_compress: List[Message]) -> str:
        """Fallback: compress using simple truncation (original behavior)."""
        summary_parts = [f"[Summary of {len(messages_to_compress)} previous turns:]"]
        for msg in messages_to_compress:
            summary_parts.append(f"- {msg.role.value.upper()}: {self._summarize_content(msg.content)}")
        return "\n".join(summary_parts)

    def get_history(self) -> List[Message]:
        """
        Return the current conversation history.
        
        If archived context exists, injects a reference to the history.md
        file so the agent knows about available context files.
        """
        history = list(self._messages)
        
        # If we have archived context, inject reference into the first message
        if self.archiver and self.archiver.has_archived_context():
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

    def clear(self):
        """Clear all stored messages and reset token count."""
        self._messages.clear()
        self._total_tokens = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory into a dictionary.
        
        Note: summarizer_llm and logger are not serialized and must be
        re-injected when deserializing via from_dict/from_json.
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
            "session_id": self.session_id,
        }
        
        # Include archiver state if present
        if self.archiver:
            result["archiver"] = self.archiver.to_dict()
        
        return result

    def to_json(self) -> str:
        """Serialize the memory into a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=True)

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        summarizer_llm: Optional["LLM"] = None,
        logger: Optional["Logger"] = None,
    ) -> "FullCompressionMemory":
        """Construct memory from a serialized dictionary payload.
        
        Args:
            payload: Serialized memory data
            summarizer_llm: Optional LLM instance for intelligent compression
            logger: Optional logger for compression events
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
        session_id = payload.get("session_id")
        
        # Deserialize compression config
        config_data = payload.get("compression_config", {})
        # Handle backward compatibility: convert old keep_recent (message count) to tokens
        # Rough estimate: 25 messages * ~400 tokens per message = ~10k tokens
        if "keep_recent" in config_data and "keep_recent_tokens" not in config_data:
            # Convert old message-based config to token-based (estimate 400 tokens per message)
            keep_recent_tokens = config_data.get("keep_recent", 25) * 400
        else:
            keep_recent_tokens = config_data.get("keep_recent_tokens", 10000)
        compression_config = CompressionConfig(
            keep_recent_tokens=keep_recent_tokens,
            max_summary_tokens=config_data.get("max_summary_tokens", 2000),
            preserve_tool_names=config_data.get("preserve_tool_names", True),
            archive_tool_results=config_data.get("archive_tool_results", True),
            archive_file_reads=config_data.get("archive_file_reads", True),
            summary_style=config_data.get("summary_style", "narrative"),
            compression_ratio=config_data.get("compression_ratio", 0.3),
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
            session_id=session_id,
            archiver=archiver,
        )

    @classmethod
    def from_json(
        cls,
        data: Optional[str],
        summarizer_llm: Optional["LLM"] = None,
        logger: Optional["Logger"] = None,
        session_id: Optional[str] = None,
    ) -> "FullCompressionMemory":
        """Construct memory from a JSON string.
        
        Args:
            data: JSON string of serialized memory
            summarizer_llm: Optional LLM instance for intelligent compression
            logger: Optional logger for compression events
            session_id: Optional session ID (used if not present in serialized data)
        """
        if not data:
            return cls(summarizer_llm=summarizer_llm, logger=logger, session_id=session_id)
        
        payload = json.loads(data)
        # Use provided session_id if not in payload
        if session_id and not payload.get("session_id"):
            payload["session_id"] = session_id
        
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
