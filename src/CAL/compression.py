"""
Compression and archival utilities for conversation memory.
"""
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .message import Message
from .content_blocks import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    pass


@dataclass
class CompressionConfig:
    """Configuration for memory compression behavior."""
    keep_recent_tokens: int = 10000    # Number of recent tokens to keep uncompressed
    max_summary_tokens: int = 2000     # Max tokens for the summary output
    preserve_tool_names: bool = True   # Include tool names in summary
    archive_tool_results: bool = True  # Archive full tool results to files
    archive_file_reads: bool = True    # Archive file read contents to files
    summary_style: str = "narrative"   # "narrative", "bullet", "structured"
    compression_ratio: float = 0.3     # Target compression ratio hint


@dataclass
class ArchiveEntry:
    """Represents a single archived context file."""
    filename: str
    created_at: str
    message_range: str
    tools_used: List[str]
    key_files: List[str]
    summary: str


class CompressionArchiver:
    """
    Manages file-based archival of compressed conversation context.
    
    Creates and maintains:
    - Session-specific temp directory for context files
    - history.md index file describing archived content
    - Individual context files with tool calls and conversation data
    """
    
    def __init__(self, session_id: str, base_dir: Optional[str] = None):
        """
        Initialize the archiver.
        
        Args:
            session_id: Unique session identifier for directory naming
            base_dir: Optional base directory path. If None, uses system temp.
        """
        self.session_id = session_id
        self._entries: List[ArchiveEntry] = []
        
        if base_dir:
            self._base_dir = Path(base_dir)
        else:
            self._base_dir = Path(tempfile.gettempdir())
        
        self._session_dir = self._base_dir / f"cal_memory_{session_id}"
        self._initialized = False
    
    def _ensure_directory(self):
        """Create the session directory if it doesn't exist."""
        if not self._initialized:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
    
    @property
    def session_dir(self) -> Path:
        """Return the session directory path."""
        return self._session_dir
    
    @property
    def history_path(self) -> Path:
        """Return the path to history.md."""
        return self._session_dir / "history.md"
    
    def has_archived_context(self) -> bool:
        """Check if there is any archived context."""
        return len(self._entries) > 0
    
    def write_context_file(
        self,
        filename: str,
        content: str,
        message_range: str,
        tools_used: List[str],
        key_files: List[str],
        summary: str,
    ) -> Path:
        """
        Write a context file with the given content.
        
        Args:
            filename: Semantic filename (without .md extension)
            content: The full content to archive
            message_range: Description of message range (e.g., "15-40")
            tools_used: List of tool names used in this context
            key_files: List of key files referenced
            summary: Brief summary of the content
            
        Returns:
            Path to the created file
        """
        self._ensure_directory()
        
        # Ensure unique filename by appending counter if needed
        safe_filename = self._make_safe_filename(filename)
        file_path = self._session_dir / f"{safe_filename}.md"
        
        # Handle filename collisions
        counter = 1
        while file_path.exists():
            file_path = self._session_dir / f"{safe_filename}_{counter}.md"
            counter += 1
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Record the entry
        entry = ArchiveEntry(
            filename=file_path.name,
            created_at=datetime.utcnow().isoformat() + "Z",
            message_range=message_range,
            tools_used=tools_used,
            key_files=key_files,
            summary=summary,
        )
        self._entries.append(entry)
        
        # Update history.md
        self._update_history_file()
        
        return file_path
    
    def _make_safe_filename(self, filename: str) -> str:
        """Convert a filename to a safe format."""
        # Remove/replace unsafe characters
        safe = filename.lower()
        safe = safe.replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c == "_")
        # Ensure it's not empty and not too long
        if not safe:
            safe = "context"
        return safe[:50]
    
    def _update_history_file(self):
        """Update the history.md index file with all entries."""
        self._ensure_directory()
        
        lines = [
            "# Conversation History Index",
            "",
            f"## Session: {self.session_id}",
            f"Last Updated: {datetime.utcnow().isoformat()}Z",
            "",
            "## Archived Context",
            "",
        ]
        
        for entry in self._entries:
            lines.extend([
                f"### {entry.filename}",
                f"- **Created**: {entry.created_at}",
                f"- **Messages**: {entry.message_range}",
                f"- **Tools Used**: {', '.join(entry.tools_used) if entry.tools_used else 'None'}",
                f"- **Key Files**: {', '.join(entry.key_files) if entry.key_files else 'None'}",
                f"- **Summary**: {entry.summary}",
                "",
            ])
        
        with open(self.history_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def get_history_reference(self) -> str:
        """
        Get a text reference to the history.md for injection into context.
        
        Returns a formatted string that can be included in conversation context.
        """
        if not self.has_archived_context():
            return ""
        
        lines = [
            "[Previous conversation context has been archived to files.]",
            f"History index: {self.history_path}",
            "",
            "Archived files:",
        ]
        
        for entry in self._entries:
            file_path = self._session_dir / entry.filename
            lines.append(f"- {file_path}: {entry.summary}")
        
        lines.append("")
        lines.append("Use file reading tools to access detailed context if needed.")
        
        return "\n".join(lines)
    
    def cleanup(self):
        """Remove the session directory and all archived files."""
        if self._session_dir.exists():
            shutil.rmtree(self._session_dir)
        self._entries.clear()
        self._initialized = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the archiver state for persistence."""
        return {
            "session_id": self.session_id,
            "base_dir": str(self._base_dir),
            "entries": [
                {
                    "filename": e.filename,
                    "created_at": e.created_at,
                    "message_range": e.message_range,
                    "tools_used": e.tools_used,
                    "key_files": e.key_files,
                    "summary": e.summary,
                }
                for e in self._entries
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionArchiver":
        """Restore archiver from serialized state."""
        archiver = cls(
            session_id=data.get("session_id", "unknown"),
            base_dir=data.get("base_dir"),
        )
        
        entries_data = data.get("entries", [])
        for entry_data in entries_data:
            entry = ArchiveEntry(
                filename=entry_data.get("filename", ""),
                created_at=entry_data.get("created_at", ""),
                message_range=entry_data.get("message_range", ""),
                tools_used=entry_data.get("tools_used", []),
                key_files=entry_data.get("key_files", []),
                summary=entry_data.get("summary", ""),
            )
            archiver._entries.append(entry)
        
        # Mark as initialized if entries exist (directory should exist)
        if archiver._entries:
            archiver._initialized = True
        
        return archiver


@dataclass
class CategorizedMessages:
    """Result of message categorization for compression."""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # Paired tool use + result
    file_reads: List[Dict[str, Any]] = field(default_factory=list)  # File read tool results
    conversations: List[Message] = field(default_factory=list)       # Text-only exchanges
    images: List[Dict[str, Any]] = field(default_factory=list)       # Image blocks


class MessageCategorizer:
    """
    Categorizes messages for intelligent compression.
    
    Separates messages into:
    - Tool calls with their results
    - File read operations (preserved with more detail)
    - Regular conversation text
    - Images (reference only)
    """
    
    # Tool names that indicate file reading operations
    FILE_READ_TOOLS = {"read_file", "cat", "view_file", "get_file_content", "file_read"}
    
    @classmethod
    def categorize(cls, messages: List[Message]) -> CategorizedMessages:
        """
        Categorize a list of messages.
        
        Args:
            messages: List of messages to categorize
            
        Returns:
            CategorizedMessages with separated content types
        """
        result = CategorizedMessages()
        
        # Track tool uses waiting for results
        pending_tool_uses: Dict[str, ToolUseBlock] = {}
        
        for msg in messages:
            if isinstance(msg.content, str):
                result.conversations.append(msg)
                continue
            
            has_text_only = True
            
            for block in msg.content:
                if isinstance(block, TextBlock):
                    # Will be included in conversation if no tool blocks
                    pass
                elif isinstance(block, ToolUseBlock):
                    has_text_only = False
                    pending_tool_uses[block.id] = block
                elif isinstance(block, ToolResultBlock):
                    has_text_only = False
                    tool_use = pending_tool_uses.pop(block.tool_use_id, None)
                    
                    # Determine if this is a file read
                    tool_name = block.name or (tool_use.name if tool_use else "")
                    is_file_read = tool_name.lower() in cls.FILE_READ_TOOLS
                    
                    entry = {
                        "tool_use": tool_use,
                        "tool_result": block,
                        "tool_name": tool_name,
                    }
                    
                    if is_file_read:
                        result.file_reads.append(entry)
                    else:
                        result.tool_calls.append(entry)
                        
                elif isinstance(block, ImageBlock):
                    has_text_only = False
                    result.images.append({
                        "block": block,
                        "message_role": msg.role,
                    })
            
            # If message only has text blocks, treat as conversation
            if has_text_only and msg.content:
                result.conversations.append(msg)
        
        return result
    
    @classmethod
    def extract_tools_and_files(
        cls, 
        categorized: CategorizedMessages
    ) -> tuple[List[str], List[str]]:
        """
        Extract unique tool names and key file paths from categorized messages.
        
        Returns:
            Tuple of (tool_names, file_paths)
        """
        tools_used = set()
        key_files = set()
        
        # From tool calls
        for entry in categorized.tool_calls:
            if entry.get("tool_name"):
                tools_used.add(entry["tool_name"])
            # Try to extract file paths from tool inputs
            tool_use = entry.get("tool_use")
            if tool_use and hasattr(tool_use, "input"):
                for key in ["path", "file_path", "filename", "target_file"]:
                    if key in tool_use.input:
                        key_files.add(str(tool_use.input[key]))
        
        # From file reads
        for entry in categorized.file_reads:
            if entry.get("tool_name"):
                tools_used.add(entry["tool_name"])
            tool_use = entry.get("tool_use")
            if tool_use and hasattr(tool_use, "input"):
                for key in ["path", "file_path", "filename", "target_file"]:
                    if key in tool_use.input:
                        key_files.add(str(tool_use.input[key]))
        
        return list(tools_used), list(key_files)
