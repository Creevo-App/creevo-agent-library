"""
Compression and archival utilities for conversation memory.
"""
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CompressionConfig:
    """Configuration for memory compression behavior."""
    keep_recent_tokens: int = 15000    # Number of recent tokens to keep uncompressed
    max_summary_tokens: int = 8000     # Max tokens for the summary output
    preserve_tool_names: bool = True   # Include tool names in summary
    archive_tool_results: bool = True  # Archive full tool results to files
    archive_file_reads: bool = True    # Archive file read contents to files
    summary_style: str = "narrative"   # "narrative", "bullet", "structured"
    compression_ratio: float = 0.4     # Target compression ratio hint


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
    
    def __init__(self, agent_name: str, base_dir: Optional[str] = None):
        """
        Initialize the archiver.
        
        Args:
            agent_name: Unique agent identifier for directory naming
            base_dir: Optional base directory path. If None, uses system temp.
        """
        self.agent_name = agent_name
        self._entries: List[ArchiveEntry] = []
        
        if base_dir:
            self._base_dir = Path(base_dir)
        else:
            ## We have a ticket to address this in the future for temp clean up. 
            self._base_dir = Path(tempfile.gettempdir())
        
        self._session_dir = self._base_dir / f"cal_memory_{agent_name}"
        self._initialized = False
    
    def _ensure_directory(self):
        """Create the session directory if it doesn't exist."""
        if not self._initialized or not self._session_dir.exists():
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
        safe_filename = self._sanitize_filename(filename)
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
    
    def _sanitize_filename(self, filename: str) -> str:
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
            f"## Session: {self.agent_name}",
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
            "agent_name": self.agent_name,
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
            agent_name=data.get("agent_name", data.get("session_id", "unknown")),
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


