"""Research Partner Agent - A CAL agent for web research and note-taking."""

from .agent import create_research_agent, connect_search_mcp, run_interactive, run_single_query
from .prompt import SYSTEM_PROMPT
from .tools import (
    save_research_note,
    append_to_research_note,
    list_research_notes,
    read_research_note,
    RESEARCH_NOTES_DIR,
)

__all__ = [
    "create_research_agent",
    "connect_search_mcp",
    "run_interactive",
    "run_single_query",
    "SYSTEM_PROMPT",
    "save_research_note",
    "append_to_research_note",
    "list_research_notes",
    "read_research_note",
    "RESEARCH_NOTES_DIR",
]
