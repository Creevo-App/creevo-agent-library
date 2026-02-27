"""Custom tools for the Research Partner agent."""

from datetime import datetime
from pathlib import Path

from CAL import tool

RESEARCH_NOTES_DIR = Path(__file__).parent / "notes"


def ensure_notes_directory():
    """Create the research-notes directory if it doesn't exist."""
    RESEARCH_NOTES_DIR.mkdir(parents=True, exist_ok=True)


@tool
async def save_research_note(
    topic: str,
    content: str,
    sources: str = "",
) -> dict:
    """
    Save a research note to the research-notes folder.
    
    Args:
        topic: The research topic or title for this note
        content: The main content/findings to save
        sources: Optional list of source URLs or references used
    
    Returns:
        Confirmation of the saved note with its file path
    """
    ensure_notes_directory()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
    safe_topic = safe_topic.replace(" ", "_")[:50]
    filename = f"{timestamp}_{safe_topic}.md"
    filepath = RESEARCH_NOTES_DIR / filename
    
    note_content = f"""# {topic}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Research Findings

{content}
"""
    
    if sources:
        note_content += f"""
## Sources

{sources}
"""
    
    filepath.write_text(note_content, encoding="utf-8")
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Research note saved successfully to: {filepath}",
            }
        ],
        "metadata": {"filepath": str(filepath), "topic": topic},
    }


@tool
async def append_to_research_note(
    filename: str,
    content: str,
    section_title: str = "",
) -> dict:
    """
    Append additional content to an existing research note.
    
    Args:
        filename: The filename of the existing note (in research-notes folder)
        content: The content to append
        section_title: Optional section heading for the appended content
    
    Returns:
        Confirmation of the update
    """
    filepath = RESEARCH_NOTES_DIR / filename
    
    if not filepath.exists():
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: Note file '{filename}' not found in research-notes folder",
                }
            ],
            "metadata": {"error": True},
        }
    
    append_content = "\n\n"
    if section_title:
        append_content += f"## {section_title}\n\n"
    append_content += content
    
    with filepath.open("a", encoding="utf-8") as f:
        f.write(append_content)
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Content appended to: {filepath}",
            }
        ],
        "metadata": {"filepath": str(filepath)},
    }


@tool
async def list_research_notes() -> dict:
    """
    List all existing research notes in the research-notes folder.
    
    Returns:
        A list of all note files with their topics and dates
    """
    ensure_notes_directory()
    
    notes = list(RESEARCH_NOTES_DIR.glob("*.md"))
    
    if not notes:
        return {
            "content": [
                {"type": "text", "text": "No research notes found yet."}
            ],
            "metadata": {"count": 0},
        }
    
    notes_list = []
    for note in sorted(notes, key=lambda p: p.stat().st_mtime, reverse=True):
        first_line = note.read_text(encoding="utf-8").split("\n")[0]
        title = first_line.replace("# ", "") if first_line.startswith("# ") else note.stem
        notes_list.append(f"- **{note.name}**: {title}")
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {len(notes)} research note(s):\n\n" + "\n".join(notes_list),
            }
        ],
        "metadata": {"count": len(notes), "files": [n.name for n in notes]},
    }


@tool
async def read_research_note(filename: str) -> dict:
    """
    Read the contents of an existing research note.
    
    Args:
        filename: The filename of the note to read
    
    Returns:
        The full content of the research note
    """
    filepath = RESEARCH_NOTES_DIR / filename
    
    if not filepath.exists():
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: Note file '{filename}' not found",
                }
            ],
            "metadata": {"error": True},
        }
    
    content = filepath.read_text(encoding="utf-8")
    
    return {
        "content": [{"type": "text", "text": content}],
        "metadata": {"filepath": str(filepath)},
    }
