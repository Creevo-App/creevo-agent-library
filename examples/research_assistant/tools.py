import httpx
import os
import json
from datetime import datetime
from CAL import tool
from dotenv import load_dotenv

load_dotenv()


@tool
async def web_search(query: str):
    """Search the web for information on any topic.
    
    Args:
        query: The search query string to find relevant information
    
    Returns:
        Search results with titles, URLs, and content summaries
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.getenv("TAVILY_API"),
                "query": query,
                "max_results": 5
            }
        )
        results = response.json().get("results", [])
    
    formatted = "\n\n".join(
        f"**{r['title']}**\n{r['url']}\n{r['content']}" for r in results
    )
    
    return {
        "content": [{"type": "text", "text": formatted or "No results found."}],
        "metadata": {"query": query, "result_count": len(results)}
    }


@tool
async def save_note(title: str, content: str, tags: str = ""):
    """Save a research note to a file for later reference.
    
    Args:
        title: Title of the note (will be used as filename, sanitized)
        content: The note content to save
        tags: Optional comma-separated tags for categorization
    
    Returns:
        Confirmation with the saved filename
    """
    os.makedirs("research_notes", exist_ok=True)
    
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_notes/{safe_title}_{timestamp}.md"
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tag_line = f"Tags: {', '.join(tag_list)}" if tag_list else ""
    
    note_content = f"""# {title}

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{tag_line}

---

{content}
"""
    
    with open(filename, "w") as f:
        f.write(note_content)
    
    return {
        "content": [{"type": "text", "text": f"Note saved to: {filename}"}],
        "metadata": {"filename": filename, "tags": tag_list}
    }


@tool
async def list_notes():
    """List all saved research notes.
    
    Returns:
        A list of all saved note files with their metadata
    """
    notes_dir = "research_notes"
    
    if not os.path.exists(notes_dir):
        return {
            "content": [{"type": "text", "text": "No research notes found yet."}],
            "metadata": {"count": 0}
        }
    
    notes = []
    for filename in sorted(os.listdir(notes_dir), reverse=True):
        if filename.endswith(".md"):
            filepath = os.path.join(notes_dir, filename)
            with open(filepath, "r") as f:
                first_lines = f.read(500)
            
            title = filename.replace("_", " ").replace(".md", "")
            notes.append(f"- **{filename}**\n  Preview: {first_lines[:200]}...")
    
    if not notes:
        return {
            "content": [{"type": "text", "text": "No research notes found yet."}],
            "metadata": {"count": 0}
        }
    
    formatted = f"## Research Notes ({len(notes)} files)\n\n" + "\n\n".join(notes)
    
    return {
        "content": [{"type": "text", "text": formatted}],
        "metadata": {"count": len(notes)}
    }


@tool
async def read_note(filename: str):
    """Read the full content of a specific research note.
    
    Args:
        filename: The filename of the note to read
    
    Returns:
        The full content of the note
    """
    filepath = os.path.join("research_notes", filename)
    
    if not os.path.exists(filepath):
        return {
            "content": [{"type": "text", "text": f"Note not found: {filename}"}],
            "metadata": {"found": False}
        }
    
    with open(filepath, "r") as f:
        content = f.read()
    
    return {
        "content": [{"type": "text", "text": content}],
        "metadata": {"found": True, "filename": filename}
    }


@tool
async def generate_report(topic: str, notes_to_include: str = ""):
    """Generate a comprehensive research report from notes.
    
    Args:
        topic: The main topic/title for the report
        notes_to_include: Optional comma-separated list of note filenames to include
    
    Returns:
        Confirmation of report generation
    """
    os.makedirs("research_reports", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
    safe_topic = safe_topic.replace(" ", "_").lower()
    filename = f"research_reports/{safe_topic}_report_{timestamp}.md"
    
    notes_content = []
    if notes_to_include:
        note_files = [n.strip() for n in notes_to_include.split(",") if n.strip()]
        for note_file in note_files:
            filepath = os.path.join("research_notes", note_file)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    notes_content.append(f"### From: {note_file}\n\n{f.read()}")
    
    report_content = f"""# Research Report: {topic}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Overview

This report compiles research findings on: {topic}

## Compiled Notes

{chr(10).join(notes_content) if notes_content else "No specific notes included. Use save_note to capture findings, then regenerate this report."}

---

*Generated by Research Assistant Agent*
"""
    
    with open(filename, "w") as f:
        f.write(report_content)
    
    return {
        "content": [{"type": "text", "text": f"Report generated: {filename}"}],
        "metadata": {"filename": filename, "notes_included": len(notes_content)}
    }
