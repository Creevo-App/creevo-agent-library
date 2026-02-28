"""
Custom Tools for the Research Partner Agent

This file demonstrates how to create tools for your CAL agent.

Key concepts:
1. Use the @tool decorator to turn any async function into an agent tool
2. Tools must return a dict with 'content' and 'metadata' keys
3. 'content' is a list of content blocks (usually text) that the LLM sees
4. 'metadata' is extra data you can use for logging/debugging

The @tool decorator automatically:
- Extracts the function name as the tool name
- Uses the docstring as the tool description (shown to the LLM)
- Converts function parameters to the tool's input schema
"""

from datetime import datetime
from pathlib import Path

import httpx

from CAL import tool

# Directory where we'll save research notes
NOTES_DIR = Path(__file__).parent / "notes"

# DuckDuckGo API endpoint (free, no API key required!)
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


# =============================================================================
# TOOL 1: Web Search (using DuckDuckGo)
# =============================================================================

@tool
async def web_search(query: str, num_results: int = 5) -> dict:
    """
    Search the web for information on any topic using DuckDuckGo.
    
    Use this tool to find relevant web pages, articles, and information
    about any topic. Returns titles, snippets, and URLs.
    
    Args:
        query: The search term or question to look up
        num_results: Maximum number of results to return (default: 5)
    
    Returns:
        A list of search results with titles, snippets, and URLs
    """
    import re
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    data = {"q": query, "b": ""}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                DUCKDUCKGO_HTML_URL,
                data=data,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Search failed: {str(e)}. Please try again.",
                }
            ],
            "metadata": {"error": True, "query": query},
        }
    
    results = []
    
    result_pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.+?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.+?)</a>',
        re.DOTALL
    )
    
    for match in result_pattern.finditer(html):
        if len(results) >= num_results:
            break
        
        url = match.group(1)
        title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        snippet = re.sub(r'<[^>]+>', '', match.group(3)).strip()
        
        if url and title:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })
    
    if not results:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No results found for '{query}'. Try different search terms.",
                }
            ],
            "metadata": {"query": query, "results_count": 0},
        }
    
    result_text = f"## Web Search Results for '{query}'\n\n"
    result_text += f"Found {len(results)} result(s):\n\n"
    
    for i, result in enumerate(results, 1):
        result_text += f"### {i}. {result['title']}\n"
        result_text += f"**URL:** {result['url']}\n"
        result_text += f"{result['snippet']}\n\n"
    
    result_text += "\n*Use `read_webpage` to read the full content of any page.*"
    
    return {
        "content": [{"type": "text", "text": result_text}],
        "metadata": {"query": query, "results_count": len(results), "results": results},
    }


# =============================================================================
# TOOL 2: Read Webpage Content
# =============================================================================

@tool
async def read_webpage(url: str) -> dict:
    """
    Read and extract the main text content from a webpage.
    
    Use this tool after searching to read the full content of a webpage.
    It extracts the main text and removes navigation, ads, etc.
    
    Args:
        url: The URL of the webpage to read
    
    Returns:
        The main text content of the webpage
    """
    import re
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Could not fetch webpage: {str(e)}",
                }
            ],
            "metadata": {"error": True, "url": url},
        }
    
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown Page"
    
    for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']:
        html = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    text = re.sub(r'<[^>]+>', ' ', html)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    lines = []
    for line in text.split('. '):
        line = line.strip()
        if len(line) > 20:
            lines.append(line)
    
    text = '. '.join(lines)
    
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [Content truncated for brevity]"
    
    if len(text) < 100:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Could not extract meaningful content from {url}. The page might be JavaScript-heavy or protected.",
                }
            ],
            "metadata": {"error": True, "url": url},
        }
    
    result_text = f"## {title}\n\n**Source:** {url}\n\n---\n\n{text}"
    
    return {
        "content": [{"type": "text", "text": result_text}],
        "metadata": {
            "title": title,
            "url": url,
            "char_count": len(text),
        },
    }


# =============================================================================
# TOOL 3: Save Notes
# =============================================================================

@tool
async def save_note(topic: str, content: str) -> dict:
    """
    Save research notes to a file.
    
    Use this tool to save important findings, summaries, or any information
    the user wants to remember for later.
    
    Args:
        topic: A short title/topic for the note (used as filename)
        content: The note content to save
    
    Returns:
        Confirmation that the note was saved with file path
    """
    # Ensure the notes directory exists
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from the topic
    safe_topic = "".join(c if c.isalnum() or c in "- _" else "_" for c in topic)
    safe_topic = safe_topic.strip().replace(" ", "_").lower()
    
    if not safe_topic:
        safe_topic = "research_note"
    
    # Add timestamp to make filenames unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.md"
    filepath = NOTES_DIR / filename
    
    # Format the note with metadata
    note_content = f"""# {topic}

*Saved on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}*

---

{content}
"""
    
    # Write the note to file
    filepath.write_text(note_content, encoding="utf-8")
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Note saved successfully!\n\n**Topic:** {topic}\n**File:** {filename}\n**Location:** {filepath}",
            }
        ],
        "metadata": {"filename": filename, "filepath": str(filepath), "topic": topic},
    }


# =============================================================================
# TOOL 4: Read Notes
# =============================================================================

@tool
async def read_notes(topic: str = "") -> dict:
    """
    Read saved research notes.
    
    Use this tool to recall previously saved research notes.
    You can optionally filter by topic to find specific notes.
    
    Args:
        topic: Optional topic to filter notes by (searches filenames)
    
    Returns:
        The content of matching notes or a list of available notes
    """
    # Ensure the notes directory exists
    if not NOTES_DIR.exists():
        return {
            "content": [{"type": "text", "text": "No notes have been saved yet."}],
            "metadata": {"notes_count": 0},
        }
    
    # Find all note files
    note_files = list(NOTES_DIR.glob("*.md"))
    
    if not note_files:
        return {
            "content": [{"type": "text", "text": "No notes have been saved yet."}],
            "metadata": {"notes_count": 0},
        }
    
    # Filter by topic if specified
    if topic:
        topic_lower = topic.lower()
        note_files = [f for f in note_files if topic_lower in f.stem.lower()]
    
    if not note_files:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No notes found matching '{topic}'. Try a different search term or leave topic empty to see all notes.",
                }
            ],
            "metadata": {"notes_count": 0, "filter": topic},
        }
    
    # Sort by modification time (newest first)
    note_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Read and combine note contents
    result_text = "## Your Research Notes\n\n"
    
    if topic:
        result_text += f"*Filtered by: '{topic}'*\n\n"
    
    notes_data = []
    for note_file in note_files[:5]:  # Limit to 5 most recent
        content = note_file.read_text(encoding="utf-8")
        result_text += f"---\n\n**File:** {note_file.name}\n\n{content}\n\n"
        notes_data.append({"filename": note_file.name, "content": content})
    
    if len(note_files) > 5:
        result_text += f"\n*... and {len(note_files) - 5} more note(s)*"
    
    return {
        "content": [{"type": "text", "text": result_text}],
        "metadata": {"notes_count": len(note_files), "notes": notes_data},
    }
