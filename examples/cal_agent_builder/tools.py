import httpx
from CAL import tool
import os
from dotenv import load_dotenv
load_dotenv()

@tool
async def web_search(query: str):
    """Search the web for information about external tools, libraries, or anything not in the CAL documentation.
    
    Args:
        query: The search query string
    
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
        "content": [{"type": "text", "text": formatted}],
        "metadata": {}
    }

@tool
async def write_file(filename: str, content: str):
    """Write code or text content to a file.
    
    Args:
        filename: The path and name of the file to write (e.g. 'my_agent.py')
        content: The content to write to the file
    
    Returns:
        Confirmation of file write with the filename
    """
    with open(filename, "w") as f:
        f.write(content)
    
    return {
        "content": [{"type": "text", "text": f"Successfully wrote to {filename}"}],
        "metadata": {}
    }