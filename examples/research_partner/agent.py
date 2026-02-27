"""
Research Partner Agent

A CAL agent that searches the web for information and saves research notes
to a local folder. Uses Tavily MCP server for web search capabilities.

Prerequisites:
    1. Set GEMINI_API_KEY in .env or environment
    2. Set TAVILY_API_KEY in .env or environment (get one at https://tavily.com)
    3. Node.js / npx installed (for MCP server)

Usage:
    python -m examples.research_partner.agent "Your research topic here"
    
    Or interactively:
    python -m examples.research_partner.agent
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from CAL import Agent, GeminiLLM, StopTool, DefaultMemoryEngine, ContextPolicy
from CAL.content_blocks import TextBlock
from CAL.message import MessageRole
from CAL.mcp import connect_mcp_server, disconnect_mcp_tools

if __name__ == "__main__" and __package__ is None:
    from prompt import SYSTEM_PROMPT
    from tools import (
        RESEARCH_NOTES_DIR,
        save_research_note,
        append_to_research_note,
        list_research_notes,
        read_research_note,
    )
else:
    from .prompt import SYSTEM_PROMPT
    from .tools import (
        RESEARCH_NOTES_DIR,
        save_research_note,
        append_to_research_note,
        list_research_notes,
        read_research_note,
    )

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


async def create_research_agent(mcp_tools: list = None) -> Agent:
    """Create and configure the research partner agent."""
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Please set it in .env or environment.")
    
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        max_tokens=8192,
    )
    
    context_policy = ContextPolicy(
        total_token_budget=100_000,
        recent_tokens=40_000,
        semantic_tokens=20_000,
        working_tokens=10_000,
    )
    
    memory_engine = DefaultMemoryEngine(context_policy=context_policy)
    
    tools = [
        StopTool(),
        save_research_note,
        append_to_research_note,
        list_research_notes,
        read_research_note,
    ]
    
    if mcp_tools:
        tools.extend(mcp_tools)
    
    agent = Agent(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        max_calls=30,
        max_tokens=8192,
        memory_engine=memory_engine,
        context_policy=context_policy,
        agent_name="research-partner",
        tools=tools,
    )
    
    return agent


async def connect_search_mcp() -> list:
    """Connect to Tavily MCP server for web search capabilities."""
    
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY not set. Web search will not be available.")
        print("Get an API key at https://tavily.com and set TAVILY_API_KEY in .env")
        return []
    
    try:
        mcp_tools = await connect_mcp_server(
            command="npx",
            args=["-y", "tavily-mcp@0.1.4"],
            env={"TAVILY_API_KEY": TAVILY_API_KEY},
        )
        print(f"Connected to Tavily MCP - discovered {len(mcp_tools)} tool(s)")
        for t in mcp_tools:
            print(f"  - {t.name}: {t.description[:80]}...")
        return mcp_tools
    except Exception as e:
        print(f"Warning: Could not connect to Tavily MCP server: {e}")
        print("Web search will not be available. Continuing with note-taking only.")
        return []


def extract_final_response(agent: Agent) -> str:
    """Extract the final text response from the agent's conversation history."""
    text_parts = []
    for msg in agent.conversation_history:
        if msg.role == MessageRole.ASSISTANT and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    text_parts.append(block.text)
    
    return text_parts[-1] if text_parts else ""


async def run_interactive():
    """Run the research agent in interactive mode."""
    
    print("\n" + "=" * 60)
    print("  Research Partner Agent")
    print("=" * 60)
    print("\nConnecting to web search service...")
    
    mcp_tools = await connect_search_mcp()
    agent = await create_research_agent(mcp_tools)
    
    print(f"\nAgent ready with {len(agent.tools)} tools:")
    for t in agent.tools:
        print(f"  - {t.name}")
    
    print("\n" + "-" * 60)
    print("Enter your research topics or questions.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'notes' to list all saved research notes.")
    print("-" * 60 + "\n")
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                break
            
            if user_input.lower() == "notes":
                result = await list_research_notes()
                print(f"\nResearch Partner: {result['content'][0]['text']}")
                continue
            
            print("\nResearching...")
            await agent.run_async(user_input)
            
            final_response = extract_final_response(agent)
            if final_response:
                print(f"\nResearch Partner: {final_response}")
            
    finally:
        if mcp_tools:
            await disconnect_mcp_tools(mcp_tools)
            print("\nDisconnected from search service.")
    
    print("\nSession ended. Research notes saved to:", RESEARCH_NOTES_DIR.absolute())


async def run_single_query(query: str):
    """Run a single research query and exit."""
    
    print(f"\nResearching: {query}")
    print("-" * 60)
    
    mcp_tools = await connect_search_mcp()
    
    try:
        agent = await create_research_agent(mcp_tools)
        await agent.run_async(query)
        
        final_response = extract_final_response(agent)
        if final_response:
            print(f"\n{final_response}")
        
        print("\n" + "-" * 60)
        print(f"Research notes saved to: {RESEARCH_NOTES_DIR.absolute()}")
        
    finally:
        if mcp_tools:
            await disconnect_mcp_tools(mcp_tools)


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
