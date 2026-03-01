"""
Research Partner Agent

A CAL agent that helps with research by searching Wikipedia,
taking notes, and summarizing findings.

This example demonstrates how to build a complete agent using the CAL library.
It's designed to be easy to follow and modify for your own projects.

Prerequisites:
    1. Install CAL: pip install git+https://github.com/Creevo-App/creevo-agent-library.git
    2. Set GEMINI_API_KEY in .env or environment
       Get a free API key at: https://makersuite.google.com/app/apikey

Usage:
    # Interactive mode (recommended for exploring):
    python agent.py
    
    # Single query mode:
    python agent.py "Tell me about the history of the internet"
    
    # From the project root:
    python -m examples.research_partner.agent "What is machine learning?"

What You'll Learn:
    - How to create an Agent with the CAL library
    - How to configure the LLM (Gemini)
    - How to set up memory and context management
    - How to register custom tools
    - How to run the agent and extract responses
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# CAL library imports
from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory

# Local imports (our custom prompt and tools)
# This handles both running as a module and directly
if __name__ == "__main__" and __package__ is None:
    from prompt import SYSTEM_PROMPT
    from tools import web_search, read_webpage, save_note, read_notes
else:
    from .prompt import SYSTEM_PROMPT
    from .tools import web_search, read_webpage, save_note, read_notes

# =============================================================================
# CONFIGURATION
# Load environment variables from .env file
# =============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =============================================================================
# AGENT FACTORY
# This function creates and configures our research agent
# =============================================================================

async def create_research_agent() -> Agent:
    """
    Create and configure the Research Partner agent.
    
    This function demonstrates the key steps to build a CAL agent:
    1. Validate the API key
    2. Configure the LLM (language model)
    3. Set up the context policy (memory management)
    4. Create the memory engine
    5. Register tools
    6. Create the agent
    
    Returns:
        A configured Agent ready to run
    """
    
    # Step 1: Check that we have an API key
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set!\n"
            "Please set it in a .env file or as an environment variable.\n"
            "Get a free key at: https://makersuite.google.com/app/apikey"
        )
    
    # Step 2: Configure the LLM
    # GeminiLLM is our interface to Google's Gemini API
    llm = GeminiLLM(
        model="gemini-2.0-flash",  # Fast, capable model
        api_key=GEMINI_API_KEY,
        max_tokens=4096,  # Maximum response length
    )
    
    # Step 3: Create memory with compression support
    # Uses a summarizer LLM for intelligent context compression
    summarizer_llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        max_tokens=2048,
    )
    
    memory = FullCompressionMemory(
        summarizer_llm=summarizer_llm,
        max_tokens=50_000,
    )
    
    # Step 4: Register tools
    # Tools are functions the agent can call to perform actions
    tools = [
        StopTool(),       # Built-in: signals task completion
        web_search,       # Our custom: search the web via DuckDuckGo
        read_webpage,     # Our custom: read webpage content
        save_note,        # Our custom: save research notes
        read_notes,       # Our custom: read saved notes
    ]
    
    # Step 5: Create the agent
    # This brings everything together
    agent = Agent(
        llm=llm,                         # The language model to use
        system_prompt=SYSTEM_PROMPT,     # Agent's instructions/personality
        max_calls=15,                    # Max tool calls per run (safety limit)
        max_tokens=4096,                 # Max tokens per response
        memory=memory,                   # How to manage context
        agent_name="research-partner",   # Name for logging
        tools=tools,                     # Available tools
    )
    
    return agent


# =============================================================================
# INTERACTIVE MODE
# Run the agent in a loop, accepting user input
# =============================================================================

async def run_interactive():
    """
    Run the Research Partner in interactive mode.
    
    This creates a chat-like interface where you can:
    - Ask questions about any topic
    - Have the agent research and summarize information
    - Save and retrieve notes
    
    The agent maintains conversation context between messages,
    so you can have a natural back-and-forth conversation.
    """
    
    # Print a nice header
    print("\n" + "=" * 60)
    print("  Research Partner Agent")
    print("=" * 60)
    print("\nInitializing...")
    
    # Create the agent
    agent = await create_research_agent()
    
    # Show what tools are available
    print(f"\nAgent ready with {len(agent.tools)} tools:")
    for t in agent.tools:
        print(f"  - {t.name}")
    
    # Print usage instructions
    print("\n" + "-" * 60)
    print("I can help you research any topic!")
    print("\nTry asking things like:")
    print("  - Tell me about quantum computing")
    print("  - What's the history of the internet?")
    print("  - Save a note about what we learned")
    print("  - What notes do I have saved?")
    print("\nType 'quit' or 'exit' to end the session.")
    print("-" * 60 + "\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
        except EOFError:
            # Handle Ctrl+D
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        # Check for exit commands
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        # Run the agent with the user's input
        print("\nResearching...")
        await agent.run_async(user_input)
        
        # Extract and display the response
        final_response = agent.get_final_response()
        if final_response:
            print(f"\nResearch Partner: {final_response}")
    
    print("\nGoodbye! Happy researching!")


# =============================================================================
# SINGLE QUERY MODE
# Run a single query and exit
# =============================================================================

async def run_single_query(query: str):
    """
    Run a single research query and exit.
    
    This mode is useful for:
    - Quick lookups from the command line
    - Scripting and automation
    - Testing your agent
    
    Args:
        query: The research question to answer
    """
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    # Create and run the agent
    agent = await create_research_agent()
    await agent.run_async(query)
    
    # Display the response
    final_response = agent.get_final_response()
    if final_response:
        print(f"\n{final_response}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the Research Partner agent.
    
    Usage:
        python agent.py                    # Interactive mode
        python agent.py "your question"    # Single query mode
    """
    
    if len(sys.argv) > 1:
        # If arguments provided, treat them as a query
        query = " ".join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        # Otherwise, run in interactive mode
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
