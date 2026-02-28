"""
Research Partner Agent

A CAL agent that helps with research by searching Wikipedia,
taking notes, and summarizing findings.

This example demonstrates how to build a complete agent using the CAL library.
It's designed to be easy to follow and modify for your own projects.

Prerequisites:
    1. Set GEMINI_API_KEY in .env or environment
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
from pathlib import Path

# =============================================================================
# PATH SETUP
# This lets us import from the local CAL source code
# =============================================================================
_src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# =============================================================================
# IMPORTS
# =============================================================================
from dotenv import load_dotenv

# CAL library imports
from CAL import Agent, GeminiLLM, StopTool, DefaultMemoryEngine, ContextPolicy
from CAL.content_blocks import TextBlock
from CAL.message import MessageRole

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
    
    # Step 3: Configure the context policy
    # This controls how much conversation history the agent remembers
    # Token budgets determine how much context goes to the LLM
    context_policy = ContextPolicy(
        total_token_budget=50_000,  # Total tokens available for context
        recent_tokens=18_000,       # Recent conversation turns
        semantic_tokens=12_000,     # Semantically relevant past context
        working_tokens=5_000,       # Working memory (important facts)
    )
    
    # Step 4: Create the memory engine
    # This manages the agent's conversation history and context
    memory_engine = DefaultMemoryEngine(context_policy=context_policy)
    
    # Step 5: Register tools
    # Tools are functions the agent can call to perform actions
    tools = [
        StopTool(),       # Built-in: signals task completion
        web_search,       # Our custom: search the web via DuckDuckGo
        read_webpage,     # Our custom: read webpage content
        save_note,        # Our custom: save research notes
        read_notes,       # Our custom: read saved notes
    ]
    
    # Step 6: Create the agent
    # This brings everything together
    agent = Agent(
        llm=llm,                         # The language model to use
        system_prompt=SYSTEM_PROMPT,     # Agent's instructions/personality
        max_calls=15,                    # Max tool calls per run (safety limit)
        max_tokens=4096,                 # Max tokens per response
        memory_engine=memory_engine,     # How to manage context
        context_policy=context_policy,   # Memory budgets
        agent_name="research-partner",   # Name for logging
        tools=tools,                     # Available tools
    )
    
    return agent


# =============================================================================
# RESPONSE EXTRACTION
# Helper function to get the agent's final text response
# =============================================================================

def extract_final_response(agent: Agent) -> str:
    """
    Extract the final text response from the agent's conversation history.
    
    The agent's conversation is stored as a list of messages.
    This function finds the last assistant message and extracts its text.
    
    Args:
        agent: The agent after running
    
    Returns:
        The final text response from the agent
    """
    text_parts = []
    
    # Look through all messages in the conversation
    for msg in agent.conversation_history:
        # Only look at assistant messages (not user or tool results)
        if msg.role == MessageRole.ASSISTANT and isinstance(msg.content, list):
            for block in msg.content:
                # Extract text from TextBlocks
                if isinstance(block, TextBlock) and block.text.strip():
                    text_parts.append(block.text)
    
    # Return the last text part (most recent response)
    return text_parts[-1] if text_parts else ""


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
        final_response = extract_final_response(agent)
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
    final_response = extract_final_response(agent)
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
