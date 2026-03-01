"""
Weather Agent

A CAL agent that reports current weather conditions for any city
using the free Open-Meteo API (no API key required).

Prerequisites:
    1. Install CAL: pip install git+https://github.com/Creevo-App/creevo-agent-library.git
    2. Set GEMINI_API_KEY in .env or environment

Usage:
    python -m examples.weather_agent.agent "What's the weather in Tokyo?"
    
    Or interactively:
    python -m examples.weather_agent.agent
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory

if __name__ == "__main__" and __package__ is None:
    from prompt import SYSTEM_PROMPT
    from tools import geocode_city, get_current_weather
else:
    from .prompt import SYSTEM_PROMPT
    from .tools import geocode_city, get_current_weather

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def create_weather_agent() -> Agent:
    """Create and configure the weather agent."""
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Please set it in .env or environment.")
    
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        max_tokens=4096,
    )
    
    summarizer_llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        max_tokens=2048,
    )
    
    memory = FullCompressionMemory(
        summarizer_llm=summarizer_llm,
        max_tokens=50_000,
    )
    
    tools = [
        StopTool(),
        geocode_city,
        get_current_weather,
    ]
    
    agent = Agent(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        max_calls=10,
        max_tokens=4096,
        memory=memory,
        agent_name="weather-agent",
        tools=tools,
    )
    
    return agent


async def run_interactive():
    """Run the weather agent in interactive mode."""
    
    print("\n" + "=" * 60)
    print("  Weather Agent")
    print("=" * 60)
    print("\nInitializing...")
    
    agent = await create_weather_agent()
    
    print(f"\nAgent ready with {len(agent.tools)} tools:")
    for t in agent.tools:
        print(f"  - {t.name}")
    
    print("\n" + "-" * 60)
    print("Ask about the weather in any city!")
    print("Examples:")
    print("  - What's the weather in Paris?")
    print("  - How's the weather in Tokyo, Japan?")
    print("  - Is it raining in London?")
    print("\nType 'quit' or 'exit' to end the session.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            break
        
        print("\nChecking weather...")
        await agent.run_async(user_input)
        
        final_response = agent.get_final_response()
        if final_response:
            print(f"\nWeather Agent: {final_response}")
    
    print("\nGoodbye!")


async def run_single_query(query: str):
    """Run a single weather query and exit."""
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    agent = await create_weather_agent()
    await agent.run_async(query)
    
    final_response = agent.get_final_response()
    if final_response:
        print(f"\n{final_response}")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
