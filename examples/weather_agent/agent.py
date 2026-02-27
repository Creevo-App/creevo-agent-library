"""
Weather Agent Example

A CAL agent that provides weather information using the Open-Meteo API.
No API key required - Open-Meteo is a free, open-source weather service.

Features:
- Location search by city name
- Current weather conditions
- Multi-day forecasts (up to 16 days)
- Hourly forecasts (up to 48 hours)
- Multi-location comparison

This example showcases:
- Custom tools calling external APIs
- DefaultMemoryEngine with ContextPolicy
- Practical conversational weather assistant

Usage:
    python agent.py

Requirements:
    - GEMINI_API_KEY in .env file
    - No weather API key needed (Open-Meteo is free)
"""

import os
from CAL import (
    Agent,
    GeminiLLM,
    StopTool,
    DefaultMemoryEngine,
    ContextPolicy,
)
from dotenv import load_dotenv
from tools import (
    search_location,
    get_current_weather,
    get_weather_forecast,
    get_hourly_forecast,
    compare_weather,
)
from prompt import SYSTEM_PROMPT

load_dotenv()

llm = GeminiLLM(
    model='gemini-3-flash-preview',
    api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=4096
)

memory_engine = DefaultMemoryEngine()

context_policy = ContextPolicy(
    total_token_budget=30000,
    recent_tokens=12000,
    semantic_tokens=8000,
    working_tokens=3000,
)

agent = Agent(
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
    max_calls=30,
    max_tokens=4096,
    memory_engine=memory_engine,
    context_policy=context_policy,
    thread_id="weather-thread",
    resource_id="weather-user",
    agent_name="weather-assistant",
    tools=[
        StopTool(),
        search_location,
        get_current_weather,
        get_weather_forecast,
        get_hourly_forecast,
        compare_weather,
    ]
)


def main():
    print("=" * 60)
    print("Weather Assistant")
    print("=" * 60)
    print("\nAsk me about the weather anywhere in the world!")
    print("Examples:")
    print("  - What's the weather in Tokyo?")
    print("  - Will it rain in London tomorrow?")
    print("  - Compare weather in New York and Los Angeles")
    print("  - 7-day forecast for Sydney, Australia")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Stay weather-aware! 🌤️")
                break
            
            if not user_input:
                continue
            
            result = agent.run(user_input)
            print(f"\nAssistant: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Stay weather-aware! 🌤️")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
