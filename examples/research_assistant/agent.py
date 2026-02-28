"""
Research Assistant Agent Example

A practical example demonstrating how to build a CAL agent that can:
- Search the web for information on any topic
- Save organized research notes
- Generate comprehensive research reports

This example showcases:
- Custom tool creation with the @tool decorator
- Memory management with DefaultMemoryEngine and ContextPolicy
- Structured system prompts
- Tool return format patterns

Usage:
    python agent.py

Requirements:
    - GEMINI_API_KEY in .env file
    - TAVILY_API in .env file (for web search)
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
from tools import web_search, save_note, list_notes, read_note, generate_report
from prompt import SYSTEM_PROMPT

load_dotenv()

llm = GeminiLLM(
    model='gemini-3-flash-preview',
    api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=4096
)

memory_engine = DefaultMemoryEngine()

context_policy = ContextPolicy(
    total_token_budget=50000,
    recent_tokens=18000,
    semantic_tokens=12000,
    working_tokens=5000,
)

agent = Agent(
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
    max_calls=50,
    max_tokens=4096,
    memory_engine=memory_engine,
    context_policy=context_policy,
    thread_id="research-thread",
    resource_id="research-user",
    agent_name="research-assistant",
    tools=[
        StopTool(),
        web_search,
        save_note,
        list_notes,
        read_note,
        generate_report
    ]
)


def main():
    print("=" * 60)
    print("Research Assistant Agent")
    print("=" * 60)
    print("\nI can help you research any topic. I'll search the web,")
    print("take notes, and generate reports for you.\n")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Your research notes are saved in ./research_notes/")
                break
            
            if not user_input:
                continue
            
            result = agent.run(user_input)
            print(f"\nAssistant: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Your research notes are saved in ./research_notes/")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
