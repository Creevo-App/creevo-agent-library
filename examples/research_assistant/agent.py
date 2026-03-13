"""
Research Assistant Agent Example

A practical example demonstrating how to build a CAL agent that can:
- Search the web for information on any topic
- Save organized research notes
- Generate comprehensive research reports

This example showcases:
- Custom tool creation with the @tool decorator
- Memory management with FullCompressionMemory
- Structured system prompts
- Tool return format patterns

Prerequisites:
    1. Install CAL: pip install git+https://github.com/Creevo-App/creevo-agent-library.git
    2. Set GEMINI_API_KEY in .env file
    3. Set TAVILY_API in .env file (for web search)

Usage:
    python agent.py
"""

import os
from CAL import (
    Agent,
    GeminiLLM,
    StopTool,
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

agent = Agent(
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
    max_calls=50,
    max_tokens=4096,
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
