"""
CAL Agent with Context7 MCP Server

Standalone script version of the mcp_context7_agent.ipynb notebook.
Connects to the Context7 MCP server for library documentation lookups
using a standard CAL Agent.

Usage:
    python examples/mcp_context7_agent.py

Prerequisites:
  - Node.js / npx
  - GEMINI_API_KEY in examples/.env (or as an environment variable)
  - pip install "creevo-agent-library[mcp]"
"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory
from CAL.mcp import connect_mcp_server, disconnect_mcp_tools
from CAL.content_blocks import TextBlock, ToolUseBlock
from CAL.message import MessageRole


# ── helpers ──────────────────────────────────────────────────────────────────


def show_run(agent):
    """Print the tool-call trace followed by the agent's final text answer."""
    step = 0
    for msg in agent.memory.get_history():
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                step += 1
                args = json.dumps(block.input, ensure_ascii=False)
                if len(args) > 80:
                    args = args[:77] + "..."
                print(f"  [{step}] {block.name}({args})")

    text_parts = []
    for msg in agent.memory.get_history():
        if msg.role == MessageRole.ASSISTANT and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    text_parts.append(block.text)
    print("\n" + "=" * 60)
    if text_parts:
        print("\n".join(text_parts[-2:]))
    else:
        print("(No text response found)")


# ── main ─────────────────────────────────────────────────────────────────────


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key, "Set GEMINI_API_KEY in .env or as an environment variable"

    # 1. Connect to the Context7 MCP server
    print("Connecting to Context7 MCP server...")
    mcp_tools = await connect_mcp_server(
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
    )
    print(f"Connected — discovered {len(mcp_tools)} tool(s):\n")
    for t in mcp_tools:
        print(f"  - {t.name}: {t.description[:90]}")
    print()

    try:
        # 2. Create the agent
        llm = GeminiLLM(model="gemini-3.0-flash", api_key=api_key, max_tokens=4096)
        summarizer_llm = GeminiLLM(
            model="gemini-3.0-flash", api_key=api_key, max_tokens=2048
        )

        agent = Agent(
            llm=llm,
            system_prompt=(
                "You are a helpful coding assistant. "
                "Use the Context7 MCP tools to look up library documentation before answering. "
                "Always cite the library version you referenced. "
                "Call stop when you have a complete answer."
            ),
            max_calls=15,
            max_tokens=4096,
            memory=FullCompressionMemory(
                summarizer_llm=summarizer_llm, max_tokens=50_000
            ),
            agent_name="context7-agent",
            tools=[StopTool(), *mcp_tools],
        )

        print(f"Agent ready — {len(agent.tools)} tools registered:")
        for t in agent.tools:
            print(f"  - {t.name}")
        print()

        # 3. Run a query
        print("=" * 60)
        print("Query: What does React useEffect do?")
        print("=" * 60)
        await agent.run_async(
            "What does React useEffect do? Give a brief explanation with a small code example."
        )
        show_run(agent)

        # 4. Follow-up query (conversation memory carries context forward)
        print()
        print("=" * 60)
        print("Follow-up: useEffect with cleanup")
        print("=" * 60)
        await agent.run_async(
            "Now show me how to use useEffect with a cleanup function."
        )
        show_run(agent)

    finally:
        # 5. Cleanup
        await disconnect_mcp_tools(mcp_tools)
        print("\nMCP server disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
