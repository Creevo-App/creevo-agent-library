import os
from CAL import Agent, GeminiLLM, StopTool, FullCompressionMemory
from dotenv import load_dotenv
from retrieve_docs import fetch_repo_context
from tools import web_search, write_file
from prompt import ROLE_PROMPT, GENERAL_GUIDELINES_PROMPT, TOOL_USAGE_PROMPT
load_dotenv()

context = fetch_repo_context()

llm = GeminiLLM(model='gemini-3-flash-preview', api_key=os.getenv("GEMINI_API_KEY"), max_tokens=4096)
summarizer_llm = GeminiLLM(model='gemini-3-flash-preview', api_key=os.getenv("GEMINI_API_KEY"), max_tokens=2048)
memory = FullCompressionMemory(summarizer_llm=summarizer_llm, max_tokens=50000)

agent = Agent(
    llm=llm,
    system_prompt=ROLE_PROMPT + GENERAL_GUIDELINES_PROMPT + TOOL_USAGE_PROMPT,
    max_calls=250,
    max_tokens=4096,
    memory=memory,
    agent_name="my-session",
    tools=[StopTool(), web_search, write_file]
)

result = agent.run("Make me an agent that ...")
print(agent.memory._messages)