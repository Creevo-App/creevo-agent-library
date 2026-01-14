import os

import pytest

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.logger import LangSmithLogger, MaximLogger
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from conftest import QueueLLM


def test_langsmith_logger_requires_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(EnvironmentError):
        LangSmithLogger(project_name="test")


def test_maxim_logger_noop_without_env(monkeypatch):
    monkeypatch.delenv("MAXIM_API_KEY", raising=False)
    monkeypatch.delenv("MAXIM_LOG_REPO_ID", raising=False)

    logger = MaximLogger(session_id="session")
    assert logger.logger_instance is None
    assert logger.start_trace("run", "prompt") is None

    tool_use = ToolUseBlock(id="tool-use-1", name="tool", input={})
    tool_result = ToolResultBlock(tool_use_id="tool-use-1", content="ok", name="tool")

    logger.log_tool_response(tool_use, tool_result)
    logger.flush()
    logger.shutdown()


def test_maxim_logger_records_trace_data():
    api_key = os.getenv("MAXIM_API_KEY")
    log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")
    assert api_key, "MAXIM_API_KEY must be set for Maxim tests"
    assert log_repo_id, "MAXIM_LOG_REPO_ID must be set for Maxim tests"

    logger = MaximLogger(session_id="session")
    trace_id = logger.start_trace("run", "prompt")
    assert trace_id

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    logger.log_llm_response(message, iteration=0, model="model", provider="provider")

    tool_use = ToolUseBlock(id="tool-use-2", name="tool", input={"value": "x"})
    tool_result = ToolResultBlock(tool_use_id="tool-use-2", content="ok", name="tool")
    logger.log_tool_response(tool_use, tool_result)

    trace_data = logger.root_trace.data()
    assert isinstance(trace_data, dict)
    assert "User Prompt:" in str(trace_data)

    logger.end_trace(output="done", metadata={"status": "ok"})


def test_maxim_logger_records_trace_from_agent():
    api_key = os.getenv("MAXIM_API_KEY")
    log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")
    assert api_key, "MAXIM_API_KEY must be set for Maxim tests"
    assert log_repo_id, "MAXIM_LOG_REPO_ID must be set for Maxim tests"

    class RecordingMaximLogger(MaximLogger):
        def __init__(self, session_id: str):
            super().__init__(session_id=session_id)
            self.trace_data = None

        def end_trace(self, output: str, metadata: dict) -> None:
            if self.root_trace:
                self.trace_data = self.root_trace.data()
            super().end_trace(output, metadata)

    logger = RecordingMaximLogger(session_id="session")
    llm = QueueLLM([Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="done")])])
    memory = FullCompressionMemory()
    agent = Agent(
        llm=llm,
        system_prompt="system",
        max_calls=1,
        max_tokens=10,
        memory=memory,
        session_id="session",
        tools=[],
        logger=logger,
    )

    result = agent.run("prompt")

    assert result.content[0].text == "done"
    assert logger.trace_data
    assert "User Prompt:" in str(logger.trace_data)


# TODO: Add LangSmith trace verification once test repo is available.
