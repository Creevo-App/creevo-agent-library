import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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

    logger = MaximLogger(agent_name="session")
    assert logger.logger_instance is None
    assert logger.start_trace("run", "prompt") is None

    tool_use = ToolUseBlock(id="tool-use-1", name="tool", input={})
    tool_result = ToolResultBlock(tool_use_id="tool-use-1", content="ok", name="tool")

    logger.log_tool_response(tool_use, tool_result)
    logger.flush()
    logger.shutdown()


@pytest.mark.integration
def test_maxim_logger_records_trace_data(capsys):
    api_key = os.getenv("MAXIM_API_KEY")
    log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")
    if not api_key or not log_repo_id:
        pytest.skip("MAXIM_API_KEY and MAXIM_LOG_REPO_ID must be set")

    logger = MaximLogger(agent_name="test-session")
    assert logger.logger_instance is not None

    trace_id = logger.start_trace("test-run", "test prompt")
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

    assert logger.root_trace is not None
    trace_data = logger.root_trace.data()
    assert isinstance(trace_data, dict)
    assert trace_data.get("name") == "test-run"

    logger.end_trace(output="done", metadata={"status": "ok"})
    assert logger.root_trace is None
    logger.shutdown()
    _, err = capsys.readouterr()
    assert "MaximSDK" not in err


# TODO: Add LangSmith trace verification once test repo is available.
