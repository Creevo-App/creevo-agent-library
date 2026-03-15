import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from CAL.agent import Agent
from CAL.content_blocks import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from CAL.logger import LangSmithLogger, MaximLogger, SentryLogger
from CAL.message import Message, MessageRole
from conftest import QueueLLM


# ---------------------------------------------------------------------------
# LangSmith Logger — unit tests (mocked SDK)
# ---------------------------------------------------------------------------


def test_langsmith_logger_requires_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(EnvironmentError):
        LangSmithLogger(project_name="test")


def test_langsmith_logger_requires_langsmith_package(monkeypatch):
    """If langsmith is not installed, __init__ raises ImportError."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")

    import importlib
    import CAL.logger as logger_mod

    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise ImportError("No module named 'langsmith'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="langsmith"):
            LangSmithLogger(project_name="test")


@pytest.fixture
def langsmith_logger(monkeypatch):
    """Create a LangSmithLogger with mocked RunTree to avoid network calls."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    logger = LangSmithLogger(project_name="test-project", agent_name="test-agent")

    # Replace RunTree with a MagicMock factory so start_trace/create_child
    # don't make real HTTP requests.
    def make_mock_run(**kwargs):
        run = MagicMock()
        run.id = "mock-run-id"
        run.extra = kwargs.get("extra", {})
        # create_child should also return mocks with the same interface
        run.create_child.side_effect = lambda **kw: make_mock_run(**kw)
        return run

    logger._RunTree = MagicMock(side_effect=lambda **kw: make_mock_run(**kw))
    return logger


def test_langsmith_start_and_end_trace(langsmith_logger):
    trace_id = langsmith_logger.start_trace("agent.run", "hello world")
    assert trace_id is not None
    assert langsmith_logger.root_run is not None
    assert langsmith_logger.user_prompt == "hello world"

    langsmith_logger.end_trace("done", {"status": "ok"})
    assert langsmith_logger.root_run is None


def test_langsmith_log_metadata(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    langsmith_logger.log_metadata({"agent_name": "new-agent", "system_prompt": "be helpful"})
    assert langsmith_logger.agent_name == "new-agent"
    assert langsmith_logger.system_prompt == "be helpful"

    langsmith_logger.end_trace("done", {})


def test_langsmith_log_llm_response(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    # Should not raise
    langsmith_logger.log_llm_response(
        message, iteration=0, model="gemini-2.0-flash", provider="google",
        start_time=1000000000, end_time=2000000000,
    )

    langsmith_logger.end_trace("done", {})


def test_langsmith_log_llm_response_with_thinking(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ThinkingBlock(text="let me think..."),
            TextBlock(text="the answer is 42"),
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    langsmith_logger.log_llm_response(message, iteration=0)
    langsmith_logger.end_trace("done", {})


def test_langsmith_log_tool_response(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    tool_use = ToolUseBlock(id="tu-1", name="search", input={"query": "weather"})
    tool_result = ToolResultBlock(tool_use_id="tu-1", content="sunny", name="search")

    langsmith_logger.log_tool_response(
        tool_use, tool_result,
        start_time=1000000000, end_time=2000000000,
    )

    langsmith_logger.end_trace("done", {})


def test_langsmith_log_tool_error(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    tool_use = ToolUseBlock(id="tu-2", name="broken", input={})
    tool_result = ToolResultBlock(
        tool_use_id="tu-2", content="something went wrong",
        is_error=True, name="broken",
    )

    langsmith_logger.log_tool_response(tool_use, tool_result)
    langsmith_logger.end_trace("done", {})


def test_langsmith_noop_without_trace(langsmith_logger):
    """Logging calls should silently no-op without an active trace."""
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={},
    )
    langsmith_logger.log_llm_response(message, iteration=0)

    tool_use = ToolUseBlock(id="tu-3", name="test", input={})
    tool_result = ToolResultBlock(tool_use_id="tu-3", content="ok", name="test")
    langsmith_logger.log_tool_response(tool_use, tool_result)

    # Should not raise
    langsmith_logger.flush()
    langsmith_logger.shutdown()


def test_langsmith_create_child_logger(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")

    child = langsmith_logger.create_child_logger("code_reviewer", agent_name="reviewer-agent")
    assert child._is_child is True
    assert child.agent_name == "reviewer-agent"
    assert child._parent_run is not None

    # Child should be able to log
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="review complete")],
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )
    child.log_llm_response(message, iteration=0, model="gemini-2.0-flash")

    child.end_child()
    assert child._parent_run is None

    langsmith_logger.end_trace("done", {})


def test_langsmith_child_inherits_agent_name(langsmith_logger):
    langsmith_logger.agent_name = "parent-agent"
    langsmith_logger.start_trace("test", "prompt")

    child = langsmith_logger.create_child_logger("reviewer")
    assert child.agent_name == "parent-agent"

    child.end_child()
    langsmith_logger.end_trace("done", {})


def test_langsmith_child_start_trace_is_noop(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")
    child = langsmith_logger.create_child_logger("sub")

    result = child.start_trace("should-be-ignored", "child prompt")
    assert result is None
    assert child.user_prompt == "child prompt"

    child.end_child()
    langsmith_logger.end_trace("done", {})


def test_langsmith_child_end_trace_is_noop(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")
    child = langsmith_logger.create_child_logger("sub")

    # end_trace on child should be a no-op (not end parent's trace)
    child.end_trace("output", {"key": "value"})
    assert langsmith_logger.root_run is not None  # Parent trace still active

    child.end_child()
    langsmith_logger.end_trace("done", {})


def test_langsmith_end_child_raises_on_parent(langsmith_logger):
    with pytest.raises(RuntimeError, match="not a child"):
        langsmith_logger.end_child()


def test_langsmith_create_child_raises_without_trace(langsmith_logger):
    with pytest.raises(RuntimeError, match="no active trace"):
        langsmith_logger.create_child_logger("sub")


def test_langsmith_shutdown_idempotent(langsmith_logger):
    langsmith_logger.start_trace("test", "prompt")
    langsmith_logger.shutdown()
    langsmith_logger.shutdown()  # Should not raise


def test_langsmith_double_start_trace_warns(langsmith_logger, capsys):
    langsmith_logger.start_trace("first", "prompt1")
    langsmith_logger.start_trace("second", "prompt2")

    _, err = capsys.readouterr()
    assert "Warning" in err
    assert langsmith_logger.root_run is not None

    langsmith_logger.end_trace("done", {})


# ---------------------------------------------------------------------------
# Maxim Logger — unit tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sentry Logger — unit tests (mocked SDK)
# ---------------------------------------------------------------------------


def _make_mock_span():
    """Create a MagicMock that behaves like a sentry_sdk span."""
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    span.span_id = "mock-span-id"
    return span


@pytest.fixture
def sentry_logger():
    """Create a SentryLogger with mocked sentry_sdk to avoid real tracing."""
    mock_sdk = MagicMock()
    created_spans = []

    def mock_start_span(**kwargs):
        span = _make_mock_span()
        span.op = kwargs.get("op", "")
        span.name = kwargs.get("name", "")
        created_spans.append(span)
        return span

    mock_sdk.start_span = MagicMock(side_effect=mock_start_span)
    mock_sdk.is_initialized = MagicMock(return_value=True)

    with patch.dict("sys.modules", {"sentry_sdk": mock_sdk}):
        logger = SentryLogger(agent_name="test-agent")

    logger._sentry_sdk = mock_sdk
    logger._created_spans = created_spans
    return logger


def test_sentry_start_and_end_trace(sentry_logger):
    trace_id = sentry_logger.start_trace("agent.run", "hello world")
    assert trace_id is not None
    assert sentry_logger._root_span is not None
    assert sentry_logger.user_prompt == "hello world"

    sentry_logger._sentry_sdk.start_span.assert_called_once()
    call_kwargs = sentry_logger._sentry_sdk.start_span.call_args[1]
    assert call_kwargs["op"] == "gen_ai.invoke_agent"
    assert "invoke_agent test-agent" in call_kwargs["name"]

    sentry_logger.end_trace("done", {"status": "ok"})
    assert sentry_logger._root_span is None


def test_sentry_log_metadata(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    sentry_logger.log_metadata({"agent_name": "new-agent", "system_prompt": "be helpful"})
    assert sentry_logger.agent_name == "new-agent"
    assert sentry_logger.system_prompt == "be helpful"

    sentry_logger.end_trace("done", {})


def test_sentry_log_llm_response(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    sentry_logger.log_llm_response(
        message, iteration=0, model="gemini-2.0-flash", provider="google",
        start_time=1000000000, end_time=2000000000,
    )

    calls = sentry_logger._sentry_sdk.start_span.call_args_list
    llm_call = calls[-1]
    assert llm_call[1]["op"] == "gen_ai.request"
    assert "gemini-2.0-flash" in llm_call[1]["name"]
    assert llm_call[1]["start_timestamp"] is not None

    # Model should be propagated to the root/parent span (required attribute)
    root_span = sentry_logger._root_span
    root_span.set_data.assert_any_call("gen_ai.request.model", "gemini-2.0-flash")

    # Span should be finished with the caller-provided end_time
    llm_span = sentry_logger._created_spans[-1]
    llm_span.finish.assert_called_once()
    finish_kwargs = llm_span.finish.call_args[1]
    assert finish_kwargs["end_timestamp"] is not None

    sentry_logger.end_trace("done", {})


def test_sentry_log_llm_response_with_thinking(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ThinkingBlock(text="let me think..."),
            TextBlock(text="the answer is 42"),
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    sentry_logger.log_llm_response(message, iteration=0)

    # The LLM span is the last one created (index 1; index 0 is the root span)
    llm_span = sentry_logger._created_spans[-1]
    llm_span.set_data.assert_any_call("gen_ai.response.thinking", "let me think...")

    sentry_logger.end_trace("done", {})


def test_sentry_log_tool_response(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    tool_use = ToolUseBlock(id="tu-1", name="search", input={"query": "weather"})
    tool_result = ToolResultBlock(tool_use_id="tu-1", content="sunny", name="search")

    sentry_logger.log_tool_response(
        tool_use, tool_result,
        start_time=1000000000, end_time=2000000000,
    )

    calls = sentry_logger._sentry_sdk.start_span.call_args_list
    tool_call = calls[-1]
    assert tool_call[1]["op"] == "gen_ai.execute_tool"
    assert "search" in tool_call[1]["name"]
    assert tool_call[1]["start_timestamp"] is not None

    # Span should be finished with the caller-provided end_time
    tool_span = sentry_logger._created_spans[-1]
    tool_span.finish.assert_called_once()
    finish_kwargs = tool_span.finish.call_args[1]
    assert finish_kwargs["end_timestamp"] is not None

    sentry_logger.end_trace("done", {})


def test_sentry_log_tool_error(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    tool_use = ToolUseBlock(id="tu-2", name="broken", input={})
    tool_result = ToolResultBlock(
        tool_use_id="tu-2", content="something went wrong",
        is_error=True, name="broken",
    )

    sentry_logger.log_tool_response(tool_use, tool_result)

    # Error tool spans should be marked with error status
    tool_span = sentry_logger._created_spans[-1]
    tool_span.set_status.assert_called_once_with("internal_error")

    sentry_logger.end_trace("done", {})


def test_sentry_noop_without_trace(sentry_logger):
    """Logging calls should silently no-op without an active trace."""
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={},
    )
    sentry_logger.log_llm_response(message, iteration=0)

    tool_use = ToolUseBlock(id="tu-3", name="test", input={})
    tool_result = ToolResultBlock(tool_use_id="tu-3", content="ok", name="test")
    sentry_logger.log_tool_response(tool_use, tool_result)

    sentry_logger.flush()
    sentry_logger.shutdown()


def test_sentry_create_child_logger(sentry_logger):
    sentry_logger.start_trace("test", "prompt")

    child = sentry_logger.create_child_logger("code_reviewer", agent_name="reviewer-agent")
    assert child._is_child is True
    assert child.agent_name == "reviewer-agent"
    assert child._parent_span is not None

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="review complete")],
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )
    child.log_llm_response(message, iteration=0, model="gemini-2.0-flash")

    child.end_child()
    assert child._parent_span is None

    sentry_logger.end_trace("done", {})


def test_sentry_child_inherits_agent_name(sentry_logger):
    sentry_logger.agent_name = "parent-agent"
    sentry_logger.start_trace("test", "prompt")

    child = sentry_logger.create_child_logger("reviewer")
    assert child.agent_name == "parent-agent"

    child.end_child()
    sentry_logger.end_trace("done", {})


def test_sentry_child_start_trace_is_noop(sentry_logger):
    sentry_logger.start_trace("test", "prompt")
    child = sentry_logger.create_child_logger("sub")

    result = child.start_trace("should-be-ignored", "child prompt")
    assert result is None
    assert child.user_prompt == "child prompt"

    child.end_child()
    sentry_logger.end_trace("done", {})


def test_sentry_child_end_trace_is_noop(sentry_logger):
    sentry_logger.start_trace("test", "prompt")
    child = sentry_logger.create_child_logger("sub")

    child.end_trace("output", {"key": "value"})
    assert sentry_logger._root_span is not None

    child.end_child()
    sentry_logger.end_trace("done", {})


def test_sentry_end_child_raises_on_parent(sentry_logger):
    with pytest.raises(RuntimeError, match="not a child"):
        sentry_logger.end_child()


def test_sentry_create_child_raises_without_trace(sentry_logger):
    with pytest.raises(RuntimeError, match="no active trace"):
        sentry_logger.create_child_logger("sub")


def test_sentry_flush_calls_sdk(sentry_logger):
    sentry_logger.start_trace("test", "prompt")
    sentry_logger.flush()
    sentry_logger._sentry_sdk.flush.assert_called_once()
    sentry_logger.end_trace("done", {})


def test_sentry_shutdown_flushes(sentry_logger):
    sentry_logger.start_trace("test", "prompt")
    sentry_logger.shutdown()
    sentry_logger._sentry_sdk.flush.assert_called_once()


def test_sentry_shutdown_idempotent(sentry_logger):
    sentry_logger.start_trace("test", "prompt")
    sentry_logger.shutdown()
    sentry_logger.shutdown()  # Should not raise


def test_sentry_double_start_trace_warns(sentry_logger, capsys):
    sentry_logger.start_trace("first", "prompt1")
    sentry_logger.start_trace("second", "prompt2")

    _, err = capsys.readouterr()
    assert "Warning" in err
    assert sentry_logger._root_span is not None

    sentry_logger.end_trace("done", {})


def test_sentry_start_trace_warns_if_not_initialized(sentry_logger, capsys):
    """start_trace should return None and warn if sentry_sdk.init() was never called."""
    sentry_logger._sentry_sdk.is_initialized.return_value = False

    result = sentry_logger.start_trace("agent.run", "hello")
    assert result is None
    assert sentry_logger._root_span is None

    _, err = capsys.readouterr()
    assert "not initialized" in err


def test_sentry_requires_sentry_sdk_package():
    """If sentry-sdk is not installed, __init__ raises ImportError."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "sentry_sdk" or name.startswith("sentry_sdk."):
            raise ImportError("No module named 'sentry_sdk'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="sentry-sdk"):
            SentryLogger(agent_name="test")


@pytest.mark.integration
def test_sentry_logger_integration():
    """Integration test: sends real spans to Sentry. Requires SENTRY_DSN."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        pytest.skip("SENTRY_DSN must be set for integration test")

    import sentry_sdk
    sentry_sdk.init(dsn=dsn, traces_sample_rate=1.0)

    logger = SentryLogger(agent_name="test-agent")

    trace_id = logger.start_trace("integration-test", "test prompt")
    assert trace_id is not None

    message = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hello")],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    logger.log_llm_response(message, iteration=0, model="gemini-2.0-flash", provider="google")

    tool_use = ToolUseBlock(id="tu-int", name="calculator", input={"expression": "2+2"})
    tool_result = ToolResultBlock(tool_use_id="tu-int", content="4", name="calculator")
    logger.log_tool_response(tool_use, tool_result)

    child = logger.create_child_logger("math_expert", agent_name="math-sub")
    child_msg = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="confirmed: 4")],
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )
    child.log_llm_response(child_msg, iteration=0, model="gemini-2.0-flash")
    child.end_child()

    logger.end_trace("The answer is 4", {"status": "completed_success", "total_iterations": 1})
    logger.shutdown()


@pytest.mark.integration
def test_langsmith_logger_integration():
    """Integration test: sends real traces to LangSmith. Requires LANGSMITH_API_KEY."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY must be set for integration test")

    logger = LangSmithLogger(project_name="cal-integration-test", agent_name="test-agent")

    trace_id = logger.start_trace("integration-test", "what is 2+2?")
    assert trace_id is not None

    # Log an LLM response
    llm_msg = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="2+2 is 4")],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    logger.log_llm_response(
        llm_msg, iteration=0, model="gemini-2.0-flash", provider="google",
    )

    # Log a tool call
    tool_use = ToolUseBlock(id="tu-int", name="calculator", input={"expression": "2+2"})
    tool_result = ToolResultBlock(tool_use_id="tu-int", content="4", name="calculator")
    logger.log_tool_response(tool_use, tool_result)

    # Create and use a child logger
    child = logger.create_child_logger("math_expert", agent_name="math-sub")
    child_msg = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="confirmed: 4")],
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )
    child.log_llm_response(child_msg, iteration=0, model="gemini-2.0-flash")
    child.end_child()

    logger.end_trace("The answer is 4", {"status": "completed_success", "total_iterations": 1})
    logger.shutdown()
