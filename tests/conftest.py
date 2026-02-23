from pathlib import Path
import sys
from typing import Callable, List, Optional

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env before any CAL imports to ensure Maxim SDK gets credentials
load_dotenv(Path(__file__).parent / ".env")

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.llm import LLM
from CAL.logger import Logger
from CAL.message import Message, MessageRole
from CAL.tool import Tool


class QueueLLM(LLM):
    def __init__(self, responses: List[Message], max_tokens: int = 128):
        super().__init__(max_tokens=max_tokens, name="queue-llm", provider="test")
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        self.calls.append({
            "system_prompt": system_prompt,
            "history": list(conversation_history),
            "tools": list(tools or []),
        })
        if not self._responses:
            raise RuntimeError("QueueLLM has no queued responses")
        return self._responses.pop(0)


class FakeTool(Tool):
    def __init__(self, name: str, response_text: str = "ok"):
        async def _fn(text: str = response_text):
            return {"content": [{"type": "text", "text": text}], "metadata": {}}
        super().__init__(_fn)
        self.name = name
        self.response_text = response_text

    async def execute(self, **kwargs) -> ToolResultBlock:
        tool_use_id = kwargs.pop("tool_use_id", "stub_tool_use_id")
        text = kwargs.get("text", self.response_text)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=[TextBlock(text=text)],
            is_error=False,
            name=self.name,
        )


class FakeLogger(Logger):
    def __init__(self):
        self.events = []

    def log_llm_response(self, message: Message, iteration: int, model: Optional[str] = None, provider: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None) -> None:
        self.events.append(("llm", message, iteration, model, provider))

    def log_tool_response(self, tool_use: ToolUseBlock, result: ToolResultBlock, start_time: Optional[int] = None, end_time: Optional[int] = None) -> None:
        self.events.append(("tool", tool_use, result))

    def start_trace(self, name: str, user_prompt: str) -> Optional[str]:
        self.events.append(("start_trace", name, user_prompt))
        return "trace-id"

    def end_trace(self, output: str, metadata: dict) -> None:
        self.events.append(("end_trace", output, metadata))

    def log_metadata(self, metadata: dict) -> None:
        self.events.append(("metadata", metadata))

    def flush(self, timeout_millis: int = 5000) -> None:
        self.events.append(("flush", timeout_millis))

    def shutdown(self) -> None:
        self.events.append(("shutdown",))

    def create_child_logger(self, name: str, agent_name: str = None) -> "FakeLogger":
        self.events.append(("create_child_logger", name, agent_name))
        return FakeLogger()

    def end_child(self) -> None:
        self.events.append(("end_child",))


class FakeLLM(LLM):
    """Minimal LLM for testing that returns a simple summary."""
    def __init__(self):
        super().__init__(max_tokens=128, name="fake-llm", provider="test")
        self.calls = []

    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        self.calls.append({
            "system_prompt": system_prompt,
            "history": list(conversation_history),
            "tools": list(tools or []),
        })
        return Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="[Summary]")])


def make_text_message(role: MessageRole, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def make_tool_use_message(name: str, tool_use_id: str = "tool_use_1", input_data: Optional[dict] = None) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id=tool_use_id, name=name, input=input_data or {})],
    )


@pytest.fixture
def queue_llm_factory() -> Callable[[List[Message]], QueueLLM]:
    def factory(responses: List[Message]) -> QueueLLM:
        return QueueLLM(responses)
    return factory


@pytest.fixture
def fake_logger() -> FakeLogger:
    return FakeLogger()


