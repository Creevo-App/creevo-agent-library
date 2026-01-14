import os

import pytest

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.llm import AnthropicVertexLLM, GeminiLLM
from CAL.message import Message, MessageRole


class DummyFunctionCall:
    def __init__(self, name: str, args: dict, call_id: str = "call-id"):
        self.id = call_id
        self.name = name
        self.args = args


class DummyPart:
    def __init__(self, text: str = None, function_call: DummyFunctionCall = None):
        self.text = text
        self.function_call = function_call


class DummyContent:
    def __init__(self, parts):
        self.parts = parts


class DummyCandidate:
    def __init__(self, parts, finish_reason: str = "STOP"):
        self.content = DummyContent(parts)
        self.finish_reason = finish_reason


class DummyUsageMetadata:
    def __init__(self, prompt: int, completion: int, total: int):
        self.prompt_token_count = prompt
        self.candidates_token_count = completion
        self.total_token_count = total


class DummyResponse:
    def __init__(self, candidates, usage_metadata=None, model_version: str = "v1"):
        self.candidates = candidates
        self.usage_metadata = usage_metadata
        self.model_version = model_version


class FakeModels:
    def __init__(self, response: DummyResponse):
        self.calls = []
        self._response = response

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


class FakeClient:
    def __init__(self, response: DummyResponse):
        self.models = FakeModels(response)


def test_anthropic_vertex_llm_returns_stub_message():
    llm = AnthropicVertexLLM(api_key="key", model="model", max_tokens=8)
    message = llm.generate_content("system", [], tools=None)

    assert message.role == MessageRole.ASSISTANT
    assert message.content[0].text == "Hi"


def test_gemini_llm_formats_history_and_extracts_usage():
    response = DummyResponse(
        candidates=[
            DummyCandidate(
                parts=[
                    DummyPart(text="answer"),
                    DummyPart(function_call=DummyFunctionCall(name="tool", args={"x": 1})),
                ],
                finish_reason="MAX_TOKENS",
            )
        ],
        usage_metadata=DummyUsageMetadata(prompt=3, completion=5, total=8),
    )
    llm = GeminiLLM(api_key="key", model="model", max_tokens=10)
    llm.client = FakeClient(response)

    history = [
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.USER, content=[TextBlock(text="there")]),
        Message(role=MessageRole.TOOL_RESPONSE, content=[ToolResultBlock(tool_use_id="1", content="ok", name="tool")]),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="prior")]),
    ]

    message = llm.generate_content("system", history, tools=None)

    call = llm.client.models.calls[0]
    contents = call["contents"]
    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert len(contents[0]["parts"]) == 3
    assert contents[1]["role"] == "model"

    assert message.metadata["finish_reason"] == "MAX_TOKENS"
    assert message.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}

    tool_blocks = [block for block in message.content if isinstance(block, ToolUseBlock)]
    assert tool_blocks[0].name == "tool"
    assert tool_blocks[0].input == {"x": 1}


@pytest.mark.integration
def test_gemini_llm_live_request():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    assert api_key, "GEMINI_API_KEY must be set for live Gemini tests"
    assert model, "GEMINI_MODEL must be set for live Gemini tests"

    llm = GeminiLLM(api_key=api_key, model=model, max_tokens=64)
    history = [Message(role=MessageRole.USER, content="Hello")]
    message = llm.generate_content("system", history, tools=None)

    assert message.role == MessageRole.ASSISTANT
    assert message.content
