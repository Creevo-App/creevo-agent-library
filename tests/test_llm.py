import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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


class DummyAnthropicTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class DummyAnthropicToolUseBlock:
    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class DummyAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class DummyAnthropicResponse:
    def __init__(self, content, usage=None, stop_reason="end_turn"):
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


class FakeAnthropicMessages:
    def __init__(self, response: DummyAnthropicResponse):
        self.calls = []
        self._response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: DummyAnthropicResponse):
        self.messages = FakeAnthropicMessages(response)


def test_anthropic_vertex_llm_formats_history_and_extracts_usage():
    response = DummyAnthropicResponse(
        content=[
            DummyAnthropicTextBlock(text="Hello!"),
            DummyAnthropicToolUseBlock(id="tool-1", name="my_tool", input={"arg": "value"}),
        ],
        usage=DummyAnthropicUsage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )
    llm = AnthropicVertexLLM(project_id="test-project", region="global", model="claude-opus-4-5@20251101", max_tokens=8)
    llm.client = FakeAnthropicClient(response)

    history = [
        Message(role=MessageRole.USER, content="hello"),
        Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="hi")]),
        Message(role=MessageRole.USER, content="request tool"),
    ]

    message = llm.generate_content("system prompt", history, tools=None)

    # Verify the call was made with correct parameters
    call = llm.client.messages.calls[0]
    assert call["model"] == "claude-opus-4-5@20251101"
    assert call["system"] == "system prompt"
    assert call["max_tokens"] == 8
    assert len(call["messages"]) == 3

    # Verify response parsing
    assert message.role == MessageRole.ASSISTANT
    assert len(message.content) == 2
    assert message.content[0].text == "Hello!"
    assert isinstance(message.content[1], ToolUseBlock)
    assert message.content[1].name == "my_tool"
    assert message.content[1].input == {"arg": "value"}

    # Verify usage extraction
    assert message.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert message.metadata["finish_reason"] == "tool_use"


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
    import time
    from google.genai.errors import ClientError

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        pytest.skip("GEMINI_API_KEY and GEMINI_MODEL must be set")

    # Wait 10 seconds to respect rate limits between API calls
    time.sleep(10)

    llm = GeminiLLM(api_key=api_key, model=model, max_tokens=64)
    history = [Message(role=MessageRole.USER, content="Hello")]

    try:
        message = llm.generate_content("system", history, tools=None)
    except ClientError as e:
        if e.code == 429:
            pytest.skip("Gemini API rate limit exceeded")
        raise

    assert message.role == MessageRole.ASSISTANT
    assert message.content


def _extract_roles(contents: list) -> list:
    """Extract roles from Gemini formatted contents."""
    return [c["role"] for c in contents]


def _is_strictly_alternating(roles: list) -> bool:
    """Check if roles strictly alternate (no consecutive same roles)."""
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            return False
    return True


class TestGeminiFormattingAlternation:
    """Tests verifying Gemini formatting produces strictly alternating roles."""

    def test_simple_user_assistant_alternates(self):
        response = DummyResponse(
            candidates=[DummyCandidate(parts=[DummyPart(text="ok")])],
            usage_metadata=DummyUsageMetadata(1, 1, 2),
        )
        llm = GeminiLLM(api_key="key", model="model", max_tokens=10)
        llm.client = FakeClient(response)

        history = [
            Message(role=MessageRole.USER, content="hello"),
            Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="hi")]),
            Message(role=MessageRole.USER, content="bye"),
        ]
        llm.generate_content("system", history, tools=None)

        contents = llm.client.models.calls[0]["contents"]
        roles = _extract_roles(contents)
        assert _is_strictly_alternating(roles)
        assert roles == ["user", "model", "user"]

    def test_consecutive_user_messages_get_merged(self):
        response = DummyResponse(
            candidates=[DummyCandidate(parts=[DummyPart(text="ok")])],
            usage_metadata=DummyUsageMetadata(1, 1, 2),
        )
        llm = GeminiLLM(api_key="key", model="model", max_tokens=10)
        llm.client = FakeClient(response)

        history = [
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.USER, content="second"),
            Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="reply")]),
        ]
        llm.generate_content("system", history, tools=None)

        contents = llm.client.models.calls[0]["contents"]
        roles = _extract_roles(contents)
        assert _is_strictly_alternating(roles)
        # Two user messages should be merged into one
        assert roles == ["user", "model"]
        # Verify parts are combined
        assert len(contents[0]["parts"]) == 2

    def test_tool_response_maps_to_user_role(self):
        response = DummyResponse(
            candidates=[DummyCandidate(parts=[DummyPart(text="ok")])],
            usage_metadata=DummyUsageMetadata(1, 1, 2),
        )
        llm = GeminiLLM(api_key="key", model="model", max_tokens=10)
        llm.client = FakeClient(response)

        history = [
            Message(role=MessageRole.USER, content="request"),
            Message(role=MessageRole.ASSISTANT, content=[ToolUseBlock(id="t1", name="fn", input={})]),
            Message(role=MessageRole.TOOL_RESPONSE, content=[ToolResultBlock(tool_use_id="t1", content="result", name="fn")]),
        ]
        llm.generate_content("system", history, tools=None)

        contents = llm.client.models.calls[0]["contents"]
        roles = _extract_roles(contents)
        assert _is_strictly_alternating(roles)
        assert roles == ["user", "model", "user"]

    def test_consecutive_tool_responses_get_merged(self):
        response = DummyResponse(
            candidates=[DummyCandidate(parts=[DummyPart(text="ok")])],
            usage_metadata=DummyUsageMetadata(1, 1, 2),
        )
        llm = GeminiLLM(api_key="key", model="model", max_tokens=10)
        llm.client = FakeClient(response)

        # Two tool calls followed by two tool responses
        history = [
            Message(role=MessageRole.USER, content="request"),
            Message(role=MessageRole.ASSISTANT, content=[
                ToolUseBlock(id="t1", name="fn1", input={}),
                ToolUseBlock(id="t2", name="fn2", input={}),
            ]),
            Message(role=MessageRole.TOOL_RESPONSE, content=[ToolResultBlock(tool_use_id="t1", content="r1", name="fn1")]),
            Message(role=MessageRole.TOOL_RESPONSE, content=[ToolResultBlock(tool_use_id="t2", content="r2", name="fn2")]),
        ]
        llm.generate_content("system", history, tools=None)

        contents = llm.client.models.calls[0]["contents"]
        roles = _extract_roles(contents)
        assert _is_strictly_alternating(roles)
        # Tool responses should be merged into single user turn
        assert roles == ["user", "model", "user"]
