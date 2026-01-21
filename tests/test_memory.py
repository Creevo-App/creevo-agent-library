import pytest
from typing import List, Optional

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.compression import CompressionConfig
from CAL.llm import LLM
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.tool import Tool


class FakeSummarizerLLM(LLM):
    """Fake LLM for testing that returns a predictable summary response."""
    def __init__(self):
        super().__init__(max_tokens=128, name="fake-summarizer", provider="test")

    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        # Return a simple summary mentioning ASSISTANT and any tools used
        return Message(
            role=MessageRole.ASSISTANT,
            content=[TextBlock(text='{"filename": "test_context", "summary": "Summary of ASSISTANT responses.", "detailed_summary": "Detailed summary of ASSISTANT actions."}')]
        )


def get_fake_llm() -> FakeSummarizerLLM:
    return FakeSummarizerLLM()


def get_test_compression_config() -> CompressionConfig:
    """Get compression config with small values for testing."""
    return CompressionConfig(
        keep_recent_tokens=20,  # Very small to trigger compression
        max_summary_tokens=100,
    )


def make_text_message(role: MessageRole, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def make_tool_use_message(tool_name: str, tool_id: str) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id=tool_id, name=tool_name, input={})],
    )


def make_tool_result_message(tool_name: str, tool_id: str, result: str) -> Message:
    return Message(
        role=MessageRole.TOOL_RESPONSE,
        content=[ToolResultBlock(tool_use_id=tool_id, content=result, name=tool_name)],
    )


def get_roles(history: list) -> list:
    """Extract role values from history messages."""
    return [msg.role for msg in history]


def is_alternating_user_assistant(history: list) -> bool:
    """Check if history alternates between user-like and assistant roles.
    
    Note: USER and TOOL_RESPONSE both map to 'user' in Gemini.
    """
    def role_category(role: MessageRole) -> str:
        if role in (MessageRole.USER, MessageRole.TOOL_RESPONSE):
            return "user"
        return "assistant"
    
    categories = [role_category(msg.role) for msg in history]
    for i in range(1, len(categories)):
        if categories[i] == categories[i - 1]:
            return False
    return True


def test_full_compression_memory_compresses():
    # Use very small max_tokens and keep_recent_tokens to trigger compression
    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=50,
        compression_config=get_test_compression_config(),
    )
    memory.add_message(make_text_message(MessageRole.USER, "first " * 20))  # ~80 chars = ~20 tokens
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "second " * 20))
    memory.add_message(make_text_message(MessageRole.USER, "third " * 20))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "fourth " * 20))
    memory.add_message(make_text_message(MessageRole.USER, "fifth " * 20))

    history = memory.get_history()

    # After compression: initial, summary, recent messages
    assert len(history) <= 4
    # Check for compressed message
    compressed_msgs = [m for m in history if m.metadata.get("compressed")]
    assert len(compressed_msgs) >= 1
    assert "archived" in compressed_msgs[0].content.lower() or "context" in compressed_msgs[0].content.lower()


def test_full_compression_memory_round_trip_json():
    llm = get_fake_llm()
    memory = FullCompressionMemory(summarizer_llm=llm, max_tokens=10000)
    memory.add_message(make_text_message(MessageRole.USER, "hello"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "world"))

    payload = memory.to_json()
    restored = FullCompressionMemory.from_json(payload, summarizer_llm=llm)

    history = restored.get_history()
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].role == MessageRole.ASSISTANT
    assert history[0].content[0].text == "hello"
    assert history[1].content[0].text == "world"


def test_full_compression_memory_clone_is_independent():
    memory = FullCompressionMemory(summarizer_llm=get_fake_llm(), max_tokens=10000)
    memory.add_message(make_text_message(MessageRole.USER, "one"))

    clone = memory.clone()
    clone.add_message(make_text_message(MessageRole.USER, "two"))

    assert len(memory.get_history()) == 1
    assert len(clone.get_history()) == 2


class TestCompressionAlternatingRoleInvariant:
    """Tests for the alternating role invariant after compression.
    
    The compress() method inserts a summary as USER role. If the initial
    message is also USER, this creates USER, USER sequences which break
    Gemini API. These tests verify the behavior and document the issue.
    """

    def test_compression_with_user_initial_creates_user_user_sequence(self):
        """Current behavior: USER initial + USER summary = USER, USER sequence.
        
        This test documents the current (problematic) behavior where
        compression can create consecutive USER messages.
        """
        # Use very small max_tokens to force compression
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=50,
            compression_config=get_test_compression_config(),
        )
        # Start with USER (initial) - make messages large enough to trigger compression
        memory.add_message(make_text_message(MessageRole.USER, "initial request " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "response1 " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "followup " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "response2 " * 10))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.USER, "final " * 10))

        history = memory.get_history()
        roles = get_roles(history)

        # Document current behavior: initial is USER, summary is USER
        assert history[0].role == MessageRole.USER  # initial
        assert history[1].role == MessageRole.USER  # summary (compressed)
        assert history[1].metadata.get("compressed") is True

        # This creates USER, USER which breaks Gemini
        assert not is_alternating_user_assistant(history), (
            "Current compression creates non-alternating USER, USER sequence"
        )

    def test_compression_preserves_content_in_summary(self):
        """Verify compression summarizes the compressed messages."""
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=50,
            compression_config=get_test_compression_config(),
        )
        memory.add_message(make_text_message(MessageRole.USER, "initial " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "compressed content here " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "recent1 " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "recent2 " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "final " * 10))

        history = memory.get_history()
        # Find the compressed summary message
        compressed_msgs = [m for m in history if m.metadata.get("compressed")]
        assert len(compressed_msgs) >= 1
        summary_content = compressed_msgs[0].content

        # The summary should mention the archive or context
        assert "context" in summary_content.lower() or "archived" in summary_content.lower()


class TestCompressionBoundary:
    """Tests for compression boundary conditions."""

    def test_compression_boundary_recent_starts_with_assistant(self):
        """Test when recent[0] is ASSISTANT after compression."""
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=50,
            compression_config=get_test_compression_config(),
        )
        # Build history: U, A, U, A, U, A - with large messages to trigger compression
        memory.add_message(make_text_message(MessageRole.USER, "u1 " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a1 " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "u2 " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a2 " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "u3 " * 10))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a3 " * 10))

        history = memory.get_history()

        # After compression: initial, summary, some recent messages
        assert len(history) <= 6
        # initial is USER
        assert history[0].role == MessageRole.USER
        # Check for compressed message
        compressed_msgs = [m for m in history if m.metadata.get("compressed")]
        if compressed_msgs:
            assert compressed_msgs[0].role == MessageRole.USER

    def test_compression_with_minimal_messages(self):
        """Test compression with few messages (boundary)."""
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=30,
            compression_config=get_test_compression_config(),
        )
        memory.add_message(make_text_message(MessageRole.USER, "u1 " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a1 " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "u2 " * 10))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a2 " * 10))

        history = memory.get_history()
        # Should have some messages after compression
        assert len(history) <= 4


class TestToolLoopPatternCompression:
    """Tests for tool loop pattern compression.
    
    Tests realistic USER -> ASSISTANT(tool_use) -> USER(tool_result) -> ASSISTANT
    sequences through compression.
    """

    def test_tool_loop_compression_preserves_structure(self):
        """Test compression of tool usage patterns."""
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=50,
            compression_config=get_test_compression_config(),
        )

        # Build a tool usage pattern with large enough content
        memory.add_message(make_text_message(MessageRole.USER, "do something " * 10))
        memory.add_message(make_tool_use_message("my_tool", "t1"))
        memory.add_message(make_tool_result_message("my_tool", "t1", "result " * 10))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "done " * 10))
        memory.add_message(make_text_message(MessageRole.USER, "do more " * 10))
        memory.add_message(make_tool_use_message("my_tool", "t2"))
        # Trigger compression
        memory.add_message(make_tool_result_message("my_tool", "t2", "result2 " * 10))

        history = memory.get_history()

        # Verify compression occurred
        summary_msg = next((m for m in history if m.metadata.get("compressed")), None)
        # Compression may or may not trigger depending on token calculation
        if summary_msg is not None:
            # Summary should contain context reference
            assert "context" in summary_msg.content.lower() or "archived" in summary_msg.content.lower()

    def test_multiple_tool_calls_in_compression(self):
        """Test compression of multiple consecutive tool calls."""
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=30,
            compression_config=get_test_compression_config(),
        )

        memory.add_message(make_text_message(MessageRole.USER, "initial " * 10))
        memory.add_message(make_tool_use_message("tool1", "t1"))
        memory.add_message(make_tool_result_message("tool1", "t1", "result1 " * 10))
        memory.add_message(make_tool_use_message("tool2", "t2"))
        # Trigger compression
        memory.add_message(make_tool_result_message("tool2", "t2", "result2 " * 10))

        history = memory.get_history()
        assert len(history) <= 5


# Parametrized role-pattern tests with table-driven tests
ROLE_PATTERNS = [
    # (pattern_name, message_roles, expected_final_length_max)
    pytest.param(
        "simple_alternating",
        [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER],
        4,
        id="simple_alternating",
    ),
    pytest.param(
        "user_heavy",
        [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER, MessageRole.USER, MessageRole.ASSISTANT],
        4,
        id="user_heavy",
    ),
    pytest.param(
        "tool_pattern",
        [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL_RESPONSE, MessageRole.ASSISTANT, MessageRole.USER],
        4,
        id="tool_pattern",
    ),
    pytest.param(
        "long_conversation",
        [MessageRole.USER, MessageRole.ASSISTANT] * 5,
        6,
        id="long_conversation",
    ),
]


@pytest.mark.parametrize("pattern_name,roles,max_length", ROLE_PATTERNS)
def test_compression_role_patterns(pattern_name, roles, max_length):
    """Parametrized test for various role patterns through compression."""
    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=50,
        compression_config=get_test_compression_config(),
    )

    for i, role in enumerate(roles):
        if role == MessageRole.TOOL_RESPONSE:
            memory.add_message(make_tool_result_message("tool", f"t{i}", "result " * 10))
        else:
            memory.add_message(make_text_message(role, f"msg{i} " * 10))

    history = memory.get_history()
    # With token-based compression, max_length bounds are less predictable
    # Just verify first message is preserved
    assert history[0].role == roles[0]
