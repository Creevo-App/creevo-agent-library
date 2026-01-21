import pytest

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole


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
    memory = FullCompressionMemory(max_items=4)
    memory.add_message(make_text_message(MessageRole.USER, "first"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "second"))
    memory.add_message(make_text_message(MessageRole.USER, "third"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "fourth"))
    memory.add_message(make_text_message(MessageRole.USER, "fifth"))

    history = memory.get_history()

    assert len(history) == 4
    assert history[1].metadata.get("compressed") is True
    assert "[Summary of 2 previous turns:]" in history[1].content


def test_full_compression_memory_round_trip_json():
    memory = FullCompressionMemory(max_items=10)
    memory.add_message(make_text_message(MessageRole.USER, "hello"))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "world"))

    payload = memory.to_json()
    restored = FullCompressionMemory.from_json(payload)

    history = restored.get_history()
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].role == MessageRole.ASSISTANT
    assert history[0].content[0].text == "hello"
    assert history[1].content[0].text == "world"


def test_full_compression_memory_clone_is_independent():
    memory = FullCompressionMemory(max_items=10)
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
        memory = FullCompressionMemory(max_items=4)
        # Start with USER (initial)
        memory.add_message(make_text_message(MessageRole.USER, "initial request"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "response1"))
        memory.add_message(make_text_message(MessageRole.USER, "followup"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "response2"))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.USER, "final"))

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
        memory = FullCompressionMemory(max_items=4)
        memory.add_message(make_text_message(MessageRole.USER, "initial"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "compressed content here"))
        memory.add_message(make_text_message(MessageRole.USER, "recent1"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "recent2"))
        memory.add_message(make_text_message(MessageRole.USER, "final"))

        history = memory.get_history()
        summary_content = history[1].content

        assert "Summary" in summary_content
        assert "ASSISTANT" in summary_content


class TestCompressionBoundary:
    """Tests for compression boundary conditions."""

    def test_compression_boundary_recent_starts_with_assistant(self):
        """Test when recent[0] is ASSISTANT after compression."""
        memory = FullCompressionMemory(max_items=4)
        # Build history: U, A, U, A, U, A
        memory.add_message(make_text_message(MessageRole.USER, "u1"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a1"))
        memory.add_message(make_text_message(MessageRole.USER, "u2"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a2"))
        memory.add_message(make_text_message(MessageRole.USER, "u3"))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a3"))

        history = memory.get_history()

        # Verify structure: [initial, summary, recent[0], recent[1]]
        assert len(history) == 4
        # initial is USER
        assert history[0].role == MessageRole.USER
        # summary is USER (compressed)
        assert history[1].role == MessageRole.USER
        assert history[1].metadata.get("compressed") is True

    def test_compression_with_minimal_messages(self):
        """Test compression with exactly 4 messages (boundary)."""
        memory = FullCompressionMemory(max_items=3)
        memory.add_message(make_text_message(MessageRole.USER, "u1"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a1"))
        memory.add_message(make_text_message(MessageRole.USER, "u2"))
        # Trigger compression
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "a2"))

        history = memory.get_history()
        # Should have: initial, summary (if any), recent
        assert len(history) <= 4


class TestToolLoopPatternCompression:
    """Tests for tool loop pattern compression.
    
    Tests realistic USER -> ASSISTANT(tool_use) -> USER(tool_result) -> ASSISTANT
    sequences through compression.
    """

    def test_tool_loop_compression_preserves_structure(self):
        """Test compression of tool usage patterns."""
        memory = FullCompressionMemory(max_items=6)

        # Build a tool usage pattern
        memory.add_message(make_text_message(MessageRole.USER, "do something"))
        memory.add_message(make_tool_use_message("my_tool", "t1"))
        memory.add_message(make_tool_result_message("my_tool", "t1", "result"))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "done"))
        memory.add_message(make_text_message(MessageRole.USER, "do more"))
        memory.add_message(make_tool_use_message("my_tool", "t2"))
        # Trigger compression
        memory.add_message(make_tool_result_message("my_tool", "t2", "result2"))

        history = memory.get_history()

        # Verify summary mentions tool usage
        summary_msg = next((m for m in history if m.metadata.get("compressed")), None)
        assert summary_msg is not None
        assert "my_tool" in summary_msg.content

    def test_multiple_tool_calls_in_compression(self):
        """Test compression of multiple consecutive tool calls."""
        memory = FullCompressionMemory(max_items=4)

        memory.add_message(make_text_message(MessageRole.USER, "initial"))
        memory.add_message(make_tool_use_message("tool1", "t1"))
        memory.add_message(make_tool_result_message("tool1", "t1", "result1"))
        memory.add_message(make_tool_use_message("tool2", "t2"))
        # Trigger compression
        memory.add_message(make_tool_result_message("tool2", "t2", "result2"))

        history = memory.get_history()
        assert len(history) <= 4


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
    memory = FullCompressionMemory(max_items=4)

    for i, role in enumerate(roles):
        if role == MessageRole.TOOL_RESPONSE:
            memory.add_message(make_tool_result_message("tool", f"t{i}", "result"))
        else:
            memory.add_message(make_text_message(role, f"msg{i}"))

    history = memory.get_history()
    assert len(history) <= max_length, f"Pattern {pattern_name} exceeded max length"
    # Verify first message is preserved
    assert history[0].role == roles[0]
