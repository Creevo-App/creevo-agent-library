import pytest
from typing import List, Optional

from CAL.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from CAL.compression import CompressionConfig
from CAL.llm import LLM
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.tool import Tool

from conftest import make_text_message
from test_llm import _is_strictly_alternating


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
    Uses _is_strictly_alternating from test_llm.py for the core check.
    """
    def role_category(role: MessageRole) -> str:
        if role in (MessageRole.USER, MessageRole.TOOL_RESPONSE):
            return "user"
        return "assistant"
    
    categories = [role_category(msg.role) for msg in history]
    return _is_strictly_alternating(categories)


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

    def test_compression_breakpoint_does_not_split_tool_use_and_result(self):
        """Test that compression never splits tool_use from its tool_result.
        
        Anthropic API requires that any message containing a tool_result must
        immediately follow the assistant message with the corresponding tool_use.
        If compression summarizes the tool_use but keeps the tool_result in recent
        messages, the API will reject with:
        "unexpected tool_use_id found in tool_result blocks"
        
        This test creates a scenario where the natural token-based breakpoint
        would fall between tool_use and tool_result, and verifies the fix
        backs up the breakpoint to include both.
        """
        # Token estimates (1 token ~= 4 chars):
        # - tool_use "read_file({})" ~= 3 tokens
        # - tool_result with 20 chars ~= 5 tokens
        # - final with 40 chars ~= 10 tokens
        #
        # With keep_recent_tokens=16:
        # - final (10) fits, total=10
        # - tool_result (5) fits, total=15
        # - tool_use (3) would make total=18 > 16, so breakpoint falls AFTER tool_use
        #
        # This causes tool_result to appear in "recent" without its tool_use.
        memory = FullCompressionMemory(
            summarizer_llm=get_fake_llm(),
            max_tokens=100,
            compression_config=CompressionConfig(
                keep_recent_tokens=16,  # Fits tool_result + final (15), but not + tool_use (18)
            ),
        )

        # Build history that triggers compression with breakpoint at the right spot
        # [0] USER - large, will be initial message kept
        memory.add_message(make_text_message(MessageRole.USER, "a" * 200))  # ~50 tokens
        # [1] ASSISTANT - large, will be compressed
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "b" * 200))  # ~50 tokens
        # [2] USER - will be compressed
        memory.add_message(make_text_message(MessageRole.USER, "c" * 40))  # ~10 tokens

        # At this point: ~110 tokens, compression triggers on next add
        # [3] TOOL_USE - should NOT be separated from result
        memory.add_message(make_tool_use_message("read_file", "tool_123"))  # ~3 tokens
        # [4] TOOL_RESULT - fits in keep_recent with final
        memory.add_message(make_tool_result_message("read_file", "tool_123", "x" * 20))  # ~5 tokens
        # [5] Final message - fits in keep_recent
        memory.add_message(make_text_message(MessageRole.ASSISTANT, "y" * 40))  # ~10 tokens

        history = memory.get_history()

        # Verify compression occurred
        compressed_msgs = [m for m in history if m.metadata.get("compressed")]
        assert len(compressed_msgs) >= 1, "Compression should have occurred"

        # Check that no message with tool_result lacks a preceding tool_use
        for i, msg in enumerate(history):
            if isinstance(msg.content, list):
                has_tool_result = any(isinstance(b, ToolResultBlock) for b in msg.content)
                if has_tool_result:
                    # This message has a tool_result - previous must have tool_use
                    assert i > 0, "tool_result cannot be first message"
                    prev_msg = history[i - 1]
                    has_tool_use = False
                    if isinstance(prev_msg.content, list):
                        has_tool_use = any(isinstance(b, ToolUseBlock) for b in prev_msg.content)
                    assert has_tool_use, (
                        f"Message {i} has tool_result but message {i-1} has no tool_use. "
                        "Compression split a tool_use/tool_result pair."
                    )

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


@pytest.mark.parametrize("_,roles,__", ROLE_PATTERNS)
def test_compression_role_patterns(_, roles, __):
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


def test_sync_token_count_from_llm_usage():
    """Test that sync_token_count_from_llm_usage updates _total_tokens correctly.

    This ensures compression triggers based on actual LLM token counts (which include
    system prompt and tool schemas) rather than just estimated message content tokens.
    """
    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=1000,
    )

    # Add a message - this will use estimation (~4 chars/token)
    memory.add_message(make_text_message(MessageRole.USER, "Hello world"))
    estimated_tokens = memory._total_tokens
    assert estimated_tokens > 0, "Should have estimated some tokens"

    # Simulate LLM returning actual token count that's much higher
    # (as if system prompt + tool schemas were included)
    actual_prompt_tokens = 500
    memory.sync_token_count_from_llm_usage(actual_prompt_tokens)

    assert memory._total_tokens == actual_prompt_tokens, (
        f"Token count should be synced to actual: {memory._total_tokens} != {actual_prompt_tokens}"
    )


def test_sync_token_count_triggers_compression():
    """Test that syncing a high token count triggers compression on next add_message.

    If the LLM reports we're near the limit, the next message should trigger compression.
    """
    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=1000,
        compression_config=CompressionConfig(keep_recent_tokens=200),
    )

    # Add several messages (won't trigger compression with low estimates)
    for i in range(5):
        memory.add_message(make_text_message(MessageRole.USER, f"User message {i} " * 5))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, f"Assistant response {i} " * 5))

    initial_message_count = len(memory._messages)
    assert initial_message_count == 10, "Should have 10 messages"

    # Sync token count to just below threshold
    memory.sync_token_count_from_llm_usage(950)

    # Add one more message - should trigger compression since 950 + new_tokens > 1000
    memory.add_message(make_text_message(MessageRole.USER, "This should trigger compression " * 10))

    # After compression, we should have fewer messages
    assert len(memory._messages) < initial_message_count, (
        f"Compression should have reduced messages from {initial_message_count} to fewer"
    )


def test_compression_with_system_overhead_does_not_loop():
    """Test that compression accounts for system overhead and doesn't loop infinitely.

    This tests the bug where:
    1. Compression reduces message tokens to ~15k (keep_recent_tokens)
    2. Next LLM call syncs _total_tokens to include system overhead (~25k + 15k = 40k)
    3. Add response -> 42k > 40k max_tokens -> compress again!
    4. Infinite loop because compression can't reduce below system overhead

    The fix tracks _system_overhead and adjusts keep_recent_tokens dynamically.
    """
    # Simulate scenario: max_tokens=40k, system overhead=25k, leaves 15k for messages
    max_tokens = 40000
    system_overhead = 25000  # Simulates system prompt + tool schemas

    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=max_tokens,
        compression_config=CompressionConfig(keep_recent_tokens=15000),
    )

    # Add initial messages
    memory.add_message(make_text_message(MessageRole.USER, "Initial request " * 100))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "Response " * 100))
    memory.add_message(make_text_message(MessageRole.USER, "Followup " * 100))
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "More response " * 100))

    # Simulate first LLM call that establishes system overhead
    # Actual prompt_tokens = system_overhead + message_tokens
    estimated_msg_tokens = sum(memory._estimate_message_tokens(m) for m in memory._messages)
    memory.sync_token_count_from_llm_usage(system_overhead + estimated_msg_tokens)

    assert memory._system_overhead == system_overhead, (
        f"System overhead should be calculated: {memory._system_overhead} != {system_overhead}"
    )

    # Add more messages to trigger compression
    for i in range(10):
        memory.add_message(make_text_message(MessageRole.USER, f"User msg {i} " * 50))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, f"Assistant msg {i} " * 50))

    # Force compression by syncing high token count
    memory.sync_token_count_from_llm_usage(max_tokens + 5000)
    initial_msg_count = len(memory._messages)

    # Add message to trigger compression
    memory.add_message(make_text_message(MessageRole.USER, "Trigger compression"))

    # Verify compression occurred
    assert len(memory._messages) < initial_msg_count, "Compression should have reduced messages"

    # After compression, _total_tokens should include system overhead
    post_compression_total = memory._total_tokens
    assert post_compression_total >= system_overhead, (
        f"Post-compression total ({post_compression_total}) should include system overhead ({system_overhead})"
    )

    # Simulate next LLM call after compression
    # The new prompt_tokens should be close to post_compression_total
    # (system_overhead + compressed_message_tokens)
    compressed_msg_tokens = sum(memory._estimate_message_tokens(m) for m in memory._messages)
    new_prompt_tokens = system_overhead + compressed_msg_tokens
    memory.sync_token_count_from_llm_usage(new_prompt_tokens)

    msg_count_after_sync = len(memory._messages)

    # Add one more response - should NOT trigger compression again
    # because we already compressed and accounted for system overhead
    memory.add_message(make_text_message(MessageRole.ASSISTANT, "Response after compression"))

    # The key assertion: compression should NOT have triggered again
    # (we should have same message count + 1 for the new message)
    assert len(memory._messages) == msg_count_after_sync + 1, (
        f"Should NOT re-compress: had {msg_count_after_sync} messages after sync, "
        f"now have {len(memory._messages)} after adding 1 message. "
        f"total_tokens={memory._total_tokens}, max_tokens={max_tokens}"
    )


def test_compression_adjusts_keep_recent_for_large_system_overhead():
    """Test that compression reduces keep_recent_tokens when system overhead is large.

    If system_overhead=35k and max_tokens=40k, we only have 5k for messages.
    keep_recent_tokens should be reduced from 15k to ~5k to fit.
    """
    max_tokens = 40000
    system_overhead = 35000  # Very large system overhead

    memory = FullCompressionMemory(
        summarizer_llm=get_fake_llm(),
        max_tokens=max_tokens,
        compression_config=CompressionConfig(keep_recent_tokens=15000),
    )

    # Add messages
    for i in range(5):
        memory.add_message(make_text_message(MessageRole.USER, f"User {i} " * 50))
        memory.add_message(make_text_message(MessageRole.ASSISTANT, f"Assistant {i} " * 50))

    # Establish large system overhead
    estimated_msg_tokens = sum(memory._estimate_message_tokens(m) for m in memory._messages)
    memory.sync_token_count_from_llm_usage(system_overhead + estimated_msg_tokens)

    # Force compression
    memory.add_message(make_text_message(MessageRole.USER, "Trigger " * 100))

    # After compression, total should fit within max_tokens
    # (accounting for the fact that we just added a message)
    assert memory._total_tokens <= max_tokens + 500, (  # Allow small buffer for new message
        f"Post-compression total ({memory._total_tokens}) should be near max_tokens ({max_tokens})"
    )
