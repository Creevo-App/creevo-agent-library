"""
Unit tests for Agent.get_token_usage().
"""
import pytest
from CAL.agent import Agent
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.content_blocks import TextBlock
from unittest.mock import Mock


class TestAgentTokenUsage:
    """Test Agent.get_token_usage() method."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.name = "test-llm"
        llm.provider = "test"
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an agent instance for tests."""
        memory = FullCompressionMemory()
        return Agent(
            llm=mock_llm,
            system_prompt="Test system prompt",
            max_calls=10,
            max_tokens=1000,
            memory=memory,
            session_id="test_session",
        )
    
    def test_returns_correct_keys(self, agent):
        """Test that get_token_usage returns dict with correct keys."""
        usage = agent.get_token_usage()
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
    
    def test_returns_zeros_when_no_usage_data(self, agent):
        """Test that zeros are returned when no usage data exists."""
        usage = agent.get_token_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
    
    def test_only_counts_assistant_messages(self, agent):
        """Test that only assistant messages with usage are counted."""
        # Add user message (no usage)
        user_msg = Message(role=MessageRole.USER, content="User message")
        agent.memory.add_message(user_msg)
        
        # Add assistant message with usage
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content="Assistant response",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        agent.memory.add_message(assistant_msg)
        
        usage = agent.get_token_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
    
    def test_sums_multiple_messages(self, agent):
        """Test that multiple assistant messages are summed correctly."""
        # Add multiple assistant messages
        for i in range(3):
            msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"Response {i}",
                usage={
                    "prompt_tokens": 100 + i * 10,
                    "completion_tokens": 50 + i * 5,
                    "total_tokens": 150 + i * 15
                }
            )
            agent.memory.add_message(msg)
        
        usage = agent.get_token_usage()
        assert usage["prompt_tokens"] == 330  # 100 + 110 + 120
        assert usage["completion_tokens"] == 165  # 50 + 55 + 60
        assert usage["total_tokens"] == 495  # 150 + 165 + 180
    
    def test_handles_missing_usage_fields(self, agent):
        """Test graceful handling of missing usage fields."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Response",
            usage={"total_tokens": 100}  # Missing prompt/completion
        )
        agent.memory.add_message(msg)
        
        usage = agent.get_token_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 100
    
    def test_ignores_messages_without_usage(self, agent):
        """Test that assistant messages without usage are ignored."""
        msg1 = Message(
            role=MessageRole.ASSISTANT,
            content="Response with usage",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        msg2 = Message(
            role=MessageRole.ASSISTANT,
            content="Response without usage",
            usage={}  # Empty usage
        )
        
        agent.memory.add_message(msg1)
        agent.memory.add_message(msg2)
        
        usage = agent.get_token_usage()
        assert usage["total_tokens"] == 150  # Only first message counted
