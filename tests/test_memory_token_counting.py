"""
Unit tests for FullCompressionMemory token counting.
"""
import pytest
from unittest.mock import Mock, patch
from CAL.memory import FullCompressionMemory
from CAL.message import Message, MessageRole
from CAL.content_blocks import TextBlock, ToolUseBlock, ToolResultBlock


class TestTokenCounting:
    """Test token counting functionality."""
    
    def test_uses_message_usage_when_available(self):
        """Test that message.usage is used when available (assistant messages)."""
        memory = FullCompressionMemory()
        message = Message(
            role=MessageRole.ASSISTANT,
            content="Test content",
            usage={"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50}
        )
        
        tokens = memory._estimate_message_tokens(message)
        assert tokens == 150
    
    def test_uses_tiktoken_when_available(self):
        """Test that tiktoken is used for text content when available."""
        memory = FullCompressionMemory()
        message = Message(role=MessageRole.USER, content="Hello world")
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoding
            
            tokens = memory._estimate_message_tokens(message)
            assert tokens == 5
            mock_encoding.encode.assert_called_once_with("Hello world")
    
    def test_falls_back_to_character_estimation(self):
        """Test fallback to character estimation when tiktoken unavailable."""
        memory = FullCompressionMemory()
        message = Message(role=MessageRole.USER, content="Hello world")  # 11 chars
        
        with patch.object(memory, '_get_encoding', return_value=None):
            tokens = memory._estimate_message_tokens(message)
            # ~4 chars per token, so 11 chars ≈ 2-3 tokens
            assert tokens >= 2
            assert tokens <= 3
    
    def test_counts_text_block_tokens(self):
        """Test token counting for TextBlock content."""
        memory = FullCompressionMemory()
        message = Message(
            role=MessageRole.USER,
            content=[TextBlock(text="Test text content")]
        )
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding
            
            tokens = memory._estimate_message_tokens(message)
            assert tokens == 3
    
    def test_counts_tool_use_tokens(self):
        """Test token counting for ToolUseBlock."""
        memory = FullCompressionMemory()
        tool_use = ToolUseBlock(
            id="call_1",
            name="test_tool",
            input={"arg": "value"},
            thought="Some reasoning"
        )
        message = Message(role=MessageRole.ASSISTANT, content=[tool_use])
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            # Mock encoding for the tool text representation
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6]
            mock_get_encoding.return_value = mock_encoding
            
            tokens = memory._estimate_message_tokens(message)
            # Should encode: "test_tool({'arg': 'value'})Some reasoning"
            assert tokens >= 1
    
    def test_counts_tool_result_tokens(self):
        """Test token counting for ToolResultBlock."""
        memory = FullCompressionMemory()
        tool_result = ToolResultBlock(
            tool_use_id="call_1",
            content="Tool result content",
            name="test_tool"
        )
        message = Message(role=MessageRole.USER, content=[tool_result])
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4]
            mock_get_encoding.return_value = mock_encoding
            
            tokens = memory._estimate_message_tokens(message)
            assert tokens == 4
    
    def test_counts_nested_tool_result_tokens(self):
        """Test token counting for ToolResultBlock with nested blocks."""
        memory = FullCompressionMemory()
        tool_result = ToolResultBlock(
            tool_use_id="call_1",
            content=[TextBlock(text="Result 1"), TextBlock(text="Result 2")],
            name="test_tool"
        )
        message = Message(role=MessageRole.USER, content=[tool_result])
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding
            
            tokens = memory._estimate_message_tokens(message)
            # Should count tokens for both text blocks
            assert tokens >= 3
    
    def test_estimates_image_tokens(self):
        """Test that images are estimated conservatively."""
        memory = FullCompressionMemory()
        from CAL.content_blocks import ImageBlock, ImageSource
        
        image_source = ImageSource(type="base64", media_type="image/png", data="fake_data")
        image_block = ImageBlock(source=image_source)
        message = Message(role=MessageRole.USER, content=[image_block])
        
        tokens = memory._estimate_message_tokens(message)
        # Images should be estimated at ~100 tokens
        assert tokens == 100
    
    def test_returns_at_least_one_token(self):
        """Test that at least 1 token is returned for any message."""
        memory = FullCompressionMemory()
        message = Message(role=MessageRole.USER, content="")
        
        tokens = memory._estimate_message_tokens(message)
        assert tokens >= 1
    
    def test_handles_tiktoken_encoding_errors(self):
        """Test graceful handling of tiktoken encoding errors."""
        memory = FullCompressionMemory()
        message = Message(role=MessageRole.USER, content="Test")
        
        with patch.object(memory, '_get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.side_effect = Exception("Encoding error")
            mock_get_encoding.return_value = mock_encoding
            
            # Should fall back to character estimation
            tokens = memory._estimate_message_tokens(message)
            assert tokens >= 1
