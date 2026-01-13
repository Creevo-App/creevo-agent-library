"""
Unit tests for MessageCategorizer.
"""
import pytest
from CAL.message import Message, MessageRole
from CAL.content_blocks import TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, ImageSource
from CAL.compression import MessageCategorizer, CategorizedMessages


class TestMessageCategorizer:
    """Test MessageCategorizer class."""
    
    def test_categorize_text_only_messages(self):
        """Test that text-only messages are categorized as conversations."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        
        categorized = MessageCategorizer.categorize(messages)
        assert len(categorized.conversations) == 2
        assert len(categorized.tool_calls) == 0
        assert len(categorized.file_reads) == 0
    
    def test_categorize_tool_calls(self):
        """Test that tool calls are paired with results."""
        tool_use = ToolUseBlock(id="call_1", name="test_tool", input={"arg": "value"})
        tool_result = ToolResultBlock(
            tool_use_id="call_1",
            content="Result",
            name="test_tool"
        )
        
        messages = [
            Message(role=MessageRole.ASSISTANT, content=[tool_use]),
            Message(role=MessageRole.USER, content=[tool_result]),
        ]
        
        categorized = MessageCategorizer.categorize(messages)
        assert len(categorized.tool_calls) == 1
        assert categorized.tool_calls[0]["tool_name"] == "test_tool"
        assert categorized.tool_calls[0]["tool_use"] == tool_use
        assert categorized.tool_calls[0]["tool_result"] == tool_result
    
    def test_categorize_file_reads(self):
        """Test that file read tools are identified correctly."""
        tool_use = ToolUseBlock(id="call_1", name="read_file", input={"path": "test.py"})
        tool_result = ToolResultBlock(
            tool_use_id="call_1",
            content="file content",
            name="read_file"
        )
        
        messages = [
            Message(role=MessageRole.ASSISTANT, content=[tool_use]),
            Message(role=MessageRole.USER, content=[tool_result]),
        ]
        
        categorized = MessageCategorizer.categorize(messages)
        assert len(categorized.file_reads) == 1
        assert len(categorized.tool_calls) == 0
        assert categorized.file_reads[0]["tool_name"] == "read_file"
    
    def test_categorize_mixed_content(self):
        """Test messages with mixed content types."""
        tool_use = ToolUseBlock(id="call_1", name="test_tool", input={})
        tool_result = ToolResultBlock(tool_use_id="call_1", content="Result", name="test_tool")
        text_block = TextBlock(text="Some text")
        
        messages = [
            Message(role=MessageRole.ASSISTANT, content=[text_block, tool_use]),
            Message(role=MessageRole.USER, content=[tool_result]),
        ]
        
        categorized = MessageCategorizer.categorize(messages)
        # Mixed content messages are not treated as pure conversations
        assert len(categorized.tool_calls) == 1
    
    def test_extract_tools_and_files(self):
        """Test extraction of unique tool names and file paths."""
        tool_use1 = ToolUseBlock(id="call_1", name="read_file", input={"path": "file1.py"})
        tool_result1 = ToolResultBlock(tool_use_id="call_1", content="", name="read_file")
        
        tool_use2 = ToolUseBlock(id="call_2", name="read_file", input={"file_path": "file2.py"})
        tool_result2 = ToolResultBlock(tool_use_id="call_2", content="", name="read_file")
        
        tool_use3 = ToolUseBlock(id="call_3", name="other_tool", input={"target_file": "file3.py"})
        tool_result3 = ToolResultBlock(tool_use_id="call_3", content="", name="other_tool")
        
        messages = [
            Message(role=MessageRole.ASSISTANT, content=[tool_use1]),
            Message(role=MessageRole.USER, content=[tool_result1]),
            Message(role=MessageRole.ASSISTANT, content=[tool_use2]),
            Message(role=MessageRole.USER, content=[tool_result2]),
            Message(role=MessageRole.ASSISTANT, content=[tool_use3]),
            Message(role=MessageRole.USER, content=[tool_result3]),
        ]
        
        categorized = MessageCategorizer.categorize(messages)
        tools_used, key_files = MessageCategorizer.extract_tools_and_files(categorized)
        
        assert "read_file" in tools_used
        assert "other_tool" in tools_used
        assert len(tools_used) == 2  # Unique tools only
        
        assert "file1.py" in key_files
        assert "file2.py" in key_files
        assert "file3.py" in key_files
        assert len(key_files) == 3
    
    def test_orphaned_tool_result(self):
        """Test handling of tool result without matching tool use."""
        tool_result = ToolResultBlock(tool_use_id="nonexistent", content="Result", name="test_tool")
        messages = [Message(role=MessageRole.USER, content=[tool_result])]
        
        categorized = MessageCategorizer.categorize(messages)
        # Orphaned results should still be categorized
        assert len(categorized.tool_calls) == 1
        assert categorized.tool_calls[0]["tool_use"] is None
    
    def test_orphaned_tool_use(self):
        """Test handling of tool use without matching result."""
        tool_use = ToolUseBlock(id="call_1", name="test_tool", input={})
        messages = [Message(role=MessageRole.ASSISTANT, content=[tool_use])]
        
        categorized = MessageCategorizer.categorize(messages)
        # Orphaned tool use should not be in tool_calls
        assert len(categorized.tool_calls) == 0
        assert len(categorized.file_reads) == 0
