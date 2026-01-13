# Quick Test Reference

## Critical Test Scenarios

### 1. Token Counting Accuracy
```python
# Test that token counts match actual usage
message = Message(role=ASSISTANT, content="...", usage={"total_tokens": 150})
assert memory._estimate_message_tokens(message) == 150
```

### 2. Compression Trigger
```python
# Test compression triggers at token limit
memory = FullCompressionMemory(max_tokens=1000)
# Add messages until total_tokens > 1000
# Verify compress() is called
```

### 3. Archive File Creation
```python
# Test archive file is created with correct content
archiver.write_context_file(...)
assert archive_file.exists()
assert "Summary" in archive_file.read_text()
```

### 4. History Reference Injection
```python
# Test history reference is injected after compression
history = memory.get_history()
assert "[Previous conversation context" in str(history[0].content)
```

### 5. Backward Compatibility
```python
# Test old format deserialization
old_data = {"max_items": 50, "keep_recent": 25}
memory = FullCompressionMemory.from_dict(old_data)
assert memory.max_tokens > 0  # Should convert
```

### 6. LLM Fallback
```python
# Test LLM failure falls back to truncation
# Mock LLM to raise exception
# Verify truncation compression is used
```

### 7. Token Usage Tracking
```python
# Test cumulative token usage
agent.memory.add_message(assistant_msg_with_usage)
usage = agent.get_token_usage()
assert usage["total_tokens"] > 0
```

## Common Test Patterns

### Mocking Tiktoken
```python
with patch.object(memory, '_get_encoding') as mock_get:
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
    mock_get.return_value = mock_encoding
```

### Creating Test Messages
```python
# Simple text message
msg = Message(role=MessageRole.USER, content="Hello")

# Message with usage
msg = Message(role=MessageRole.ASSISTANT, content="...", 
              usage={"total_tokens": 100})

# Message with tool calls
tool_use = ToolUseBlock(id="1", name="tool", input={})
msg = Message(role=MessageRole.ASSISTANT, content=[tool_use])
```

### Temporary Archive Directory
```python
@pytest.fixture
def temp_dir(self):
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
```

## Test Execution Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_compression_archiver.py

# Run with coverage
pytest --cov=src/CAL --cov-report=term-missing

# Run verbose
pytest -v -s

# Run specific test
pytest tests/test_compression_config.py::TestCompressionConfig::test_default_values
```
