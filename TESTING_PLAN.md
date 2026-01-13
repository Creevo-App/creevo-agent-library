# Testing and Validation Plan for Memory Compression Changes

## Overview

This document outlines a comprehensive testing plan for the new memory compression features introduced in this branch. The changes include:

1. **Token-based compression** (replacing message-count-based)
2. **Tiktoken integration** for accurate token counting
3. **File-based archival** system for compressed conversations
4. **LLM-based intelligent summarization**
5. **Message categorization** for better compression
6. **Token usage tracking** in Agent

## Test Categories

### 1. Unit Tests

#### 1.1 CompressionConfig (`compression.py`)
- [ ] Default values are correct
- [ ] All fields can be set and retrieved
- [ ] Dataclass serialization/deserialization works

#### 1.2 CompressionArchiver (`compression.py`)
- [ ] Initialization with session_id creates correct directory structure
- [ ] Initialization with custom base_dir uses provided path
- [ ] `write_context_file()` creates files with correct naming
- [ ] Filename collision handling (counter appending)
- [ ] `_make_safe_filename()` sanitizes unsafe characters
- [ ] `_update_history_file()` creates/updates history.md correctly
- [ ] `get_history_reference()` returns correct format
- [ ] `has_archived_context()` returns correct state
- [ ] `cleanup()` removes directory and clears entries
- [ ] `to_dict()` serializes correctly
- [ ] `from_dict()` restores state correctly
- [ ] Multiple archive entries are tracked correctly

#### 1.3 MessageCategorizer (`compression.py`)
- [ ] Categorizes text-only messages as conversations
- [ ] Pairs ToolUseBlock with ToolResultBlock correctly
- [ ] Identifies file read tools correctly (read_file, cat, etc.)
- [ ] Separates file reads from regular tool calls
- [ ] Handles images correctly
- [ ] Handles messages with mixed content types
- [ ] `extract_tools_and_files()` extracts unique tool names
- [ ] `extract_tools_and_files()` extracts file paths from various input keys
- [ ] Handles orphaned ToolResultBlocks (no matching ToolUseBlock)
- [ ] Handles orphaned ToolUseBlocks (no matching ToolResultBlock)

#### 1.4 FullCompressionMemory Token Counting (`memory.py`)
- [ ] `_estimate_message_tokens()` uses message.usage when available (assistant messages)
- [ ] `_estimate_message_tokens()` uses tiktoken for text content when available
- [ ] `_estimate_message_tokens()` falls back to character estimation (~4 chars/token)
- [ ] Token counting works for string content
- [ ] Token counting works for TextBlock content
- [ ] Token counting works for ToolUseBlock (includes name + input + thought)
- [ ] Token counting works for ToolResultBlock (handles str, list, nested blocks)
- [ ] Token counting estimates images conservatively (~100 tokens)
- [ ] Returns at least 1 token for any message
- [ ] Handles missing tiktoken gracefully (fallback)
- [ ] Handles tiktoken encoding errors gracefully

#### 1.5 FullCompressionMemory Compression Logic (`memory.py`)
- [ ] `compress()` does nothing if ≤3 messages
- [ ] `compress()` keeps initial user message
- [ ] `compress()` keeps recent messages up to `keep_recent_tokens` limit
- [ ] `compress()` ensures at least 1 recent message is kept if possible
- [ ] `compress()` selects correct compression method:
  - [ ] "llm_file_based" when summarizer_llm and archiver available
  - [ ] "llm_inline" when only summarizer_llm available
  - [ ] "truncation" when no summarizer_llm
- [ ] `compress()` recalculates `_total_tokens` after compression
- [ ] Compression triggers when `_total_tokens > max_tokens`
- [ ] Compression doesn't trigger unnecessarily

#### 1.6 FullCompressionMemory LLM Compression (`memory.py`)
- [ ] `_compress_with_llm()` calls file-based compression when archiver available
- [ ] `_compress_with_llm()` calls inline compression when no archiver
- [ ] `_build_compression_prompt()` formats messages correctly
- [ ] `_build_file_compression_prompt()` requests JSON output
- [ ] `_parse_llm_json_response()` handles markdown code blocks
- [ ] `_parse_llm_json_response()` handles plain JSON
- [ ] `_parse_llm_json_response()` returns defaults on parse failure
- [ ] `_format_archive_content()` includes summary, tool calls, file reads
- [ ] `_format_archive_content()` preserves full tool inputs/results (no truncation)
- [ ] `_format_archive_content()` handles errors in tool results
- [ ] LLM compression fallback to truncation on exception

#### 1.7 FullCompressionMemory Truncation Fallback (`memory.py`)
- [ ] `_compress_with_truncation()` creates summary with message count
- [ ] `_summarize_content()` truncates long strings (>150 chars)
- [ ] `_summarize_content()` handles TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock
- [ ] `_format_message_for_prompt()` formats string content
- [ ] `_format_message_for_prompt()` formats block content correctly
- [ ] `_format_message_for_prompt()` preserves tool names when configured
- [ ] `_format_message_for_prompt()` truncates large input values (>200 chars)

#### 1.8 FullCompressionMemory History Management (`memory.py`)
- [ ] `get_history()` returns messages in correct order
- [ ] `get_history()` injects history reference when archived context exists
- [ ] `get_history()` injects reference into first compressed message
- [ ] `get_history()` handles string content injection
- [ ] `get_history()` handles block content injection
- [ ] `clear()` resets messages and token count

#### 1.9 FullCompressionMemory Serialization (`memory.py`)
- [ ] `to_dict()` includes compression_config
- [ ] `to_dict()` includes archiver state if present
- [ ] `to_dict()` serializes messages correctly
- [ ] `to_json()` produces valid JSON
- [ ] `from_dict()` handles backward compatibility (max_items → max_tokens)
- [ ] `from_dict()` handles backward compatibility (keep_recent → keep_recent_tokens)
- [ ] `from_dict()` restores compression_config correctly
- [ ] `from_dict()` restores archiver if present
- [ ] `from_dict()` accepts optional summarizer_llm and logger
- [ ] `from_json()` handles None/empty input
- [ ] `from_json()` uses provided session_id if not in payload
- [ ] Round-trip serialization preserves all data

#### 1.10 Agent Token Usage (`agent.py`)
- [ ] `get_token_usage()` returns dict with correct keys
- [ ] `get_token_usage()` only counts assistant messages
- [ ] `get_token_usage()` only counts messages with usage data
- [ ] `get_token_usage()` sums prompt_tokens correctly
- [ ] `get_token_usage()` sums completion_tokens correctly
- [ ] `get_token_usage()` sums total_tokens correctly
- [ ] `get_token_usage()` handles missing usage fields gracefully
- [ ] `get_token_usage()` returns zeros when no usage data

### 2. Integration Tests

#### 2.1 End-to-End Compression Flow
- [ ] Memory compresses automatically when token limit exceeded
- [ ] Compression preserves initial message
- [ ] Compression preserves recent messages within token limit
- [ ] Compressed summary is added to history correctly
- [ ] Token count is accurate after compression
- [ ] Multiple compression cycles work correctly
- [ ] History reference is injected correctly after compression

#### 2.2 File-Based Archival Flow
- [ ] Archive directory is created on first compression
- [ ] Context file is created with correct content
- [ ] history.md index is created/updated correctly
- [ ] History reference includes correct file paths
- [ ] Multiple archive files are tracked correctly
- [ ] Archive files contain full tool call details
- [ ] Archive files contain full file read contents
- [ ] Archive cleanup removes all files

#### 2.3 LLM Summarization Integration
- [ ] LLM receives correctly formatted compression prompt
- [ ] LLM JSON response is parsed correctly
- [ ] Semantic filename is generated correctly
- [ ] Summary is included in archive file
- [ ] Summary is included in history reference
- [ ] LLM failure falls back to truncation gracefully

#### 2.4 Agent Integration
- [ ] Agent with FullCompressionMemory compresses during long conversations
- [ ] Agent.get_token_usage() tracks cumulative usage correctly
- [ ] Agent continues to function after compression
- [ ] Tool calls work correctly after compression
- [ ] Conversation context is preserved after compression

### 3. Edge Cases and Error Handling

#### 3.1 Token Counting Edge Cases
- [ ] Empty message content
- [ ] Very long messages (>100k tokens)
- [ ] Messages with only images
- [ ] Messages with only tool calls (no text)
- [ ] Messages with nested content blocks
- [ ] Unicode characters in content
- [ ] Binary data in content

#### 3.2 Compression Edge Cases
- [ ] Compression with exactly max_tokens
- [ ] Compression with all messages fitting in keep_recent_tokens
- [ ] Compression with no recent messages to keep
- [ ] Compression with only initial message
- [ ] Compression with malformed messages
- [ ] Compression during active tool execution

#### 3.3 Archival Edge Cases
- [ ] Archive with no tool calls
- [ ] Archive with no file reads
- [ ] Archive with only conversations
- [ ] Archive with very long filenames
- [ ] Archive with special characters in filenames
- [ ] Archive with duplicate semantic filenames
- [ ] Archive directory permissions issues
- [ ] Archive disk space issues

#### 3.4 LLM Integration Edge Cases
- [ ] LLM returns invalid JSON
- [ ] LLM returns empty response
- [ ] LLM times out
- [ ] LLM returns non-JSON in file-based mode
- [ ] LLM returns JSON missing required fields
- [ ] LLM compression with no messages to compress

#### 3.5 Serialization Edge Cases
- [ ] Serialization with None values
- [ ] Serialization with empty lists
- [ ] Deserialization of corrupted JSON
- [ ] Deserialization of old format (backward compatibility)
- [ ] Deserialization with missing fields
- [ ] Round-trip with special characters

### 4. Performance Tests

#### 4.1 Token Counting Performance
- [ ] Token counting is fast for typical messages (<10ms)
- [ ] Token counting scales linearly with message size
- [ ] Tiktoken is faster than character estimation
- [ ] Token counting doesn't block compression

#### 4.2 Compression Performance
- [ ] Compression completes in reasonable time (<5s for 100 messages)
- [ ] File-based compression doesn't significantly slow down
- [ ] Multiple compressions don't degrade performance
- [ ] Memory usage doesn't grow unbounded

#### 4.3 Archival Performance
- [ ] Archive file creation is fast (<1s)
- [ ] history.md updates are fast (<100ms)
- [ ] Large archive files don't cause issues
- [ ] Cleanup is fast

### 5. Backward Compatibility Tests

#### 5.1 Old Format Compatibility
- [ ] Deserialize memory with old `max_items` field
- [ ] Deserialize memory with old `keep_recent` field (message count)
- [ ] Deserialize memory without compression_config
- [ ] Deserialize memory without archiver
- [ ] Conversion estimates are reasonable (messages → tokens)

#### 5.2 API Compatibility
- [ ] Existing Agent code works without changes
- [ ] Memory interface (add_message, get_history, clear) unchanged
- [ ] Optional parameters don't break existing code
- [ ] Default behavior matches previous behavior (when possible)

### 6. Validation Tests

#### 6.1 Token Count Accuracy
- [ ] Compare tiktoken counts with actual LLM usage data
- [ ] Verify character estimation is reasonable (~4 chars/token)
- [ ] Verify token counts match across compression cycles
- [ ] Verify token counts are consistent with LLM provider counts

#### 6.2 Compression Quality
- [ ] Compressed summaries preserve key information
- [ ] Tool calls are preserved with sufficient detail
- [ ] File reads are preserved completely
- [ ] Conversation flow is coherent after compression
- [ ] Recent context is sufficient for continuation

#### 6.3 Archive File Quality
- [ ] Archive files are readable and well-formatted
- [ ] Archive files contain all necessary context
- [ ] history.md is clear and informative
- [ ] File paths are correct and accessible

### 7. Test Data Requirements

#### 7.1 Test Messages
- [ ] Simple text messages
- [ ] Messages with tool calls
- [ ] Messages with file reads
- [ ] Messages with images
- [ ] Mixed content messages
- [ ] Long messages (>10k tokens)
- [ ] Messages with usage metadata
- [ ] Messages without usage metadata

#### 7.2 Test Scenarios
- [ ] Short conversation (<10 messages)
- [ ] Medium conversation (10-50 messages)
- [ ] Long conversation (50-200 messages)
- [ ] Very long conversation (>200 messages)
- [ ] Conversation with many tool calls
- [ ] Conversation with many file reads
- [ ] Conversation with compression triggered multiple times

### 8. Manual Testing Checklist

#### 8.1 Basic Functionality
- [ ] Create Agent with FullCompressionMemory
- [ ] Run agent with conversation exceeding token limit
- [ ] Verify compression occurs automatically
- [ ] Verify conversation continues correctly
- [ ] Check token usage tracking

#### 8.2 File Archival
- [ ] Verify archive directory is created
- [ ] Verify context files are created
- [ ] Verify history.md is created
- [ ] Read archive files and verify content
- [ ] Verify history reference in conversation

#### 8.3 LLM Summarization
- [ ] Verify LLM is called for summarization
- [ ] Verify summary quality
- [ ] Verify semantic filename generation
- [ ] Test with different LLM providers

#### 8.4 Error Scenarios
- [ ] Test with missing tiktoken
- [ ] Test with LLM failure
- [ ] Test with disk space issues
- [ ] Test with permission issues

## Test Implementation Strategy

### Phase 1: Unit Tests (Priority: High)
1. Start with CompressionConfig and CompressionArchiver (isolated, no dependencies)
2. Add MessageCategorizer tests (depends on Message/ContentBlock)
3. Add token counting tests (depends on tiktoken)
4. Add compression logic tests (can use mocks for LLM)

### Phase 2: Integration Tests (Priority: High)
1. End-to-end compression flow
2. File archival flow
3. Agent integration

### Phase 3: Edge Cases and Error Handling (Priority: Medium)
1. Error scenarios
2. Boundary conditions
3. Backward compatibility

### Phase 4: Performance and Validation (Priority: Medium)
1. Performance benchmarks
2. Accuracy validation
3. Quality checks

## Test Framework Recommendations

- **Framework**: pytest (standard Python testing)
- **Mocking**: unittest.mock for LLM and file system
- **Fixtures**: pytest fixtures for common test data
- **Coverage**: Aim for >90% code coverage
- **CI/CD**: Run tests on every commit

## Success Criteria

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage >90%
- [ ] No performance regressions
- [ ] Backward compatibility maintained
- [ ] Documentation updated
- [ ] Manual testing completed successfully

## Notes

- Tests should be run with and without tiktoken installed
- Tests should verify graceful degradation when optional features unavailable
- Consider using pytest fixtures for common test scenarios
- Mock LLM calls to avoid API costs during testing
- Use temporary directories for archive tests
