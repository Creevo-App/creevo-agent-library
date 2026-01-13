# CAL Test Suite

This directory contains tests for the CAL (Creevo Agent Library) memory compression features.

## Running Tests

### Setup (First Time)

1. Create a virtual environment (if not already created):
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install the project and test dependencies:
```bash
pip install -e ".[dev]"
```

### Run All Tests

**Option 1: Using the helper script**
```bash
./run_tests.sh
```

**Option 2: Activate venv and run directly**
```bash
source venv/bin/activate
pytest
```

**Option 3: Use venv's pytest directly**
```bash
./venv/bin/pytest
```

### Run Specific Test File
```bash
pytest tests/test_compression_config.py
```

### Run with Coverage
```bash
pytest --cov=src/CAL --cov-report=html
```

### Run Verbose
```bash
pytest -v
```

## Test Structure

- `test_compression_config.py` - Tests for CompressionConfig dataclass
- `test_compression_archiver.py` - Tests for CompressionArchiver file archival
- `test_message_categorizer.py` - Tests for MessageCategorizer
- `test_memory_token_counting.py` - Tests for token counting functionality
- `test_agent_token_usage.py` - Tests for Agent.get_token_usage()

## Test Coverage Goals

- Unit tests: >90% coverage
- Integration tests: Cover all major workflows
- Edge cases: All error paths tested

## Notes

- Tests use temporary directories for file operations
- LLM calls are mocked to avoid API costs
- Tests should work with and without tiktoken installed
