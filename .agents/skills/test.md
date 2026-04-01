# Run Tests

Run the CAL test suite using pytest.

## Usage

```
/test [options]
```

## Options

- No arguments: Run all tests
- `<test_file>`: Run a specific test file (e.g., `test_subagent.py`)
- `<test_file>::<test_name>`: Run a specific test function
- `-v`: Verbose output
- `-k <pattern>`: Run tests matching pattern

## Instructions

1. First, ensure the virtual environment exists and is working:
   ```bash
   ./venv/bin/python --version
   ```

2. If the venv is broken or doesn't exist, recreate it:
   ```bash
   rm -rf venv
   python3.13 -m venv venv
   ./venv/bin/pip install -e ".[dev]"
   ```

3. Run the tests using the venv's pytest:
   ```bash
   ./venv/bin/pytest tests/ -v
   ```

4. For a specific test file:
   ```bash
   ./venv/bin/pytest tests/test_subagent.py -v
   ```

5. For a specific test:
   ```bash
   ./venv/bin/pytest tests/test_subagent.py::test_subagent_inherits_memory_max_tokens_not_agent_max_tokens -v
   ```

## Common Test Files

- `tests/test_agent.py` - Agent class tests
- `tests/test_subagent.py` - SubAgent and delegation tests
- `tests/test_mcp.py` - MCP tool and connection tests
- `tests/test_memory.py` - Memory and compression tests
- `tests/test_tool.py` - Tool decorator and execution tests
- `tests/test_integration.py` - Integration tests (require API keys)

## Notes

- Integration tests are marked with `@pytest.mark.integration` and require external services
- To skip integration tests: `./venv/bin/pytest tests/ -v -m "not integration"`
- The tests directory contains a `.env` file for test configuration
