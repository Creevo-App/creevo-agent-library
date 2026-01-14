from typing import List, Optional

import pytest

from CAL.tool import StopTool, tool


def test_tool_schema_generation():
    @tool
    async def sample(count: int, name: str, tags: List[str], score: Optional[float] = None):
        return {"content": [{"type": "text", "text": "ok"}], "metadata": {}}

    schema = sample.input_schema

    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"]["type"] == "string"
    assert schema["properties"]["score"]["type"] == "number"
    assert set(schema["required"]) == {"count", "name", "tags"}


def test_tool_schema_and_input_form_match():
    @tool
    async def sample(count: int, label: str):
        """Sample tool."""
        return {"content": [{"type": "text", "text": "ok"}], "metadata": {}}

    schema = sample.get_schema()
    gemini_tool = sample.gemini_input_form()

    assert schema["name"] == "sample"
    assert schema["description"] == "Sample tool."
    assert schema["input_schema"] == sample.input_schema
    assert gemini_tool.function_declarations[0].name == "sample"
    assert gemini_tool.function_declarations[0].parameters == sample.input_schema


@pytest.mark.asyncio
async def test_tool_decorator_executes():
    @tool
    async def echo(value: str):
        return {"content": [{"type": "text", "text": value}], "metadata": {"source": "test"}}

    result = await echo.execute(tool_use_id="tool-use-1", value="hello")

    assert result.is_error is False
    assert result.name == "echo"
    assert result.metadata == {"source": "test"}
    assert result.content[0].text == "hello"


@pytest.mark.asyncio
async def test_tool_decorator_invalid_output_returns_error():
    @tool
    async def bad():
        return "nope"

    result = await bad.execute(tool_use_id="tool-use-2")

    assert result.is_error is True
    assert "must return a dict" in result.content


@pytest.mark.asyncio
async def test_stop_tool_schema_and_execution():
    tool_instance = StopTool()

    schema = tool_instance.get_schema()
    gemini_tool = tool_instance.gemini_input_form()
    result = await tool_instance.execute(tool_use_id="tool-use-stop")

    assert tool_instance.name == "stop"
    assert schema["name"] == "stop"
    assert schema["input_schema"]["properties"] == {}
    assert gemini_tool.function_declarations[0].name == "stop"
    assert result.content == "STOP"
    assert result.is_error is False
