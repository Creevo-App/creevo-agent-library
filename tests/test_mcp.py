"""Tests for CAL MCP (Model Context Protocol) support."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from CAL.content_blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from CAL.mcp import (
    MCPServerConnection,
    MCPTool,
    MCPToolList,
    _clean_schema,
    _map_mcp_content,
    connect_mcp_server,
    disconnect_mcp_tools,
)
from CAL.tool import StopTool
from CAL.agent import Agent
from CAL.message import Message, MessageRole
from conftest import QueueLLM


# -- Helpers ----------------------------------------------------------------

def _mock_conn() -> MagicMock:
    conn = MagicMock(spec=MCPServerConnection)
    conn.call_tool = AsyncMock()
    conn.disconnect = AsyncMock()
    return conn


def _text(text: str):
    return SimpleNamespace(type="text", text=text)


def _image(mime: str, data: str):
    return SimpleNamespace(type="image", mimeType=mime, data=data)


def _resource(uri: str, text: str = None):
    resource = SimpleNamespace(uri=uri)
    if text is not None:
        resource.text = text
    return SimpleNamespace(type="resource", resource=resource)


def _result(content: list, is_error: bool = False):
    return SimpleNamespace(content=content, isError=is_error)


def _make_tool(conn=None) -> MCPTool:
    return MCPTool(
        name="search",
        description="Search the web",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        connection=conn or _mock_conn(),
    )


# -- _map_mcp_content ------------------------------------------------------

class TestMapMCPContent:
    def test_text(self):
        blocks = _map_mcp_content([_text("hello")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "hello"

    def test_image(self):
        blocks = _map_mcp_content([_image("image/png", "abc123")])
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].source.media_type == "image/png"
        assert blocks[0].source.data == "abc123"

    def test_resource_with_text(self):
        blocks = _map_mcp_content([_resource("file:///a.txt", text="contents")])
        assert blocks[0].text == "contents"

    def test_resource_without_text(self):
        blocks = _map_mcp_content([_resource("file:///a.txt")])
        assert "[Resource: file:///a.txt]" in blocks[0].text

    def test_unknown_type_fallback(self):
        blocks = _map_mcp_content([SimpleNamespace(type="custom", payload="data")])
        assert isinstance(blocks[0], TextBlock)

    def test_multiple_items(self):
        blocks = _map_mcp_content([_text("a"), _text("b"), _image("image/jpeg", "img")])
        assert len(blocks) == 3
        assert isinstance(blocks[2], ImageBlock)

    def test_empty_list(self):
        assert _map_mcp_content([]) == []


# -- _clean_schema ----------------------------------------------------------

class TestCleanSchema:
    def test_strips_dollar_schema(self):
        raw = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {}}
        assert "$schema" not in _clean_schema(raw)
        assert _clean_schema(raw)["type"] == "object"

    def test_preserves_valid_keys(self):
        raw = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        assert _clean_schema(raw) == raw

    def test_strips_nested_extras(self):
        raw = {
            "type": "object",
            "properties": {
                "config": {"type": "object", "$id": "bad", "properties": {"k": {"type": "string", "$comment": "x"}}}
            },
        }
        cleaned = _clean_schema(raw)
        assert "$id" not in cleaned["properties"]["config"]
        assert "$comment" not in cleaned["properties"]["config"]["properties"]["k"]

    def test_cleans_items(self):
        raw = {"type": "array", "items": {"type": "string", "$anchor": "x"}}
        assert "$anchor" not in _clean_schema(raw)["items"]

    def test_cleans_any_of(self):
        raw = {"anyOf": [{"type": "string", "$ref": "#/defs/a"}, {"type": "integer", "$comment": "b"}]}
        cleaned = _clean_schema(raw)
        # $ref is a valid Gemini key (mapped as "ref"), $comment is not
        assert "$comment" not in cleaned["anyOf"][1]

    def test_cleans_additional_properties_schema(self):
        raw = {
            "type": "object",
            "additionalProperties": {"type": "string", "$comment": "leak"},
        }
        cleaned = _clean_schema(raw)
        assert "$comment" not in cleaned["additionalProperties"]
        assert cleaned["additionalProperties"]["type"] == "string"

    def test_additional_properties_bool_passthrough(self):
        raw = {"type": "object", "additionalProperties": False}
        assert _clean_schema(raw)["additionalProperties"] is False

    def test_cleans_defs(self):
        raw = {
            "type": "object",
            "defs": {
                "MyType": {"type": "string", "$id": "bad", "description": "ok"},
            },
        }
        cleaned = _clean_schema(raw)
        assert "$id" not in cleaned["defs"]["MyType"]
        assert cleaned["defs"]["MyType"]["description"] == "ok"


# -- MCPTool ----------------------------------------------------------------

class TestMCPTool:
    def test_get_schema(self):
        schema = _make_tool().get_schema()
        assert schema["name"] == "search"
        assert schema["input_schema"]["properties"]["query"]["type"] == "string"

    def test_schema_strips_mcp_extras(self):
        """MCP schemas with $schema etc. are cleaned at construction time."""
        tool = MCPTool(
            name="t", description="d",
            input_schema={"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {}},
            connection=_mock_conn(),
        )
        assert "$schema" not in tool.input_schema
        assert "$schema" not in tool.get_schema()["input_schema"]

    def test_gemini_input_form(self):
        gemini = _make_tool().gemini_input_form()
        fd = gemini.function_declarations[0]
        assert fd.name == "search"
        assert fd.description == "Search the web"

    def test_gemini_form_with_mcp_schema(self):
        """FunctionDeclaration doesn't raise with MCP-style schemas after cleaning."""
        tool = MCPTool(
            name="resolve", description="Resolve a library",
            input_schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            connection=_mock_conn(),
        )
        gemini = tool.gemini_input_form()
        assert gemini.function_declarations[0].name == "resolve"

    async def test_execute_text_response(self):
        conn = _mock_conn()
        conn.call_tool.return_value = _result([_text("result text")])
        result = await _make_tool(conn).execute(tool_use_id="id1", query="hello")

        conn.call_tool.assert_awaited_once_with("search", {"query": "hello"})
        assert result.is_error is False
        assert result.content[0].text == "result text"

    async def test_execute_image_response(self):
        conn = _mock_conn()
        conn.call_tool.return_value = _result([_image("image/png", "base64data")])
        result = await _make_tool(conn).execute(tool_use_id="id2", query="cat photo")

        assert isinstance(result.content[0], ImageBlock)
        assert result.content[0].source.data == "base64data"

    async def test_execute_error_response(self):
        conn = _mock_conn()
        conn.call_tool.return_value = _result([_text("bad request")], is_error=True)
        result = await _make_tool(conn).execute(tool_use_id="id3", query="bad")

        assert result.is_error is True
        assert result.content[0].text == "bad request"

    async def test_execute_connection_error(self):
        conn = _mock_conn()
        conn.call_tool.side_effect = RuntimeError("server crashed")
        result = await _make_tool(conn).execute(tool_use_id="id4", query="oops")

        assert result.is_error is True
        assert "server crashed" in result.content

    async def test_execute_empty_content(self):
        conn = _mock_conn()
        conn.call_tool.return_value = _result([])
        result = await _make_tool(conn).execute(tool_use_id="id5", query="empty")

        assert result.content[0].text == ""

    async def test_execute_null_content(self):
        conn = _mock_conn()
        conn.call_tool.return_value = SimpleNamespace(content=None, isError=False)
        result = await _make_tool(conn).execute(tool_use_id="id6", query="null")

        assert result.content[0].text == ""

    def test_repr(self):
        assert "MCPTool" in repr(_make_tool())
        assert "search" in repr(_make_tool())


# -- MCPTool in Agent loop --------------------------------------------------

class TestMCPToolInAgentLoop:
    async def test_agent_calls_mcp_tool(self):
        conn = _mock_conn()
        conn.call_tool.return_value = _result([_text("MCP answer")])

        mcp_tool = MCPTool(
            name="lookup", description="Look things up",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            connection=conn,
        )
        responses = [
            Message(role=MessageRole.ASSISTANT, content=[ToolUseBlock(id="tc1", name="lookup", input={"q": "test"})]),
            Message(role=MessageRole.ASSISTANT, content=[ToolUseBlock(id="tc2", name="stop", input={})]),
        ]
        agent = Agent(
            llm=QueueLLM(responses), system_prompt="You are helpful.",
            max_calls=5, max_tokens=4096,
            agent_name="mcp-test",
            tools=[mcp_tool, StopTool()],
        )
        await agent.run_async("test query")
        conn.call_tool.assert_awaited_once_with("lookup", {"q": "test"})


# -- disconnect_mcp_tools ---------------------------------------------------

class TestDisconnectMCPTools:
    async def test_disconnects_unique_connections(self):
        conn1, conn2 = _mock_conn(), _mock_conn()
        tools = [
            MCPTool(name="a", description="", input_schema={}, connection=conn1),
            MCPTool(name="b", description="", input_schema={}, connection=conn1),
            MCPTool(name="c", description="", input_schema={}, connection=conn2),
        ]
        await disconnect_mcp_tools(tools)
        conn1.disconnect.assert_awaited_once()
        conn2.disconnect.assert_awaited_once()

    async def test_ignores_non_mcp_tools(self):
        conn = _mock_conn()
        mcp = MCPTool(name="x", description="", input_schema={}, connection=conn)
        await disconnect_mcp_tools([StopTool(), mcp])
        conn.disconnect.assert_awaited_once()

    async def test_empty_list(self):
        await disconnect_mcp_tools([])

    async def test_accepts_raw_connection(self):
        conn = _mock_conn()
        await disconnect_mcp_tools([conn])
        conn.disconnect.assert_awaited_once()

    async def test_deduplicates_raw_and_tool_connections(self):
        conn = _mock_conn()
        mcp = MCPTool(name="x", description="", input_schema={}, connection=conn)
        await disconnect_mcp_tools([conn, mcp])
        conn.disconnect.assert_awaited_once()


# -- connect_mcp_server -----------------------------------------------------

class TestConnectMCPServer:
    async def test_returns_mcp_tools(self):
        mock_tool = SimpleNamespace(
            name="resolve_library_id", description="Resolve a library ID",
            inputSchema={"type": "object", "properties": {"libraryName": {"type": "string"}}, "required": ["libraryName"]},
        )
        mock_conn = _mock_conn()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(return_value=[mock_tool])

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            tools = await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        assert len(tools) == 1
        assert tools[0].name == "resolve_library_id"
        assert tools[0]._connection is mock_conn

    async def test_disconnects_on_list_tools_failure(self):
        mock_conn = _mock_conn()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(side_effect=RuntimeError("protocol error"))

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="protocol error"):
                await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        mock_conn.disconnect.assert_awaited_once()

    async def test_empty_tools_returns_mcp_tool_list_with_connection(self):
        mock_conn = _mock_conn()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(return_value=[])

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            tools = await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        assert isinstance(tools, MCPToolList)
        assert len(tools) == 0
        assert tools._connection is mock_conn

    async def test_disconnect_works_with_empty_mcp_tool_list(self):
        mock_conn = _mock_conn()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(return_value=[])

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            tools = await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        await disconnect_mcp_tools(tools)
        mock_conn.disconnect.assert_awaited_once()


# -- MCPServerConnection ----------------------------------------------------

class TestMCPServerConnection:
    def test_init(self):
        assert MCPServerConnection(command="npx")._session is None

    async def test_list_tools_before_connect_raises(self):
        with pytest.raises(RuntimeError, match="not connected"):
            await MCPServerConnection(command="npx").list_tools()

    async def test_call_tool_before_connect_raises(self):
        with pytest.raises(RuntimeError, match="not connected"):
            await MCPServerConnection(command="npx").call_tool("foo", {})

    async def test_connect_cleans_up_on_partial_failure(self):
        conn = MCPServerConnection(command="npx")
        fake_stack = AsyncMock(spec=conn._exit_stack)
        fake_stack.enter_async_context = AsyncMock(
            side_effect=[("read", "write"), RuntimeError("session init failed")]
        )
        fake_stack.aclose = AsyncMock()
        conn._exit_stack = fake_stack

        with pytest.raises(RuntimeError, match="session init failed"):
            await conn.connect()

        fake_stack.aclose.assert_awaited_once()
        assert conn._session is None


# -- Integration (requires npx + internet) ----------------------------------

@pytest.mark.integration
class TestContext7MCPIntegration:
    async def test_connect_and_list_tools(self):
        tools = await connect_mcp_server(command="npx", args=["-y", "@upstash/context7-mcp"])
        try:
            assert len(tools) > 0
            for t in tools:
                assert isinstance(t, MCPTool)
                assert t.name
        finally:
            await disconnect_mcp_tools(tools)
