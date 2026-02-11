"""Tests for CAL MCP (Model Context Protocol) support."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from CAL.content_blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from CAL.mcp import (
    MCPServerConnection,
    MCPTool,
    _map_mcp_content,
    connect_mcp_server,
    disconnect_mcp_tools,
)
from CAL.tool import StopTool

from conftest import QueueLLM, TrackingMemory, make_text_message, make_tool_use_message
from CAL.agent import Agent
from CAL.message import Message, MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_connection() -> MagicMock:
    """Return a mock MCPServerConnection with a working call_tool."""
    conn = MagicMock(spec=MCPServerConnection)
    conn.call_tool = AsyncMock()
    conn.disconnect = AsyncMock()
    return conn


def _text_content(text: str):
    """Simulate an MCP TextContent item."""
    return SimpleNamespace(type="text", text=text)


def _image_content(mime: str, data: str):
    """Simulate an MCP ImageContent item."""
    return SimpleNamespace(type="image", mimeType=mime, data=data)


def _resource_content(uri: str, text: str = None):
    """Simulate an MCP EmbeddedResource item."""
    resource = SimpleNamespace(uri=uri)
    if text is not None:
        resource.text = text
    return SimpleNamespace(type="resource", resource=resource)


def _call_result(content: list, is_error: bool = False):
    """Simulate an MCP CallToolResult."""
    return SimpleNamespace(content=content, isError=is_error)


# ---------------------------------------------------------------------------
# _map_mcp_content
# ---------------------------------------------------------------------------

class TestMapMCPContent:
    def test_text(self):
        blocks = _map_mcp_content([_text_content("hello")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "hello"

    def test_image(self):
        blocks = _map_mcp_content([_image_content("image/png", "abc123")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].source.media_type == "image/png"
        assert blocks[0].source.data == "abc123"
        assert blocks[0].source.type == "base64"

    def test_resource_with_text(self):
        blocks = _map_mcp_content([_resource_content("file:///a.txt", text="contents")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "contents"

    def test_resource_without_text(self):
        blocks = _map_mcp_content([_resource_content("file:///a.txt")])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "[Resource: file:///a.txt]" in blocks[0].text

    def test_unknown_type_fallback(self):
        item = SimpleNamespace(type="custom", payload="data")
        blocks = _map_mcp_content([item])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)

    def test_multiple_items(self):
        blocks = _map_mcp_content([
            _text_content("first"),
            _text_content("second"),
            _image_content("image/jpeg", "img"),
        ])
        assert len(blocks) == 3
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], TextBlock)
        assert isinstance(blocks[2], ImageBlock)

    def test_empty_list(self):
        blocks = _map_mcp_content([])
        assert blocks == []


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class TestMCPTool:
    def _make_tool(self, conn=None) -> MCPTool:
        conn = conn or _make_mock_connection()
        return MCPTool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            connection=conn,
        )

    # -- schema / gemini form -----------------------------------------------

    def test_get_schema(self):
        tool = self._make_tool()
        schema = tool.get_schema()
        assert schema["name"] == "search"
        assert schema["description"] == "Search the web"
        assert "input_schema" in schema
        assert schema["input_schema"]["properties"]["query"]["type"] == "string"

    def test_gemini_input_form(self):
        tool = self._make_tool()
        gemini = tool.gemini_input_form()
        # Should be a types.Tool with one FunctionDeclaration
        assert hasattr(gemini, "function_declarations")
        assert len(gemini.function_declarations) == 1
        fd = gemini.function_declarations[0]
        assert fd.name == "search"
        assert fd.description == "Search the web"

    # -- execute: text response ---------------------------------------------

    async def test_execute_text_response(self):
        conn = _make_mock_connection()
        conn.call_tool.return_value = _call_result([_text_content("result text")])

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id1", query="hello")

        conn.call_tool.assert_awaited_once_with("search", {"query": "hello"})
        assert isinstance(result, ToolResultBlock)
        assert result.tool_use_id == "id1"
        assert result.is_error is False
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "result text"

    # -- execute: image response --------------------------------------------

    async def test_execute_image_response(self):
        conn = _make_mock_connection()
        conn.call_tool.return_value = _call_result([
            _image_content("image/png", "base64data"),
        ])

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id2", query="cat photo")

        assert result.is_error is False
        assert len(result.content) == 1
        assert isinstance(result.content[0], ImageBlock)
        assert result.content[0].source.data == "base64data"

    # -- execute: MCP error flag --------------------------------------------

    async def test_execute_error_response(self):
        conn = _make_mock_connection()
        conn.call_tool.return_value = _call_result(
            [_text_content("bad request")], is_error=True,
        )

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id3", query="bad")

        assert result.is_error is True
        assert isinstance(result.content, list)
        assert result.content[0].text == "bad request"

    # -- execute: connection exception --------------------------------------

    async def test_execute_connection_error(self):
        conn = _make_mock_connection()
        conn.call_tool.side_effect = RuntimeError("server crashed")

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id4", query="oops")

        assert result.is_error is True
        assert isinstance(result.content, str)
        assert "server crashed" in result.content

    # -- execute: empty content from MCP ------------------------------------

    async def test_execute_empty_content(self):
        conn = _make_mock_connection()
        conn.call_tool.return_value = _call_result([])

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id5", query="empty")

        assert result.is_error is False
        # Should fall back to empty TextBlock
        assert len(result.content) == 1
        assert result.content[0].text == ""

    # -- execute: null content from MCP -------------------------------------

    async def test_execute_null_content(self):
        conn = _make_mock_connection()
        conn.call_tool.return_value = SimpleNamespace(content=None, isError=False)

        tool = self._make_tool(conn)
        result = await tool.execute(tool_use_id="id6", query="null")

        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0].text == ""

    # -- repr ---------------------------------------------------------------

    def test_repr(self):
        tool = self._make_tool()
        assert "MCPTool" in repr(tool)
        assert "search" in repr(tool)


# ---------------------------------------------------------------------------
# MCPTool in Agent loop (end-to-end with QueueLLM)
# ---------------------------------------------------------------------------

class TestMCPToolInAgentLoop:
    async def test_agent_calls_mcp_tool(self):
        """MCPTool works inside an Agent loop alongside StopTool."""
        conn = _make_mock_connection()
        conn.call_tool.return_value = _call_result([_text_content("MCP answer")])

        mcp_tool = MCPTool(
            name="lookup",
            description="Look things up",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            connection=conn,
        )

        # LLM calls lookup first, then stop
        responses = [
            Message(
                role=MessageRole.ASSISTANT,
                content=[ToolUseBlock(id="tc1", name="lookup", input={"q": "test"})],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[ToolUseBlock(id="tc2", name="stop", input={})],
            ),
        ]

        llm = QueueLLM(responses)
        memory = TrackingMemory()

        agent = Agent(
            llm=llm,
            system_prompt="You are helpful.",
            max_calls=5,
            max_tokens=4096,
            memory=memory,
            agent_name="mcp-test",
            tools=[mcp_tool, StopTool()],
        )

        result = await agent.run_async("test query")

        # The MCP tool should have been invoked
        conn.call_tool.assert_awaited_once_with("lookup", {"q": "test"})


# ---------------------------------------------------------------------------
# disconnect_mcp_tools
# ---------------------------------------------------------------------------

class TestDisconnectMCPTools:
    async def test_disconnects_unique_connections(self):
        conn1 = _make_mock_connection()
        conn2 = _make_mock_connection()

        tools = [
            MCPTool(name="a", description="", input_schema={}, connection=conn1),
            MCPTool(name="b", description="", input_schema={}, connection=conn1),
            MCPTool(name="c", description="", input_schema={}, connection=conn2),
        ]

        await disconnect_mcp_tools(tools)

        conn1.disconnect.assert_awaited_once()
        conn2.disconnect.assert_awaited_once()

    async def test_ignores_non_mcp_tools(self):
        """Non-MCPTool instances are silently skipped."""
        stop = StopTool()
        conn = _make_mock_connection()
        mcp = MCPTool(name="x", description="", input_schema={}, connection=conn)

        await disconnect_mcp_tools([stop, mcp])

        conn.disconnect.assert_awaited_once()

    async def test_empty_list(self):
        """No error when called with an empty list."""
        await disconnect_mcp_tools([])

    async def test_accepts_raw_connection(self):
        """MCPServerConnection instances can be passed directly."""
        conn = _make_mock_connection()
        await disconnect_mcp_tools([conn])
        conn.disconnect.assert_awaited_once()

    async def test_deduplicates_raw_and_tool_connections(self):
        """Same connection passed both directly and via MCPTool is only disconnected once."""
        conn = _make_mock_connection()
        mcp = MCPTool(name="x", description="", input_schema={}, connection=conn)
        await disconnect_mcp_tools([conn, mcp])
        conn.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# connect_mcp_server (mocked subprocess)
# ---------------------------------------------------------------------------

class TestConnectMCPServer:
    async def test_returns_mcp_tools(self):
        mock_tool = SimpleNamespace(
            name="resolve_library_id",
            description="Resolve a library ID",
            inputSchema={
                "type": "object",
                "properties": {"libraryName": {"type": "string"}},
                "required": ["libraryName"],
            },
        )

        mock_conn = _make_mock_connection()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(return_value=[mock_tool])

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            tools = await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        assert len(tools) == 1
        assert isinstance(tools[0], MCPTool)
        assert tools[0].name == "resolve_library_id"
        assert tools[0].description == "Resolve a library ID"
        assert tools[0]._connection is mock_conn

    async def test_disconnects_on_list_tools_failure(self):
        """If list_tools() raises, the connection is cleaned up and the error re-raised."""
        mock_conn = _make_mock_connection()
        mock_conn.connect = AsyncMock(return_value=mock_conn)
        mock_conn.list_tools = AsyncMock(side_effect=RuntimeError("protocol error"))

        with patch("CAL.mcp.MCPServerConnection", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="protocol error"):
                await connect_mcp_server(command="npx", args=["-y", "@ctx/mcp"])

        mock_conn.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# MCPServerConnection (unit-level — no real subprocess)
# ---------------------------------------------------------------------------

class TestMCPServerConnection:
    def test_init(self):
        conn = MCPServerConnection(command="npx", args=["-y", "pkg"])
        assert conn._session is None

    async def test_list_tools_before_connect_raises(self):
        conn = MCPServerConnection(command="npx")
        with pytest.raises(RuntimeError, match="not connected"):
            await conn.list_tools()

    async def test_call_tool_before_connect_raises(self):
        conn = MCPServerConnection(command="npx")
        with pytest.raises(RuntimeError, match="not connected"):
            await conn.call_tool("foo", {})


# ---------------------------------------------------------------------------
# Integration test — requires real MCP server (npx + internet)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestContext7MCPIntegration:
    async def test_connect_and_list_tools(self):
        """Connect to the Context7 MCP server and verify tools are discovered."""
        tools = await connect_mcp_server(command="npx", args=["-y", "@upstash/context7-mcp"])
        try:
            assert len(tools) > 0
            # All items should be MCPTool instances
            for t in tools:
                assert isinstance(t, MCPTool)
                assert t.name
                assert t.description
        finally:
            await disconnect_mcp_tools(tools)
