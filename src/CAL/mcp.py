"""MCP (Model Context Protocol) client support for CAL."""

from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google.genai import types

from .tool import Tool
from .content_blocks import (
    ContentBlock,
    ImageBlock,
    ImageSource,
    TextBlock,
    ToolResultBlock,
)


def _map_mcp_content(mcp_content: list) -> List[ContentBlock]:
    """Map MCP result content items to CAL ContentBlock instances."""
    blocks: List[ContentBlock] = []
    for item in mcp_content:
        if item.type == "text":
            blocks.append(TextBlock(text=item.text))
        elif item.type == "image":
            blocks.append(ImageBlock(source=ImageSource(
                type="base64", media_type=item.mimeType, data=item.data,
            )))
        elif item.type == "resource":
            resource = item.resource
            text = getattr(resource, "text", None)
            blocks.append(TextBlock(text=text or f"[Resource: {resource.uri}]"))
        else:
            blocks.append(TextBlock(text=str(item)))
    return blocks


class MCPServerConnection:
    """Manages the lifecycle of a single MCP server subprocess."""

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self._server_params = StdioServerParameters(
            command=command, args=args or [], env=env,
        )
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPServerConnection is not connected. Call connect() first.")
        return self._session

    async def connect(self) -> "MCPServerConnection":
        """Start the server subprocess and initialise the MCP session."""
        try:
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(self._server_params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
        except Exception:
            await self._exit_stack.aclose()
            self._session = None
            raise
        return self

    async def list_tools(self) -> list:
        result = await self._require_session().list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict) -> object:
        return await self._require_session().call_tool(name, arguments)

    async def disconnect(self) -> None:
        await self._exit_stack.aclose()
        self._session = None


class MCPToolList(list):
    """A list of MCPTool instances that retains a reference to their connection.

    This allows ``disconnect_mcp_tools`` to clean up even when the server
    returns zero tools.
    """

    def __init__(self, tools: list, connection: "MCPServerConnection"):
        super().__init__(tools)
        self._connection = connection


class MCPTool(Tool):
    """Wraps a single MCP server tool as a CAL ``Tool``.

    Follows the ``SubAgentTool`` pattern: bypasses ``Tool.__init__``
    and implements the interface directly.
    """

    def __init__(self, name: str, description: str, input_schema: dict,
                 connection: MCPServerConnection):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._connection = connection

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def gemini_input_form(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=self.name,
                description=self.description,
                parameters=self.input_schema,
            )
        ])

    async def execute(self, **kwargs) -> ToolResultBlock:
        tool_use_id = kwargs.pop("tool_use_id", "stub_tool_use_id")
        try:
            result = await self._connection.call_tool(self.name, kwargs)
            blocks = _map_mcp_content(result.content) if result.content else [TextBlock(text="")]
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=blocks,
                is_error=result.isError,
                name=self.name,
            )
        except Exception as e:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Error executing {self.name}: {e}",
                is_error=True,
                name=self.name,
            )

    def __repr__(self):
        return f"MCPTool(name={self.name})"


async def connect_mcp_server(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> MCPToolList:
    """Connect to an MCP server and return its tools as ``MCPTool`` instances.

    Returns an ``MCPToolList`` which can be iterated like a normal list but
    also exposes ``_connection`` for cleanup via ``disconnect_mcp_tools``.
    """
    connection = MCPServerConnection(command=command, args=args, env=env)
    await connection.connect()
    try:
        mcp_tools_raw = await connection.list_tools()
    except Exception:
        await connection.disconnect()
        raise
    tools = [
        MCPTool(
            name=t.name,
            description=t.description or "",
            input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
            connection=connection,
        )
        for t in mcp_tools_raw
    ]
    return MCPToolList(tools, connection)


async def disconnect_mcp_tools(tools: list) -> None:
    """Disconnect all unique MCP server connections found in *tools*.

    Accepts ``MCPTool``, ``MCPServerConnection``, ``MCPToolList``, or any
    ``Tool`` (non-MCP items are silently ignored).
    """
    seen: set = set()

    # If tools is an MCPToolList, extract its connection first
    if isinstance(tools, MCPToolList):
        conn = tools._connection
        if isinstance(conn, MCPServerConnection) and id(conn) not in seen:
            seen.add(id(conn))
            await conn.disconnect()

    for t in tools:
        conn = t if isinstance(t, MCPServerConnection) else getattr(t, "_connection", None)
        if isinstance(conn, MCPServerConnection) and id(conn) not in seen:
            seen.add(id(conn))
            await conn.disconnect()
