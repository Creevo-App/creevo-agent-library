"""MCP (Model Context Protocol) client support for CAL.

Enables CAL agents to connect to external MCP servers and use their tools.
MCPTool integrates seamlessly into the existing tool system — no changes
to Agent, LLM, or the core loop are required.
"""

from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tool import Tool
from .content_blocks import (
    ContentBlock,
    ImageBlock,
    ImageSource,
    TextBlock,
    ToolResultBlock,
)

from google.genai import types


def _map_mcp_content(mcp_content: list) -> List[ContentBlock]:
    """Map MCP result content items to CAL ContentBlock instances.

    Handles text, image, and resource content types from the MCP SDK,
    with a string-fallback for unknown types.
    """
    blocks: List[ContentBlock] = []
    for item in mcp_content:
        if item.type == "text":
            blocks.append(TextBlock(text=item.text))
        elif item.type == "image":
            blocks.append(
                ImageBlock(
                    source=ImageSource(
                        type="base64",
                        media_type=item.mimeType,
                        data=item.data,
                    )
                )
            )
        elif item.type == "resource":
            resource = item.resource
            if hasattr(resource, "text") and resource.text:
                blocks.append(TextBlock(text=resource.text))
            else:
                blocks.append(TextBlock(text=f"[Resource: {resource.uri}]"))
        else:
            blocks.append(TextBlock(text=str(item)))
    return blocks


class MCPServerConnection:
    """Manages the lifecycle of a single MCP server subprocess.

    Uses ``AsyncExitStack`` to manage the stdio transport and
    ``ClientSession`` exactly as recommended by the MCP SDK.
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def connect(self) -> "MCPServerConnection":
        """Start the server subprocess and initialise the MCP session.

        Returns ``self`` for convenient chaining.
        """
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(self._server_params)
        )
        read_stream, write_stream = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        return self

    async def list_tools(self) -> list:
        """Return the raw MCP tool definitions from the server."""
        if self._session is None:
            raise RuntimeError("MCPServerConnection is not connected. Call connect() first.")
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict) -> object:
        """Invoke a tool on the remote server and return the raw MCP result."""
        if self._session is None:
            raise RuntimeError("MCPServerConnection is not connected. Call connect() first.")
        return await self._session.call_tool(name, arguments)

    async def disconnect(self) -> None:
        """Tear down the session and server subprocess."""
        await self._exit_stack.aclose()
        self._session = None


class MCPTool(Tool):
    """Wraps a single MCP server tool as a CAL ``Tool``.

    Follows the same pattern as ``SubAgentTool``: bypasses
    ``Tool.__init__`` and implements ``get_schema`` / ``gemini_input_form``
    / ``execute`` directly.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        connection: "MCPServerConnection",
    ):
        # Bypass Tool.__init__ (same pattern as SubAgentTool)
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._connection = connection

    # -- Tool interface -----------------------------------------------------

    def get_schema(self) -> dict:
        """Return schema in Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def gemini_input_form(self):
        """Convert tool schema to Gemini format."""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=self.input_schema,
                )
            ]
        )

    async def execute(self, **kwargs) -> ToolResultBlock:
        """Call the MCP server tool and return a ``ToolResultBlock``."""
        tool_use_id = kwargs.pop("tool_use_id", "stub_tool_use_id")

        try:
            result = await self._connection.call_tool(self.name, kwargs)

            if result.isError:
                content = _map_mcp_content(result.content) if result.content else f"MCP tool '{self.name}' returned an error"
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=content,
                    is_error=True,
                    name=self.name,
                )

            blocks = _map_mcp_content(result.content) if result.content else [TextBlock(text="")]
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=blocks,
                is_error=False,
                name=self.name,
            )
        except Exception as e:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Error executing MCP tool '{self.name}': {e}",
                is_error=True,
                name=self.name,
            )

    def __repr__(self):
        return f"MCPTool(name={self.name})"


# ---------------------------------------------------------------------------
# User-facing helpers
# ---------------------------------------------------------------------------

async def connect_mcp_server(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[MCPTool]:
    """Connect to an MCP server and return its tools as ``MCPTool`` instances.

    All returned tools share the same underlying ``MCPServerConnection``.

    Args:
        command: The command to start the MCP server (e.g. ``"npx"``).
        args: Command-line arguments (e.g. ``["-y", "@context7/mcp"]``).
        env: Optional environment variables for the subprocess.

    Returns:
        A list of ``MCPTool`` instances ready to be passed to an ``Agent``.
    """
    connection = MCPServerConnection(command=command, args=args, env=env)
    await connection.connect()

    try:
        mcp_tools_raw = await connection.list_tools()
    except Exception:
        await connection.disconnect()
        raise

    tools: List[MCPTool] = []
    for t in mcp_tools_raw:
        tools.append(
            MCPTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                connection=connection,
            )
        )
    return tools


async def disconnect_mcp_tools(tools: List[Tool]) -> None:
    """Disconnect all unique MCP server connections found in *tools*.

    Safe to call with a mixed list of ``MCPTool``, ``MCPServerConnection``,
    and regular ``Tool`` instances — non-MCP items are silently ignored.
    Passing ``MCPServerConnection`` instances directly is useful when the
    tool list may be empty (e.g. a server that exposes zero tools).
    """
    seen_connections: set = set()
    for t in tools:
        conn: Optional[MCPServerConnection] = None
        if isinstance(t, MCPServerConnection):
            conn = t
        elif isinstance(t, MCPTool):
            conn = t._connection
        if conn is not None and id(conn) not in seen_connections:
            seen_connections.add(id(conn))
            await conn.disconnect()
