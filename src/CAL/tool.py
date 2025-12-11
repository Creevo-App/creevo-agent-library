"""Tool classes for CAL."""

import inspect
from typing import Callable, Union, get_type_hints, get_origin, get_args
from abc import ABC, abstractmethod
from google.genai import types
from .content_blocks import ToolResultBlock, TextBlock, ImageBlock

class Tool(ABC):
    """Abstract base class for tools that wrap callable functions."""
    
    def __init__(self, function: Callable):
        self.function = function
        self.name = function.__name__
        self.description = inspect.getdoc(function) or ""
        self.input_schema = self._generate_schema()
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResultBlock:
        """Execute the tool. Must be implemented by subclasses."""
        pass
    
    def _generate_schema(self) -> dict:
        """Generate JSON schema for function parameters from type hints."""
        sig = inspect.signature(self.function)
        type_hints = get_type_hints(self.function)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, type(None))
            param_schema = self._type_to_schema(param_type)
            
            # Add description from docstring if available
            properties[param_name] = param_schema
            
            # If no default value, it's required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _type_to_schema(self, python_type) -> dict:
        """Convert Python type hints to JSON schema types."""
        origin = get_origin(python_type)
        
        type_mapping = {
            int: {"type": "integer"},
            str: {"type": "string"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        if origin is list:
            args = get_args(python_type)
            if args:
                return {"type": "array", "items": self._type_to_schema(args[0])}
            return {"type": "array"}
        
        if origin is Union:
            args = get_args(python_type)
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return self._type_to_schema(non_none_types[0])
        
        return type_mapping.get(python_type, {"type": "string"})
    
    def get_schema(self) -> dict:
        """Return schema in Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
    
    def gemini_input_form(self):
        """Convert tool schema to Gemini format."""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=self.input_schema
                )
            ]
        )
    
    def __repr__(self):
        return f"Tool(name={self.name})"


class _ToolImpl(Tool):
    """Concrete implementation of Tool for use with the @tool decorator."""
    
    async def execute(self, **kwargs) -> ToolResultBlock:
        """Execute the wrapped function and return a ToolResultBlock."""
        tool_use_id = kwargs.pop('tool_use_id', 'stub_tool_use_id')
        
        try:
            result = await self.function(**kwargs)
            
            # Tools must return: {"content": [...], "metadata": {...}}
            if not isinstance(result, dict) or "content" not in result:
                raise ValueError(
                    f"Tool {self.name} must return a dict with 'content' and 'metadata' keys. "
                    f"Example: {{'content': [{{'type': 'text', 'text': '...'}}], 'metadata': {{}}}}"
                )
            
            metadata = result.get("metadata", {})
            content = result["content"]
            
            content_blocks = []
            for item in content:
                if item["type"] == "text":
                    content_blocks.append(TextBlock(text=item["text"]))
                elif item["type"] == "image":
                    content_blocks.append(ImageBlock(source=item["data"]))
            
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=content_blocks,
                is_error=False,
                name=self.name,
                metadata=metadata
            )
        except Exception as e:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Error executing {self.name}: {str(e)}",
                is_error=True,
                name=self.name
            )


def tool(func: Callable) -> Tool:
    """
    Decorator to convert an async function into a Tool.
    
    Automatically extracts function name, docstring, and type hints as the tool schema.
    
    Example:
        @tool
        async def my_async_function(param1: int, param2: str):
            '''Description of my function'''
            await asyncio.sleep(0.5)
            return {
                "content": [{"type": "text", "text": f"Result: {param1}, {param2}"}],
                "metadata": {}
            }
    
    Args:
        func: The async function to wrap as a tool
        
    Returns:
        Tool instance
    """
    return _ToolImpl(func)


class StopTool(Tool):
    """Special tool that signals the agent to stop"""
    
    def __init__(self):
        """Initialize the stop tool"""
        def stop_function():
            """Stop the agent execution"""
            return "STOP"
        
        super().__init__(stop_function)
        self.name = "stop"
        self.description = "Stop the agent execution"
    
    async def execute(self, **kwargs) -> ToolResultBlock:
        """
        Execute the stop tool.
        
        Returns:
            "STOP" string
        """
        return ToolResultBlock(
            tool_use_id=kwargs.pop('tool_use_id', 'stub_tool_use_id'),
            content="STOP",
            is_error=False,
            name=self.name
        )

