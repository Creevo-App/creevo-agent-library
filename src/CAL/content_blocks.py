"""
Content block classes for CAL
"""
import base64
from typing import Union, List
from abc import ABC, abstractmethod
from google.genai import types


class ContentBlock(ABC):
    """Abstract base class for content blocks"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def claude_content_form(self):
        pass

    @abstractmethod
    def gemini_content_form(self):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def clone(self) -> 'ContentBlock':
        """Create a deep copy of this content block."""
        pass

    @staticmethod
    def from_dict(data: dict) -> 'ContentBlock':
        block_type = data.get("type")
        if block_type == "text":
            return TextBlock.from_dict(data)
        elif block_type == "thinking":
            return ThinkingBlock.from_dict(data)
        elif block_type == "image":
            return ImageBlock.from_dict(data)
        elif block_type == "tool_use":
            return ToolUseBlock.from_dict(data)
        elif block_type == "tool_result":
            return ToolResultBlock.from_dict(data)
        return TextBlock(text="")

class TextBlock(ContentBlock):
    """Text content block"""
    
    def __init__(self, text: str):
        """
        Initialize a text block.
        
        Args:
            text: The text content
        """
        super().__init__()
        self.type = "text"
        self.text = text
    
    def __repr__(self):
        return f"TextBlock(text={self.text!r})"
    
    def claude_content_form(self):
        return {'type': self.type, 'text': self.text}
    
    def gemini_content_form(self):
        return types.Part.from_text(text=self.text)

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}

    @staticmethod
    def from_dict(data: dict) -> 'TextBlock':
        return TextBlock(text=data.get("text", ""))

    def clone(self) -> 'TextBlock':
        """Create a deep copy of this text block."""
        return TextBlock(text=self.text)


class ImageSource:
    """Image source metadata"""
    
    def __init__(self, type: str, media_type: str, data: str | bytes):
        """
        Initialize an image source.
        
        Args:
            type: The type of image source (e.g., "base64")
            media_type: The media type (e.g., "image/png")
            data: The image data
        """
        self.type = type
        self.media_type = media_type
        self.data = data
    
    def __repr__(self):
        return f"ImageSource(type={self.type}, media_type={self.media_type})"
    
    def claude_content_form(self):
        return {
            'type': self.type,
            'media_type': self.media_type,
            'data': self.data
        }
    
    def gemini_content_form(self):
        if isinstance(self.data, bytes):
            image_base64 = base64.b64encode(self.data).decode('utf-8')
        else:
            image_base64 = self.data
        
        return types.Part(
            inline_data=types.Blob(
                mime_type=self.media_type,
                data=image_base64
            )
        )

    def to_dict(self) -> dict:
        data = self.data
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode("utf-8")
        return {
            "type": self.type,
            "media_type": self.media_type,
            "data": data,
        }

    @staticmethod
    def from_dict(data: dict) -> 'ImageSource':
        image_data = data.get("data", "")
        # If data is a base64 string and type is base64, decode it back to bytes
        if data.get("type") == "base64" and isinstance(image_data, str):
            try:
                image_data = base64.b64decode(image_data)
            except Exception:
                pass

        return ImageSource(
            type=data.get("type", "base64"),
            media_type=data.get("media_type", "image/png"),
            data=image_data,
        )

    def clone(self) -> 'ImageSource':
        """Create a deep copy of this image source."""
        return ImageSource(type=self.type, media_type=self.media_type, data=self.data)


class ImageBlock(ContentBlock):
    """Image content block"""
    
    def __init__(self, source: ImageSource):
        """
        Initialize an image block.
        
        Args:
            source: The image source
        """
        super().__init__()
        self.type = "image"
        self.source = source
    
    def __repr__(self):
        return f"ImageBlock(source={self.source})"
    
    def claude_content_form(self):
        return {'type': self.type, 'source': self.source.claude_content_form()}
    
    def gemini_content_form(self):
        return self.source.gemini_content_form()

    def to_dict(self) -> dict:
        return {
            "type": "image",
            "source": self.source.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> 'ImageBlock':
        source_data = data.get("source") or {}
        return ImageBlock(source=ImageSource.from_dict(source_data))

    def clone(self) -> 'ImageBlock':
        """Create a deep copy of this image block."""
        return ImageBlock(source=self.source.clone())


class ThinkingBlock(ContentBlock):
    """Thinking/reasoning content block (Gemini thinking mode)"""

    def __init__(self, text: str):
        super().__init__()
        self.type = "thinking"
        self.text = text

    def __repr__(self):
        return f"ThinkingBlock(text={self.text!r})"

    def claude_content_form(self):
        return {'type': 'thinking', 'text': self.text}

    def gemini_content_form(self):
        return types.Part(text=self.text, thought=True)

    def to_dict(self) -> dict:
        return {"type": "thinking", "text": self.text}

    @staticmethod
    def from_dict(data: dict) -> 'ThinkingBlock':
        return ThinkingBlock(text=data.get("text", ""))

    def clone(self) -> 'ThinkingBlock':
        return ThinkingBlock(text=self.text)


class ToolUseBlock(ContentBlock):
    """Tool use content block"""
    
    def __init__(self, id: str, name: str, input: dict, thought: str = None, thought_signature: str = None):
        """
        Initialize a tool use block.
        
        Args:
            id: Unique identifier for the tool use
            name: Name of the tool
            input: Input parameters for the tool
            thought: Chain of thought reasoning (Gemini specific)
            thought_signature: Signature for the thought (Gemini specific)
        """
        super().__init__()
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
        self.thought = thought
        self.thought_signature = thought_signature
    
    def __repr__(self):
        return f"ToolUseBlock(id={self.id}, name={self.name}, input={self.input}, thought={self.thought})"
    
    def claude_content_form(self):
        return {'type': self.type, 'id': self.id, 'name': self.name, 'input': self.input}
    
    def gemini_content_form(self):
        part_args = {
            "function_call": types.FunctionCall(
                name=self.name,
                args=self.input
            )
        }
        if self.thought:
            part_args["thought"] = self.thought
        if self.thought_signature:
            part_args["thought_signature"] = self.thought_signature
            
        return types.Part(**part_args)

    def to_dict(self) -> dict:
        thought_signature = self.thought_signature
        if isinstance(thought_signature, bytes):
            thought_signature = base64.b64encode(thought_signature).decode('utf-8')
            
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "thought": self.thought,
            "thought_signature": thought_signature,
        }

    @staticmethod
    def from_dict(data: dict) -> 'ToolUseBlock':
        thought_signature = data.get("thought_signature")
        if thought_signature and isinstance(thought_signature, str):
            try:
                thought_signature = base64.b64decode(thought_signature)
            except Exception:
                pass

        return ToolUseBlock(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input") or {},
            thought=data.get("thought"),
            thought_signature=thought_signature,
        )

    def clone(self) -> 'ToolUseBlock':
        """Create a deep copy of this tool use block."""
        import copy
        return ToolUseBlock(
            id=self.id,
            name=self.name,
            input=copy.deepcopy(self.input),
            thought=self.thought,
            thought_signature=self.thought_signature,
        )


class ToolResultBlock(ContentBlock):
    """Tool result content block"""
    
    def __init__(self, tool_use_id: str, content: Union[str, List[ContentBlock]], is_error: bool = False, name: str = None, metadata: dict = None):
        super().__init__()
        self.type = "tool_result"
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error
        self.name = name
        self.metadata = metadata
    
    def __repr__(self):
        return f"ToolResultBlock(tool_use_id={self.tool_use_id}, is_error={self.is_error}, name={self.name})"
    
    def claude_content_form(self):
        result = {
            'type': self.type,
            'tool_use_id': self.tool_use_id,
            'content': self.content if isinstance(self.content, str) else [block.claude_content_form() for block in self.content]
        }
        if self.is_error:
            result['is_error'] = True
        return result

    def gemini_content_form(self) -> types.Part:
        response_data = {}
        inline_data = {}
        if isinstance(self.content, str):
            response_data = {"result": self.content}
        else:
            result = []
            for block in self.content:
                #TODO: Unify this, both classes should have a gemini_content_form method
                if isinstance(block, ImageBlock):
                    inline_data = block.gemini_content_form()
                elif isinstance(block, TextBlock):
                    result.append(block.text)
                else:
                    result.append(str(block))
            response_data = {"result": result}
            
        if self.is_error:
            response_data["error"] = True
        parts = [types.Part(
            function_response=types.FunctionResponse(
                name=self.name or self.tool_use_id,
                response=response_data
            )
        )]
        if inline_data:
            parts.append(inline_data)
        return parts

    def to_dict(self) -> dict:
        content = self.content
        if isinstance(content, list):
            content = [child.to_dict() for child in content]
        
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": content,
            "is_error": self.is_error,
            "name": self.name,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> 'ToolResultBlock':
        content = data.get("content")
        if isinstance(content, list):
            content = [ContentBlock.from_dict(item) for item in content]

        return ToolResultBlock(
            tool_use_id=data.get("tool_use_id", ""),
            content=content,
            is_error=bool(data.get("is_error")),
            name=data.get("name"),
            metadata=data.get("metadata"),
        )

    def clone(self) -> 'ToolResultBlock':
        """Create a deep copy of this tool result block."""
        import copy
        if isinstance(self.content, str):
            cloned_content = self.content
        else:
            cloned_content = [block.clone() for block in self.content]
        return ToolResultBlock(
            tool_use_id=self.tool_use_id,
            content=cloned_content,
            is_error=self.is_error,
            name=self.name,
            metadata=copy.deepcopy(self.metadata) if self.metadata else None,
        )
