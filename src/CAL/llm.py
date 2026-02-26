"""
LLM classes for CAL
"""

from abc import ABC, abstractmethod # This is a python import for abstract base class (abc)
from typing import List, Optional
from .message import Message, MessageRole
from .content_blocks import TextBlock, ThinkingBlock, ToolUseBlock
import os
import sys
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from anthropic import AnthropicVertex
from dotenv import load_dotenv
from .tool import Tool

load_dotenv()

#This is the abstract base class for LLMs, it can not be instantiated directly, but must be subclassed
# Here's a good link on it: https://www.geeksforgeeks.org/python/abstract-classes-in-python/
class LLM(ABC):
    """Abstract base class for LLMs"""
    
    def __init__(self, max_tokens: int, name: str, provider: str):
        """
        Initialize the LLM.
        
        Args:
            max_tokens: Maximum number of tokens for generation
            name: Name of the LLM
            provider: Provider of the LLM
        """
        self.max_tokens = max_tokens
        self.name = name
        self.provider = provider
    @abstractmethod
    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        """
        Generate content based on conversation history.
        
        Args:
            system_prompt: System prompt for the LLM
            conversation_history: List of messages in the conversation
            tools: Optional list of tools available to the LLM
            
        Returns:
            Generated message
        """
        pass


class AnthropicVertexLLM(LLM):
    """Anthropic Vertex LLM implementation"""
    
    def __init__(self, project_id: str, region: str, model: str, max_tokens: int):
        """
        Initialize the Anthropic Vertex LLM.
        
        Args:
            project_id: GCP project ID
            region: Vertex AI region (e.g., "global", "us-east5")
            model: Model name to use (e.g., "claude-opus-4-5@20251101")
            max_tokens: Maximum number of tokens for generation
        """
        super().__init__(max_tokens, name=model, provider="Anthropic")
        self.project_id = project_id
        self.region = region
        self.model = model
        self.client = AnthropicVertex(project_id=project_id, region=region)
    
    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        """
        Generate content based on conversation history.
        
        Args:
            system_prompt: System prompt for the LLM
            conversation_history: List of messages in the conversation
            tools: Optional list of tools available to the LLM
            
        Returns:
            Generated message from Anthropic Vertex API
        """
        # Format the conversation history for Anthropic API
        formatted_history = []
        for message in conversation_history:
            parts = []
            if isinstance(message.content, str):
                parts.append({"type": "text", "text": message.content})
            else:
                for block in message.content:
                    parts.append(block.claude_content_form())
            
            # Map CAL roles to Anthropic roles
            if message.role == MessageRole.USER:
                role = "user"
            elif message.role == MessageRole.ASSISTANT:
                role = "assistant"
            elif message.role == MessageRole.TOOL_RESPONSE:
                role = "user"  # Tool results are sent as user messages in Anthropic
            else:
                role = "user"
            
            # Merge with previous message if role is the same
            if formatted_history and formatted_history[-1]["role"] == role:
                formatted_history[-1]["content"].extend(parts)
            else:
                formatted_history.append({"role": role, "content": parts})

        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": formatted_history,
        }
        
        # Add tools if provided
        if tools:
            request_params["tools"] = [tool.get_schema() for tool in tools]

        if os.getenv("DEBUG_LLM_RESPONSE") == "true":
            print(f"[LLM] Requesting model: {self.model}", file=sys.stderr)

        response = self.client.messages.create(**request_params)

        if os.getenv("DEBUG_LLM_RESPONSE") == "true":
            print(f"[LLM] Response: {response}", file=sys.stderr)

        # Parse response content blocks
        content_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        # Extract token usage
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                "total_tokens": (getattr(response.usage, "input_tokens", 0) or 0) + (getattr(response.usage, "output_tokens", 0) or 0),
            }

        # Extract finish reason (convert to uppercase to match expected format)
        raw_stop_reason = getattr(response, "stop_reason", None)
        finish_reason = raw_stop_reason.upper() if raw_stop_reason else None

        if os.getenv("DEBUG_LLM_RESPONSE") == "true":
            print(f"Usage metadata: {usage}", file=sys.stderr)
            print(f"Finish reason: {finish_reason}", file=sys.stderr)

        return Message(
            role=MessageRole.ASSISTANT,
            content=content_blocks,
            usage=usage,
            metadata={"finish_reason": finish_reason} if finish_reason else {},
        )
    
class GeminiLLM(LLM):
    """Gemini LLM implementation"""
    
    def __init__(self, api_key: str, model: str, max_tokens: int, timeout_ms: int = 20_000, thinking_level: Optional[str] = None):
        """
        Initialize the Gemini Vertex LLM.

        Args:
            api_key: API key for authentication
            model: Model name to use
            max_tokens: Maximum number of tokens for generation
            timeout_ms: HTTP timeout in milliseconds (default: 20000 = 20 seconds)
            thinking_level: Optional thinking level for Gemini 3.x models.
                Valid values: "minimal", "low", "medium", "high"
        """
        super().__init__(max_tokens, name=model, provider="Gemini")
        if not api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        else:
            self.api_key = api_key

        self.model = model
        self.thinking_level = thinking_level
        # Initialize client once to avoid duplicate API key warnings
        self.client = genai.Client(http_options=HttpOptions(api_version="v1alpha", timeout=timeout_ms))
    
    def generate_content(self, system_prompt: str, conversation_history: List[Message], tools: Optional[List[Tool]] = None) -> Message:
        """
        Generate content based on conversation history.
        
        Args:
            system_prompt: System prompt for the LLM
            conversation_history: List of messages in the conversation
            tools: Optional list of tools available to the LLM
            
        Returns:
            Generated message from Gemini API
        """
        # Format the conversation history for Gemini API
        formatted_history = []
        for message in conversation_history:
            parts = []
            if isinstance(message.content, str):
                parts.append(TextBlock(message.content).gemini_content_form())
            else:
                for block in message.content:
                    gemini_block = block.gemini_content_form()
                    parts.extend(gemini_block if isinstance(gemini_block, list) else [gemini_block])
            
            role = message.role.value
            if role == "tool response":
                role = "user"
            elif role == "assistant":
                role = "model"
            
            # Merge with previous message if role is the same
            if formatted_history and formatted_history[-1]['role'] == role:
                formatted_history[-1]['parts'].extend(parts)
            else:
                formatted_history.append({'role': role, 'parts': parts})

        # Prepare config parameters
        config_params = {
            'system_instruction': system_prompt,
        }
        
        # Add thinking config if set
        if self.thinking_level:
            config_params['thinking_config'] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=types.ThinkingLevel(self.thinking_level.upper()),
            )

        # Add tools if provided (convert to Gemini format)
        if tools:
            gemini_tools = []
            for tool in tools:
                gemini_tools.append(tool.gemini_input_form())
            config_params['tools'] = gemini_tools

        # Generate a response from the Gemini model using the formatted conversation
        if os.getenv("DEBUG_LLM_RESPONSE") == "true":
            print(f"[LLM] Requesting model: {self.model}", file=sys.stderr)
        response = self.client.models.generate_content(
            model=self.model,
            contents=formatted_history,
            config=types.GenerateContentConfig(**config_params)
        )
        if os.getenv("DEBUG_LLM_RESPONSE") == "true" and hasattr(response, 'model_version'):
            print(f"[LLM] Response model_version: {response.model_version}", file=sys.stderr)

        content_blocks = []
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        if getattr(part, 'thought', False):
                            content_blocks.append(ThinkingBlock(
                                text=part.text,
                                thought_signature=getattr(part, 'thought_signature', None),
                            ))
                        else:
                            content_blocks.append(TextBlock(text=part.text))
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        content_blocks.append(
                            ToolUseBlock(
                                id=fc.id if hasattr(fc, 'id') else fc.name,
                                name=fc.name,
                                input=dict(fc.args),
                                thought=getattr(part, 'thought', None),
                                thought_signature=getattr(part, 'thought_signature', None)
                            )
                        )
        
        # Extract token usage from response
        usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
                'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0) or 0,
            }
        
        # Extract finish reason
        finish_reason = None
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)
        
        # Log usage metadata (debug info)
        if os.getenv("DEBUG_LLM_RESPONSE") == "true":
            print(f"Full response: {response}", file=sys.stderr)
            print(f"Usage metadata: {usage}", file=sys.stderr)
            print(f"Finish reason: {finish_reason}", file=sys.stderr)
        
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content_blocks,
            usage=usage,
            metadata={'finish_reason': finish_reason} if finish_reason else {},
        )
        
        return message