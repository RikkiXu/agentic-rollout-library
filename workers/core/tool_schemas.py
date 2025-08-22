#!/usr/bin/env python3
"""
Tool schemas for the agentic rollout library.
Compatible with both VERL and standalone usage.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import json


class OpenAIFunctionParameters(BaseModel):
    """OpenAI function parameters schema."""
    type: str = "object"
    properties: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class OpenAIFunction(BaseModel):
    """OpenAI function schema."""
    name: str
    description: str
    parameters: OpenAIFunctionParameters


class OpenAIFunctionToolSchema(BaseModel):
    """OpenAI function tool schema."""
    type: str = "function"
    function: OpenAIFunction


class ToolResult(BaseModel):
    """Standard tool execution result."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reward_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metrics": self.metrics,
            "reward_score": self.reward_score
        }


class ToolCallInfo(BaseModel):
    """Information about a tool call."""
    tool_name: str
    instance_id: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "tool_name": self.tool_name,
            "instance_id": self.instance_id,
            "parameters": self.parameters,
            "call_id": self.call_id
        }


def create_openai_tool_schema(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
) -> OpenAIFunctionToolSchema:
    """Helper function to create OpenAI tool schema."""
    return OpenAIFunctionToolSchema(
        function=OpenAIFunction(
            name=name,
            description=description,
            parameters=OpenAIFunctionParameters(
                type="object",
                properties=parameters,
                required=required or []
            )
        )
    )


def validate_tool_parameters(parameters: Dict[str, Any], schema: OpenAIFunctionToolSchema) -> bool:
    """Validate tool parameters against schema."""
    try:
        required_params = schema.function.parameters.required
        for param in required_params:
            if param not in parameters:
                return False
        return True
    except Exception:
        return False


def format_tool_error(error: Exception, tool_name: str, instance_id: str = None) -> str:
    """Format tool execution error message."""
    base_msg = f"Tool '{tool_name}' execution failed: {str(error)}"
    if instance_id:
        base_msg += f" (instance: {instance_id})"
    return base_msg