#!/usr/bin/env python3
"""
Finish tool for terminating agent execution.
This tool is used by agents to signal task completion.
"""

import logging
from typing import Any, Dict, Optional

from ..core.base_tool import BaseAgenticTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class FinishTool(BaseAgenticTool):
    """
    Tool for terminating agent execution with a final answer or result.
    
    This tool allows agents to signal task completion and provide
    their final output or answer.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Finish tool."""
        super().__init__(config)
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI function schema for this tool."""
        return create_openai_tool_schema(
            name="finish",
            description="Complete the task and provide the final answer or result",
            parameters={
                "answer": {
                    "type": "string",
                    "description": "The final answer or result of the task"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Optional explanation of the reasoning behind the answer"
                },
                "status": {
                    "type": "string", 
                    "enum": ["success", "partial", "failed"],
                    "description": "Status of task completion",
                    "default": "success"
                }
            },
            required=["answer"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute the finish tool.
        
        Args:
            instance_id: Tool instance ID
            parameters: Tool parameters containing 'answer', optional 'reasoning' and 'status'
            **kwargs: Additional execution arguments
            
        Returns:
            ToolResult indicating successful completion
        """
        try:
            answer = parameters.get("answer", "")
            reasoning = parameters.get("reasoning", "")
            status = parameters.get("status", "success")
            
            # Log the completion
            logger.info(f"Task completed with status: {status}")
            if reasoning:
                logger.info(f"Reasoning: {reasoning}")
            
            # Create result with completion info
            result_data = {
                "answer": answer,
                "status": status,
                "completed": True
            }
            
            if reasoning:
                result_data["reasoning"] = reasoning
            
            return ToolResult(
                success=True,
                result=result_data,
                metadata={
                    "tool_name": "finish",
                    "completion_status": status,
                    "instance_id": instance_id
                }
            )
            
        except Exception as e:
            logger.error(f"Finish tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to complete task: {str(e)}"
            )
    
    async def calculate_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate reward for task completion.
        
        Args:
            instance_id: Tool instance ID
            **kwargs: Additional arguments
            
        Returns:
            Reward score (1.0 for successful completion)
        """
        return 1.0


__all__ = ['FinishTool']