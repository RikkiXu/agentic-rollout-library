#!/usr/bin/env python3
"""
OpenHands Finish tool converted to official tool implementation.
Based on the original OpenHands finish tool.
"""

import logging
from typing import Any, Dict

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class OpenHandsFinishTool(AgenticBaseTool):
    """OpenHands-style finish tool."""
    
    def __init__(self, config: Dict = None):
        """Initialize OpenHands finish tool."""
        super().__init__(config or {})
    
    def get_description(self) -> str:
        """Override to provide custom description for OpenHands finish tool."""
        # Check if we want to use custom OpenHands-style description
        if self.config.get("use_custom_description", False):
            return """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed"""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for OpenHands finish tool."""
        return create_openai_tool_schema(
            name="openhands_finish",
            description="Signals the completion of the current task or conversation. Use when task is completed or cannot proceed further.",
            parameters={
                "message": {
                    "type": "string",
                    "description": "Final message to send to the user. Should include a summary of actions taken, results, next steps, or explanation if unable to complete the task."
                }
            },
            required=["message"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute finish command."""
        try:
            message = parameters.get("message", "")
            
            if not message:
                return ToolResult(
                    success=False,
                    error="Missing required parameter 'message'."
                )
            
            # Log the completion message
            logger.info(f"Task completion signaled: {message}")
            
            return ToolResult(
                success=True,
                result={
                    "message": message,
                    "status": "completed",
                    "tool_style": "OpenHands"
                }
            )
                
        except Exception as e:
            logger.error(f"OpenHands finish execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False, 
                error=f"Tool execution error: {str(e)}",
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        return {
            "tool_style": "OpenHands",
            "purpose": "Task completion signaling"
        }
