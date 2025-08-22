#!/usr/bin/env python3
"""
R2E Submit tool converted to official tool implementation.
Simple tool to signal task completion.
Based on the original R2E submit.py.
"""

import logging
from typing import Any, Dict, Optional

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class R2ESubmitTool(AgenticBaseTool):
    """R2E-style submit tool to finish tasks."""
    
    def __init__(self, config: Dict = None):
        """Initialize R2E submit tool."""
        super().__init__(config or {})
        
        # R2E submit message
        self.submit_message = self.config.get("submit_message", "<<<Finished>>>")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E submit."""
        return create_openai_tool_schema(
            name="r2e_submit",
            description="R2E Submit tool - signals completion of a task. Simply call to indicate task completion.",
            parameters={},
            required=[]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute submit - just return the completion message."""
        try:
            return ToolResult(
                success=True,
                result={
                    "message": self.submit_message,
                    "status": "completed"
                }
            )
        except Exception as e:
            logger.error(f"R2E submit failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get execution environment info."""
        return {
            "tool_style": "R2E",
            "submit_message": self.submit_message
        }