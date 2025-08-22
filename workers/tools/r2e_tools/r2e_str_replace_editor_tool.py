#!/usr/bin/env python3
"""
R2E String Replace Editor tool converted to official tool implementation.
A simplified version of the file editor focused on specific commands.
Based on the original R2E str_replace_editor.py.
"""

from typing import Any, Dict, Optional

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

import logging
logger = logging.getLogger(__name__)

# Control what commands are visible to agents (from R2E)
ALLOWED_STR_REPLACE_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]

# Import at the end to avoid circular import
from .r2e_file_editor_tool import R2EFileEditorTool


class R2EStrReplaceEditorTool(R2EFileEditorTool):
    """
    R2E String Replace Editor - a restricted version of the file editor.
    Inherits from R2EFileEditorTool but limits available commands.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize R2E string replace editor tool."""
        # Set allowed commands before calling super().__init__
        config = config or {}
        self.allowed_commands = config.get(
            "allowed_commands", 
            ALLOWED_STR_REPLACE_EDITOR_COMMANDS
        )
        
        super().__init__(config)
        
        # str_replace_editor doesn't support undo by default
        if "undo_edit" not in self.allowed_commands:
            self.file_history.clear()  # Don't maintain history if undo not allowed
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E string replace editor."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        # Build enum list based on allowed commands
        command_enum = [cmd for cmd in self.allowed_commands]
        
        schema = create_openai_tool_schema(
            name="r2e_str_replace_editor",
            description=f"R2E String Replace Editor for viewing, creating and editing files{execution_context}. Limited command set version of file editor.",
            parameters={
                "command": {
                    "type": "string",
                    "description": f"The command to run. Allowed options: {', '.join(command_enum)}",
                    "enum": command_enum
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory"
                },
                "file_text": {
                    "type": "string",
                    "description": "Required for 'create' command - content of the file to be created"
                },
                "old_str": {
                    "type": "string",
                    "description": "Required for 'str_replace' - the exact string to replace (must be unique)"
                },
                "new_str": {
                    "type": "string",
                    "description": "For 'str_replace' - replacement string (optional, defaults to empty); For 'insert' - string to insert (required)"
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required for 'insert' - line number after which to insert new_str"
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional for 'view' - line range [start, end], use -1 for end of file"
                }
            },
            required=["command", "path"]
        )
        
        return schema
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute string replace editor command."""
        command = parameters.get("command")
        
        # Check if command is allowed
        if command not in self.allowed_commands:
            return ToolResult(
                success=False, 
                error=f"Command '{command}' is not allowed. Allowed commands: {', '.join(self.allowed_commands)}"
            )
        
        # Delegate to parent class
        return await super().execute_tool(instance_id, parameters, **kwargs)
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get execution environment info."""
        info = super().get_execution_info()
        info.update({
            "tool_style": "R2E_StrReplaceEditor",
            "allowed_commands": self.allowed_commands
        })
        return info