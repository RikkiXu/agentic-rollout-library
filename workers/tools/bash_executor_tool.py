#!/usr/bin/env python3
"""
Bash executor tool for running terminal commands.
Based on R2E's execute_bash but adapted for the core tool framework.
"""

import subprocess
import sys
import logging
from typing import Any, Dict, List

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class BashExecutorTool(AgenticBaseTool):
    """Tool for executing bash commands safely."""
    
    # Commands that are blocked for security reasons
    DEFAULT_BLOCKED_COMMANDS = [
        "git", "ipython", "jupyter", "nohup", "sudo", "rm -rf", 
        "shutdown", "reboot", "passwd", "su", "chmod 777"
    ]
    
    def __init__(self, config: Dict = None):
        """Initialize bash executor tool."""
        super().__init__(config)
        self.blocked_commands = set(
            self.config.get("blocked_commands", self.DEFAULT_BLOCKED_COMMANDS)
        )
        self.allow_dangerous = self.config.get("allow_dangerous", False)
        self.timeout = self.config.get("timeout", 30)  # 30 second default timeout
        self.execution_history = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for bash executor."""
        return create_openai_tool_schema(
            name="bash_executor",
            description="Execute bash commands in the terminal. Use with caution as this can modify system state.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute (e.g., 'ls -la', 'python script.py', 'grep pattern file.txt')"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Optional working directory for command execution"
                },
                "timeout": {
                    "type": "number",
                    "description": "Command timeout in seconds (default: 30)"
                },
                "capture_output": {
                    "type": "boolean",
                    "description": "Whether to capture and return command output (default: true)"
                }
            },
            required=["command"]
        )
    
    async def _initialize_instance(self, instance_id: str, **kwargs) -> None:
        """Initialize instance-specific execution history."""
        self.execution_history[instance_id] = []
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute bash command."""
        try:
            command = parameters["command"].strip()
            working_dir = parameters.get("working_directory")
            timeout = parameters.get("timeout", self.timeout)
            capture_output = parameters.get("capture_output", True)
            
            # Security check
            security_result = self._check_command_security(command)
            if not security_result["safe"]:
                return ToolResult(
                    success=False,
                    error=security_result["reason"]
                )
            
            # Execute command
            result = await self._run_command(
                command, 
                working_dir=working_dir,
                timeout=timeout,
                capture_output=capture_output
            )
            
            # Record execution
            execution_record = {
                "command": command,
                "working_directory": working_dir,
                "return_code": result["return_code"],
                "success": result["return_code"] == 0
            }
            
            if instance_id in self.execution_history:
                self.execution_history[instance_id].append(execution_record)
            
            if result["return_code"] == 0:
                return ToolResult(
                    success=True,
                    result={
                        "stdout": result["stdout"],
                        "stderr": result["stderr"],
                        "return_code": result["return_code"],
                        "command": command
                    },
                    metrics={
                        "return_code": result["return_code"],
                        "stdout_length": len(result["stdout"]),
                        "stderr_length": len(result["stderr"])
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Command failed with return code {result['return_code']}",
                    result={
                        "stdout": result["stdout"],
                        "stderr": result["stderr"],
                        "return_code": result["return_code"],
                        "command": command
                    }
                )
                
        except Exception as e:
            logger.error(f"Bash execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _check_command_security(self, command: str) -> Dict[str, Any]:
        """Check if command is safe to execute."""
        command_lower = command.lower().strip()
        
        # Check blocked commands
        first_token = command.strip().split()[0] if command.strip() else ""
        
        if first_token in self.blocked_commands:
            return {
                "safe": False,
                "reason": f"Command '{first_token}' is blocked for security reasons"
            }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            ":(){ :|:& };:",  # Fork bomb
            "chmod -R 777",
            "sudo rm",
            "> /dev/sda",
            "dd if=/dev/zero",
            "wget | sh",
            "curl | sh",
        ]
        
        if not self.allow_dangerous:
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    return {
                        "safe": False,
                        "reason": f"Command contains dangerous pattern: {pattern}"
                    }
        
        # Check for pipe to shell execution
        if not self.allow_dangerous and ("|sh" in command_lower or "|bash" in command_lower):
            return {
                "safe": False,
                "reason": "Piping to shell is not allowed for security reasons"
            }
        
        return {"safe": True, "reason": "Command passed security checks"}
    
    async def _run_command(self, command: str, working_dir: str = None, 
                          timeout: float = 30, capture_output: bool = True) -> Dict[str, Any]:
        """Run bash command and return results."""
        try:
            # Prepare subprocess arguments
            kwargs = {
                "shell": True,
                "timeout": timeout,
                "cwd": working_dir
            }
            
            if capture_output:
                # Use newer parameters for Python 3.7+
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        **kwargs
                    )
                except TypeError:
                    # Fallback for Python 3.5-3.6
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        **kwargs
                    )
                
                return {
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                    "return_code": result.returncode
                }
            else:
                # Run without capturing output
                result = subprocess.run(command, **kwargs)
                return {
                    "stdout": "",
                    "stderr": "",
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "return_code": -1
            }
    
    def add_blocked_command(self, command: str) -> None:
        """Add a command to the blocked list."""
        self.blocked_commands.add(command)
    
    def remove_blocked_command(self, command: str) -> None:
        """Remove a command from the blocked list."""
        self.blocked_commands.discard(command)
    
    def get_blocked_commands(self) -> List[str]:
        """Get list of blocked commands."""
        return list(self.blocked_commands)
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """Clean up instance-specific data."""
        if instance_id in self.execution_history:
            del self.execution_history[instance_id]