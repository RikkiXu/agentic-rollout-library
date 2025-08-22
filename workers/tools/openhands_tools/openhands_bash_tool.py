#!/usr/bin/env python3
"""
OpenHands Bash Executor tool converted to official tool implementation.
Supports both local execution and remote K8s pod execution.
Based on the original OpenHands bash tool.
"""

import subprocess
import logging
from typing import Any, Dict

# Optional K8s support
try:
    from kodo import KubernetesManager
    K8S_AVAILABLE = True
except ImportError:
    KubernetesManager = None
    K8S_AVAILABLE = False

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class OpenHandsBashTool(AgenticBaseTool):
    """OpenHands-style bash executor tool with K8s support."""
    
    def __init__(self, config: Dict = None):
        """Initialize OpenHands bash executor tool."""
        # Set execution mode and K8s config FIRST, before calling super().__init__
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        self.working_dir = config.get("working_dir", "/workspace")  # OpenHands default working directory
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode. Please install it from https://github.com/baidubce/kodo.git")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
        
        super().__init__(config)
        
        # OpenHands-specific settings
        self.timeout = self.config.get("timeout", 10)  # 10 second soft timeout like OpenHands
        self.use_short_description = self.config.get("use_short_description", False)
        self.k8s_manager = None
    
    def get_description(self) -> str:
        """Override to provide custom description for OpenHands bash executor."""
        # Check if we want to use custom OpenHands-style description
        if self.config.get("use_custom_description", False):
            if self.use_short_description:
                return """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, you can set the "timeout" argument to specify a hard timeout in seconds.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."""
            else:
                return """Execute a bash command in the terminal within a persistent shell session.

### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)

### Long-running Commands
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* For commands that may run for a long time (e.g. installation or testing commands), or commands that run for a fixed amount of time (e.g. sleep), you should set the "timeout" parameter of your function call to an appropriate value.
* If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process
  - If you do C-c, you can re-start the process with a longer "timeout" parameter to let it run to completion

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned."""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for OpenHands bash executor."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="openhands_bash",
            description=f"Execute a bash command in the terminal{execution_context}. OpenHands-style tool with persistent session support.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."
                },
                "is_input": {
                    "type": "string",
                    "description": "If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.",
                    "enum": ["true", "false"]
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior."
                }
            },
            required=["command"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute bash command."""
        try:
            command = parameters.get("command", "").strip()
            is_input = parameters.get("is_input", "false").lower() == "true"
            timeout = parameters.get("timeout", self.timeout)
            
            if not command and not is_input:
                return ToolResult(
                    success=False,
                    error="Missing required parameter 'command'."
                )
            
            # Execute command based on execution mode
            if self.execution_mode == "k8s":
                result = await self._run_k8s_command(command, is_input, timeout)
            else:
                result = await self._run_local_command(command, is_input, timeout)
            
            # Format output OpenHands-style
            if result["return_code"] != 0:
                # For OpenHands compatibility, handle special cases
                if result["return_code"] == -1:
                    # Process is still running or timed out
                    return ToolResult(
                        success=True,  # OpenHands treats timeout as success but with special exit code
                        result={
                            "output": result["stdout"],
                            "error": result["stderr"],
                            "exit_code": -1,
                            "command": command,
                            "status": "timeout_or_running"
                        }
                    )
                else:
                    # Command failed
                    return ToolResult(
                        success=False,
                        error=f"Command failed with exit code {result['return_code']}: {result['stderr']}",
                        result={
                            "output": result["stdout"],
                            "error": result["stderr"],
                            "exit_code": result["return_code"],
                            "command": command
                        }
                    )
            else:
                # Command succeeded
                return ToolResult(
                    success=True,
                    result={
                        "output": result["stdout"],
                        "error": result["stderr"],
                        "exit_code": result["return_code"],
                        "command": command
                    }
                )
                
        except Exception as e:
            logger.error(f"OpenHands bash execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False, 
                error=f"Tool execution error: {str(e)}\nCommand: {parameters.get('command', 'N/A')}",
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    async def _run_local_command(self, command: str, is_input: bool, timeout: int) -> Dict[str, Any]:
        """Run bash command locally (OpenHands-style)."""
        try:
            if is_input:
                # Handle input to existing process - simplified implementation
                # In a real implementation, this would interact with a persistent shell session
                return {
                    "stdout": f"Input '{command}' sent to process (simulated)",
                    "stderr": "",
                    "return_code": 0
                }
            
            # For local execution, prepend cd to working directory
            full_command = f"cd {self.working_dir} && {command}"
            
            # Try to use the new parameters (Python 3.7+)
            try:
                result = subprocess.run(
                    full_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            except TypeError:
                # Fallback for Python 3.5 and 3.6
                result = subprocess.run(
                    full_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=timeout
                )
            
            return {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
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
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager

    async def _run_k8s_command(self, command: str, is_input: bool, timeout: int) -> Dict[str, Any]:
        """Run bash command in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            if is_input:
                # Handle input to existing process - simplified implementation for K8s
                # In a real implementation, this would interact with a persistent shell session
                logger.info(f"Sending input to K8s pod {self.pod_name}: {command}")
                return {
                    "stdout": f"Input '{command}' sent to process in pod (simulated)",
                    "stderr": "",
                    "return_code": 0
                }
            
            # Prepend cd to working directory and properly handle timeout
            full_command = f"cd {self.working_dir} && timeout {timeout} {command}"
            
            logger.info(f"Executing command in K8s pod {self.pod_name}: {full_command}")
            
            # Execute command in pod using kodo API
            output, exit_code = k8s_mgr.execute_command(self.pod_name, full_command)
            
            # Log raw output for debugging
            logger.debug(f"Raw K8s output: {output}")
            logger.debug(f"Raw K8s exit code: {exit_code}")
            
            # Convert exit_code to int if it's a string
            if isinstance(exit_code, str):
                # Handle "Error: Exit code X" format
                if "Exit code" in exit_code:
                    try:
                        # Extract number from "Error: Exit code 2"
                        exit_code_int = int(exit_code.split("Exit code")[-1].strip())
                    except:
                        exit_code_int = -1
                elif exit_code.isdigit():
                    exit_code_int = int(exit_code)
                else:
                    exit_code_int = -1
            else:
                exit_code_int = exit_code
            
            # Check if output contains error information
            stderr_output = ""
            if exit_code_int != 0 and output:
                # Sometimes errors are mixed in stdout when using kubectl exec
                stderr_output = output if "error" in output.lower() or "exception" in output.lower() else ""
            
            return {
                "stdout": output,
                "stderr": stderr_output,
                "return_code": exit_code_int
            }
            
        except Exception as e:
            logger.error(f"K8s command execution failed for pod {self.pod_name}: {e}", exc_info=True)
            error_details = f"K8s execution error: {str(e)}\nPod: {self.pod_name}\nNamespace: {self.namespace}\nCommand: {command}"
            return {
                "stdout": "",
                "stderr": error_details,
                "return_code": -1
            }
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        info = {
            "execution_mode": self.execution_mode,
            "timeout": self.timeout,
            "working_dir": self.working_dir,
            "tool_style": "OpenHands"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info
