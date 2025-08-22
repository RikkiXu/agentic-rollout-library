#!/usr/bin/env python3
"""
R2E Bash Executor tool converted to official tool implementation.
Supports both local execution and remote K8s pod execution.
Based on the original R2E execute_bash.py.
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


class R2EBashExecutorTool(AgenticBaseTool):
    """R2E-style bash executor tool."""
    
    # Commands that are blocked for security reasons (from R2E)
    R2E_BLOCKED_COMMANDS = ["git", "ipython", "jupyter", "nohup"]
    
    def __init__(self, config: Dict = None):
        """Initialize R2E bash executor tool."""
        # Set execution mode and K8s config FIRST, before calling super().__init__
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        self.working_dir = config.get("working_dir", "/testbed")  # R2E default working directory
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode. Please install it from https://github.com/baidubce/kodo.git")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
        
        super().__init__(config)
        
        # R2E-specific settings
        self.blocked_commands = set(
            self.config.get("blocked_commands", self.R2E_BLOCKED_COMMANDS)
        )
        self.timeout = self.config.get("timeout", 30)  # 30 second default timeout
        self.k8s_manager = None
    
    def get_description(self) -> str:
        """Override to provide custom description for R2E bash executor."""
        # Check if we want to use custom R2E-style description
        if self.config.get("use_custom_description", False):
            return """–– BEGIN FUNCTION: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
  •    If a command may run indefinitely (long-running), consider running it in the background
  •    Command returns exit code -1 if process is still running
  •    Send command="ctrl+c" to interrupt the running process
  •    If the command times out, it will be interrupted (SIGINT)

Parameters:
  1.    cmd (string, required)
The bash command (and optional arguments) to execute.

–– END FUNCTION ––"""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E bash executor."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="r2e_bash_executor",
            description=f"Execute a bash command in the terminal{execution_context}. R2E-style tool with security restrictions.",
            parameters={
                "cmd": {
                    "type": "string",
                    "description": "The bash command to execute. For example: 'python my_script.py'"
                }
            },
            required=["cmd"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute bash command."""
        try:
            # Support both 'command' and 'cmd' parameter names for backward compatibility
            if "command" in parameters:
                command = parameters["command"].strip()
            elif "cmd" in parameters:
                command = parameters["cmd"].strip()
            else:
                return ToolResult(
                    success=False,
                    error="Missing required parameter. Please provide either 'command' or 'cmd' parameter."
                )
            
            # Security check (R2E-style)
            first_token = command.split()[0] if command.strip() else ""
            if first_token in self.blocked_commands:
                return ToolResult(
                    success=False,
                    error=f"Bash command '{first_token}' is not allowed. Please use a different command or tool."
                )
            
            # Execute command based on execution mode
            if self.execution_mode == "k8s":
                result = await self._run_k8s_command(command)
            else:
                result = await self._run_local_command(command)
            
            # Format output R2E-style
            output_parts = []
            output_parts.append("[STDOUT]")
            output_parts.append(result["stdout"].strip())
            output_parts.append("")
            output_parts.append("[STDERR]")
            output_parts.append(result["stderr"].strip())
            
            formatted_output = "\n".join(output_parts)
            
            if result["return_code"] != 0:
                # Create detailed error message based on exit code
                exit_code = result["return_code"]
                error_msg = f"Command failed with exit code {exit_code}"
                
                # Add common exit code explanations
                if exit_code == 127:
                    error_msg += " (Command not found)"
                    error_msg += "\nThis usually means:"
                    error_msg += "\n- The command/program doesn't exist"
                    error_msg += "\n- The command is not in PATH"
                    error_msg += "\n- For 'python', try 'python3' instead"
                elif exit_code == 126:
                    error_msg += " (Permission denied - cannot execute)"
                elif exit_code == 1:
                    error_msg += " (General error)"
                elif exit_code == 2:
                    error_msg += " (Misuse of shell command)"
                
                if result["stderr"]:
                    error_msg += f"\n\nSTDERR: {result['stderr']}"
                
                # Add stdout if available (sometimes errors appear in stdout)
                if result["stdout"]:
                    error_msg += f"\n\nSTDOUT: {result['stdout']}"
                
                return ToolResult(
                    success=False,
                    error=error_msg,
                    result={
                        "output": formatted_output,
                        "return_code": result["return_code"],
                        "command": command
                    }
                )
            else:
                return ToolResult(
                    success=True,
                    result={
                        "output": formatted_output,
                        "return_code": result["return_code"],
                        "command": command
                    }
                )
                
        except Exception as e:
            logger.error(f"R2E bash execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False, 
                error=f"Tool execution error: {str(e)}\nCommand: {parameters.get('command', parameters.get('cmd', 'N/A'))}",
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    async def _run_local_command(self, command: str) -> Dict[str, Any]:
        """Run bash command locally (R2E-style)."""
        try:
            # For local execution, prepend cd to working directory
            full_command = f"cd {self.working_dir} && {command}"
            
            # Try to use the new parameters (Python 3.7+)
            try:
                result = subprocess.run(
                    full_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            except TypeError:
                # Fallback for Python 3.5 and 3.6
                result = subprocess.run(
                    full_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=self.timeout
                )
            
            return {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "return_code": result.returncode
            }
                
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds",
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

    async def _run_k8s_command(self, command: str) -> Dict[str, Any]:
        """Run bash command in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Prepend cd to working directory and properly handle timeout
            # Format: cd {working_dir} && timeout {timeout} {command}
            # This ensures we're always in the right directory and have timeout protection
            full_command = f"cd {self.working_dir} && timeout {self.timeout} {command}"
            
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
            "blocked_commands": list(self.blocked_commands),
            "tool_style": "R2E"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info