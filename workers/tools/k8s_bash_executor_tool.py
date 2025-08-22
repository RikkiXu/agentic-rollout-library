#!/usr/bin/env python3
"""
K8s Bash executor tool for running terminal commands in Kubernetes pods.
Based on bash_executor_tool.py but adapted to execute commands in K8s pods using kodo.
"""

import asyncio
import logging
from typing import Any, Dict, List
try:
    from kodo import KubernetesManager
except ImportError:
    KubernetesManager = None

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class K8sBashExecutorTool(AgenticBaseTool):
    """Tool for executing bash commands in Kubernetes pods."""
    
    # Commands that are blocked for security reasons
    DEFAULT_BLOCKED_COMMANDS = [
        "git", "ipython", "jupyter", "nohup", "sudo", "rm -rf", 
        "shutdown", "reboot", "passwd", "su", "chmod 777"
    ]
    
    def __init__(self, config: Dict = None):
        """Initialize K8s bash executor tool."""
        if KubernetesManager is None:
            raise ImportError("kodo library is required for K8s tools. Please install it from https://github.com/baidubce/kodo.git")
        
        # Set K8s configuration first before calling super().__init__
        config = config or {}
        self.pod_name = config.get("pod_name", "swebench-xarray-pod")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        # Security and execution settings
        self.blocked_commands = set(
            self.config.get("blocked_commands", self.DEFAULT_BLOCKED_COMMANDS)
        )
        self.allow_dangerous = self.config.get("allow_dangerous", False)
        self.timeout = self.config.get("timeout", 30)  # 30 second default timeout
        self.execution_history = {}
        
        # Initialize K8s manager
        self.k8s_manager = None
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for K8s bash executor."""
        return create_openai_tool_schema(
            name="k8s_bash_executor",
            description=f"Execute bash commands in Kubernetes pod '{self.pod_name}'. Use with caution as this can modify system state within the pod.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute in the K8s pod (e.g., 'ls -la', 'python script.py', 'grep pattern file.txt')"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Optional working directory for command execution within the pod"
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
        """Execute bash command in K8s pod."""
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
            
            # Prepend cd command if working directory is specified
            if working_dir:
                command = f"cd {working_dir} && {command}"
            
            # Execute command in K8s pod
            result = await self._run_k8s_command(command, timeout, capture_output)
            
            # Record execution
            execution_record = {
                "command": command,
                "working_directory": working_dir,
                "return_code": result["return_code"],
                "success": result["return_code"] == 0,
                "pod_name": self.pod_name,
                "namespace": self.namespace
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
                        "command": command,
                        "pod_name": self.pod_name,
                        "namespace": self.namespace
                    },
                    metrics={
                        "return_code": result["return_code"],
                        "stdout_length": len(result["stdout"]),
                        "stderr_length": len(result["stderr"]),
                        "execution_location": f"{self.namespace}/{self.pod_name}"
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
                        "command": command,
                        "pod_name": self.pod_name,
                        "namespace": self.namespace
                    }
                )
                
        except Exception as e:
            logger.error(f"K8s bash execution failed: {e}")
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
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager

    async def _run_k8s_command(self, command: str, timeout: float = 30, 
                              capture_output: bool = True) -> Dict[str, Any]:
        """Run bash command in K8s pod and return results."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Execute command in pod using kodo API
            output, exit_code = k8s_mgr.execute_command(self.pod_name, command)
            
            return {
                "stdout": output,
                "stderr": "",  # kodo API doesn't separate stderr
                "return_code": exit_code
            }
            
        except Exception as e:
            logger.error(f"K8s command execution failed: {e}")
            return {
                "stdout": "",
                "stderr": f"K8s execution error: {str(e)}",
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
    
    def get_pod_info(self) -> Dict[str, str]:
        """Get information about the target pod."""
        return {
            "pod_name": self.pod_name,
            "namespace": self.namespace,
            "kubeconfig_path": self.kubeconfig_path or "default"
        }
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """Clean up instance-specific data."""
        if instance_id in self.execution_history:
            del self.execution_history[instance_id]