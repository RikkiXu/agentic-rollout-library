#!/usr/bin/env python3
"""
K8s File editor tool for viewing, creating, and editing files in Kubernetes pods.
Based on file_editor_tool.py but adapted to work with files in K8s pods using kodo.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from kodo import KubernetesManager
except ImportError:
    KubernetesManager = None

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class K8sFileEditorTool(AgenticBaseTool):
    """Tool for file operations in Kubernetes pods: view, create, edit, and search."""
    
    def __init__(self, config: Dict = None):
        """Initialize K8s file editor tool."""
        if KubernetesManager is None:
            raise ImportError("kodo library is required for K8s tools. Please install it from https://github.com/baidubce/kodo.git")
        
        # Set K8s configuration first before calling super().__init__
        config = config or {}
        self.pod_name = config.get("pod_name", "swebench-xarray-pod")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        # File operation settings
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB default
        self.max_response_length = self.config.get("max_response_length", 10000)
        self.allowed_extensions = set(self.config.get("allowed_extensions", [
            ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".sh", ".js", ".ts", 
            ".html", ".css", ".xml", ".csv", ".log", ".c", ".cpp", ".h", ".java"
        ]))
        self.enable_linting = self.config.get("enable_linting", False)
        self.file_history = {}  # instance_id -> {file_path -> [history]}
        
        # Initialize K8s manager
        self.k8s_manager = None
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for K8s file editor."""
        return create_openai_tool_schema(
            name="k8s_file_editor",
            description=f"File operations tool for viewing, creating, editing files and directories in K8s pod '{self.pod_name}'. Supports string replacement, insertion, and file management.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"]
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory within the pod"
                },
                "file_text": {
                    "type": "string",
                    "description": "Content for file creation (required for 'create' command)"
                },
                "old_str": {
                    "type": "string",
                    "description": "String to replace (required for 'str_replace' command)"
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (for 'str_replace' or 'insert' commands)"
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert text after (required for 'insert' command)"
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Line range to view [start_line, end_line], use -1 for end"
                }
            },
            required=["command", "path"]
        )
    
    async def _initialize_instance(self, instance_id: str, **kwargs) -> None:
        """Initialize instance-specific file history."""
        self.file_history[instance_id] = {}
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute file operation in K8s pod."""
        try:
            command = parameters["command"]
            path_str = parameters["path"]
            
            if command == "view":
                return await self._view_file(path_str, parameters.get("view_range"))
            elif command == "create":
                file_text = parameters.get("file_text")
                if file_text is None:
                    return ToolResult(success=False, error="file_text parameter required for create command")
                return await self._create_file(instance_id, path_str, file_text)
            elif command == "str_replace":
                old_str = parameters.get("old_str")
                new_str = parameters.get("new_str", "")
                if old_str is None:
                    return ToolResult(success=False, error="old_str parameter required for str_replace command")
                return await self._str_replace(instance_id, path_str, old_str, new_str)
            elif command == "insert":
                insert_line = parameters.get("insert_line")
                new_str = parameters.get("new_str")
                if insert_line is None or new_str is None:
                    return ToolResult(success=False, error="insert_line and new_str parameters required for insert command")
                return await self._insert_text(instance_id, path_str, insert_line, new_str)
            elif command == "undo_edit":
                return await self._undo_edit(instance_id, path_str)
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(f"K8s file editor execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _view_file(self, path_str: str, view_range: Optional[List[int]] = None) -> ToolResult:
        """View file or directory contents in K8s pod."""
        try:
            # Check if path exists and get type
            check_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if check_result["stdout"].strip() == "not_exists":
                return ToolResult(success=False, error=f"Path does not exist: {path_str}")
            
            # Check if it's a directory
            is_dir_result = await self._exec_command(f"test -d '{path_str}' && echo 'dir' || echo 'file'")
            is_directory = is_dir_result["stdout"].strip() == "dir"
            
            if is_directory:
                # List directory contents
                list_result = await self._exec_command(f"ls -la '{path_str}'")
                if list_result["return_code"] != 0:
                    return ToolResult(success=False, error=f"Failed to list directory: {list_result['stderr']}")
                
                content = f"Directory contents of {path_str}:\n{list_result['stdout']}"
                return ToolResult(success=True, result={"content": content, "type": "directory"})
            
            else:
                # View file
                if not self._is_allowed_file(path_str):
                    return ToolResult(success=False, error=f"File type not allowed: {Path(path_str).suffix}")
                
                # Check file size
                size_result = await self._exec_command(f"stat -c %s '{path_str}' 2>/dev/null || echo '0'")
                try:
                    file_size = int(size_result["stdout"].strip())
                    if file_size > self.max_file_size:
                        return ToolResult(success=False, error=f"File too large: {file_size} bytes")
                except ValueError:
                    pass  # Continue if we can't get size
                
                # Read file content
                cat_result = await self._exec_command(f"cat '{path_str}'")
                if cat_result["return_code"] != 0:
                    return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
                
                content = cat_result["stdout"]
                lines = content.splitlines()
                
                # Apply view range if specified
                if view_range and len(view_range) == 2:
                    start, end = view_range
                    total_lines = len(lines)
                    
                    if not (1 <= start <= total_lines):
                        return ToolResult(success=False, error=f"Invalid start line {start}, file has {total_lines} lines")
                    
                    if end != -1 and (end < start or end > total_lines):
                        return ToolResult(success=False, error=f"Invalid end line {end}")
                    
                    # Slice lines (convert to 0-based indexing)
                    if end == -1:
                        lines = lines[start-1:]
                    else:
                        lines = lines[start-1:end]
                
                # Format with line numbers
                numbered_lines = []
                start_line = 1 if not view_range else view_range[0]
                for i, line in enumerate(lines):
                    numbered_lines.append(f"{start_line + i:6d}  {line}")
                
                formatted_content = "\n".join(numbered_lines)
                
                # Truncate if too long
                if len(formatted_content) > self.max_response_length:
                    formatted_content = formatted_content[:self.max_response_length] + "\n<response clipped>"
                
                return ToolResult(
                    success=True, 
                    result={
                        "content": formatted_content,
                        "type": "file",
                        "total_lines": len(content.splitlines()),
                        "displayed_lines": len(lines),
                        "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                    }
                )
                
        except Exception as e:
            return ToolResult(success=False, error=f"View operation failed: {e}")
    
    async def _create_file(self, instance_id: str, path_str: str, file_text: str) -> ToolResult:
        """Create a new file in K8s pod."""
        try:
            # Check if file exists
            check_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if check_result["stdout"].strip() == "exists":
                return ToolResult(success=False, error=f"File already exists: {path_str}")
            
            if not self._is_allowed_file(path_str):
                return ToolResult(success=False, error=f"File type not allowed: {Path(path_str).suffix}")
            
            # Create parent directories if needed
            parent_dir = str(Path(path_str).parent)
            mkdir_result = await self._exec_command(f"mkdir -p '{parent_dir}'")
            if mkdir_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to create parent directory: {mkdir_result['stderr']}")
            
            # Lint check for Python files
            if self.enable_linting and path_str.endswith(".py"):
                lint_error = self._lint_check(file_text)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Write file using cat with here-doc to handle special characters
            escaped_content = file_text.replace("'", "'\"'\"'")  # Escape single quotes
            write_result = await self._exec_command(f"cat > '{path_str}' << 'EOF'\n{file_text}\nEOF")
            
            if write_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to create file: {write_result['stderr']}")
            
            # Initialize history
            if instance_id in self.file_history:
                self.file_history[instance_id][path_str] = [""]
            
            return ToolResult(
                success=True,
                result={
                    "message": f"File created: {path_str}",
                    "path": path_str,
                    "size": len(file_text),
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Create operation failed: {e}")
    
    async def _str_replace(self, instance_id: str, path_str: str, old_str: str, new_str: str) -> ToolResult:
        """Replace string in file in K8s pod."""
        try:
            # Check if file exists
            check_result = await self._exec_command(f"test -f '{path_str}' && echo 'exists' || echo 'not_exists'")
            if check_result["stdout"].strip() == "not_exists":
                return ToolResult(success=False, error=f"File does not exist: {path_str}")
            
            if not self._is_allowed_file(path_str):
                return ToolResult(success=False, error=f"File type not allowed: {Path(path_str).suffix}")
            
            # Read current content
            cat_result = await self._exec_command(f"cat '{path_str}'")
            if cat_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
            
            old_content = cat_result["stdout"]
            
            # Check for occurrences
            occurrences = old_content.count(old_str)
            if occurrences == 0:
                return ToolResult(success=False, error=f"String not found in file: {old_str}")
            elif occurrences > 1:
                return ToolResult(success=False, error=f"Multiple occurrences found ({occurrences}). String must be unique.")
            
            # Perform replacement
            new_content = old_content.replace(old_str, new_str)
            
            # Lint check for Python files
            if self.enable_linting and path_str.endswith(".py"):
                lint_error = self._lint_check(new_content)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Save history
            if instance_id in self.file_history:
                if path_str not in self.file_history[instance_id]:
                    self.file_history[instance_id][path_str] = []
                self.file_history[instance_id][path_str].append(old_content)
            
            # Write new content
            write_result = await self._exec_command(f"cat > '{path_str}' << 'EOF'\n{new_content}\nEOF")
            if write_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to write file: {write_result['stderr']}")
            
            # Find replacement location for context
            replacement_line = old_content.split(old_str)[0].count('\n')
            context_start = max(0, replacement_line - 2)
            context_end = min(len(new_content.splitlines()), replacement_line + new_str.count('\n') + 3)
            
            context_lines = new_content.splitlines()[context_start:context_end]
            context = "\n".join(f"{context_start + i + 1:6d}  {line}" for i, line in enumerate(context_lines))
            
            return ToolResult(
                success=True,
                result={
                    "message": f"String replaced in {path_str}",
                    "path": path_str,
                    "context": context,
                    "replacement_line": replacement_line + 1,
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"String replace operation failed: {e}")
    
    async def _insert_text(self, instance_id: str, path_str: str, insert_line: int, new_str: str) -> ToolResult:
        """Insert text at specified line in K8s pod."""
        try:
            # Check if file exists
            check_result = await self._exec_command(f"test -f '{path_str}' && echo 'exists' || echo 'not_exists'")
            if check_result["stdout"].strip() == "not_exists":
                return ToolResult(success=False, error=f"File does not exist: {path_str}")
            
            if not self._is_allowed_file(path_str):
                return ToolResult(success=False, error=f"File type not allowed: {Path(path_str).suffix}")
            
            # Read current content
            cat_result = await self._exec_command(f"cat '{path_str}'")
            if cat_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
            
            old_content = cat_result["stdout"]
            lines = old_content.splitlines()
            
            if insert_line < 0 or insert_line > len(lines):
                return ToolResult(success=False, error=f"Invalid insert line {insert_line}. File has {len(lines)} lines.")
            
            # Insert text
            new_lines = new_str.splitlines()
            updated_lines = lines[:insert_line] + new_lines + lines[insert_line:]
            new_content = "\n".join(updated_lines)
            
            # Lint check for Python files
            if self.enable_linting and path_str.endswith(".py"):
                lint_error = self._lint_check(new_content)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Save history
            if instance_id in self.file_history:
                if path_str not in self.file_history[instance_id]:
                    self.file_history[instance_id][path_str] = []
                self.file_history[instance_id][path_str].append(old_content)
            
            # Write new content
            write_result = await self._exec_command(f"cat > '{path_str}' << 'EOF'\n{new_content}\nEOF")
            if write_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to write file: {write_result['stderr']}")
            
            # Generate context
            context_start = max(0, insert_line - 2)
            context_end = min(len(updated_lines), insert_line + len(new_lines) + 3)
            context_lines = updated_lines[context_start:context_end]
            context = "\n".join(f"{context_start + i + 1:6d}  {line}" for i, line in enumerate(context_lines))
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Text inserted in {path_str} at line {insert_line}",
                    "path": path_str,
                    "context": context,
                    "lines_inserted": len(new_lines),
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Insert operation failed: {e}")
    
    async def _undo_edit(self, instance_id: str, path_str: str) -> ToolResult:
        """Undo last edit operation in K8s pod."""
        try:
            if instance_id not in self.file_history:
                return ToolResult(success=False, error="No edit history available")
            
            if path_str not in self.file_history[instance_id] or not self.file_history[instance_id][path_str]:
                return ToolResult(success=False, error=f"No edit history for file: {path_str}")
            
            # Restore previous content
            previous_content = self.file_history[instance_id][path_str].pop()
            write_result = await self._exec_command(f"cat > '{path_str}' << 'EOF'\n{previous_content}\nEOF")
            
            if write_result["return_code"] != 0:
                return ToolResult(success=False, error=f"Failed to restore file: {write_result['stderr']}")
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Undid last edit to {path_str}",
                    "path": path_str,
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Undo operation failed: {e}")
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager

    async def _exec_command(self, command: str) -> Dict[str, Any]:
        """Execute command in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
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
                "stderr": str(e),
                "return_code": -1
            }
    
    def _is_allowed_file(self, path_str: str) -> bool:
        """Check if file extension is allowed."""
        if not self.allowed_extensions:
            return True  # No restrictions
        return Path(path_str).suffix.lower() in self.allowed_extensions
    
    def _lint_check(self, content: str) -> Optional[str]:
        """Check Python syntax."""
        try:
            import ast
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)
    
    def get_pod_info(self) -> Dict[str, str]:
        """Get information about the target pod."""
        return {
            "pod_name": self.pod_name,
            "namespace": self.namespace,
            "kubeconfig_path": self.kubeconfig_path or "default"
        }
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """Clean up instance-specific data."""
        if instance_id in self.file_history:
            del self.file_history[instance_id]