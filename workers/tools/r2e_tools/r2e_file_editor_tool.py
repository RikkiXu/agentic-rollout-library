#!/usr/bin/env python3
"""
R2E File Editor tool converted to official tool implementation.
Supports both local file editing and remote K8s pod file editing.
Based on the original R2E file_editor.py with full functionality.
"""

import os
import json
import logging
import subprocess
import chardet
import ast
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

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


# R2E constants
SNIPPET_LINES = 4
MAX_RESPONSE_LEN = 10000
TRUNCATED_MESSAGE = (
    "<response clipped><NOTE>To save on context only part of this file has been "
    "shown to you. You should retry this tool after you have searched inside the file "
    "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)


class R2EFileEditorTool(AgenticBaseTool):
    """R2E-style file editor tool supporting view, create, str_replace, insert, and undo_edit."""
    
    def __init__(self, config: Dict = None):
        """Initialize R2E file editor tool."""
        # Set execution mode and K8s config FIRST
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        # R2E-specific settings
        self.enable_linting = self.config.get("enable_linting", False)
        self.max_response_len = self.config.get("max_response_len", MAX_RESPONSE_LEN)
        self.python_only = self.config.get("python_only", True)
        self.state_file = self.config.get("state_file", "/var/tmp/r2e_editor_state.json")
        
        # File history for undo functionality
        self.file_history = defaultdict(list)
        self._load_history()
        
        self.k8s_manager = None
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode.")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
    
    def get_description(self) -> str:
        """Override to provide custom description for R2E file editor."""
        if self.config.get("use_custom_description", False):
            return """–– BEGIN FUNCTION #1: file_editor ––
Description:
Custom editing tool for viewing, creating and editing files
  •    State is persistent across command calls and discussions with the user
  •    If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
  •    The create command cannot be used if the specified path already exists as a file
  •    If a command generates a long output, it will be truncated and marked with <response clipped>
  •    The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
  •    The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
  •    If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
  •    The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
  1.    command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
  2.    path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
  3.    file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
  4.    old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
  5.    new_str (string, optional)
  •    Optional for the str_replace command to specify the replacement string.
  •    Required for the insert command to specify the string to insert.
  6.    insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
  7.    view_range (array, optional)
  •    Optional for the view command (when path is a file).
  •    If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
  •    [start_line, -1] will show all lines from start_line to the end of file.
  8.    concise (boolean, optional)
  •    Optional for the view command.
  •    Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

–– END FUNCTION #1 ––"""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E file editor."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="r2e_file_editor",
            description=f"R2E-style file editing tool for viewing, creating and editing files{execution_context}. State is persistent across command calls.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The command to run",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"]
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
                    "description": "For 'str_replace' - replacement string; For 'insert' - string to insert"
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required for 'insert' - line number after which to insert new_str"
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional for 'view' - line range [start, end], use -1 for end of file"
                },
                "concise": {
                    "type": "boolean",
                    "description": "Optional for 'view' - condensed view for Python files (default: false)"
                }
            },
            required=["command", "path"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute file editor command."""
        try:
            command = parameters["command"]
            path = parameters["path"]
            
            if command == "view":
                return await self._view(
                    path, 
                    parameters.get("view_range"),
                    parameters.get("concise", False)
                )
            elif command == "create":
                if "file_text" not in parameters:
                    return ToolResult(success=False, error="file_text parameter required for create command")
                return await self._create(path, parameters["file_text"])
            elif command == "str_replace":
                if "old_str" not in parameters:
                    return ToolResult(success=False, error="old_str parameter required for str_replace command")
                return await self._str_replace(
                    path,
                    parameters["old_str"],
                    parameters.get("new_str", "")
                )
            elif command == "insert":
                if "insert_line" not in parameters or "new_str" not in parameters:
                    return ToolResult(success=False, error="insert_line and new_str parameters required for insert command")
                return await self._insert(
                    path,
                    parameters["insert_line"],
                    parameters["new_str"]
                )
            elif command == "undo_edit":
                return await self._undo_edit(path)
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(f"R2E file editor execution failed: {e}", exc_info=True)
            error_msg = f"Tool execution error: {str(e)}"
            if "parameters" in locals():
                error_msg += f"\nCommand: {parameters.get('command', 'N/A')}"
                error_msg += f"\nPath: {parameters.get('path', 'N/A')}"
            return ToolResult(
                success=False, 
                error=error_msg,
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    async def _view(self, path_str: str, view_range: Optional[List[int]], concise: bool) -> ToolResult:
        """View file or directory contents."""
        if self.execution_mode == "k8s":
            return await self._view_k8s(path_str, view_range, concise)
        else:
            return await self._view_local(path_str, view_range, concise)
    
    async def _view_local(self, path_str: str, view_range: Optional[List[int]], concise: bool) -> ToolResult:
        """View file or directory locally."""
        try:
            path = Path(path_str)
            
            if not path.exists():
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            if path.is_dir():
                # List directory contents R2E-style
                if self.python_only:
                    cmd = ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*",
                           "(", "-type", "d", "-o", "-name", "*.py", ")"]
                else:
                    cmd = ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*"]
                
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                except TypeError:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, check=False)
                
                if proc.stderr:
                    return ToolResult(success=False, error=proc.stderr.strip())
                
                msg = (f"Here's the files and directories up to 2 levels deep in {path}, "
                       "excluding hidden:\n" + proc.stdout)
                msg = self._maybe_truncate(msg)
                
                return ToolResult(success=True, result={"output": msg, "type": "directory"})
            
            else:
                # View file
                if self.python_only and path.suffix != ".py":
                    return ToolResult(
                        success=False,
                        error=f"Viewing non-Python files is disallowed. File '{path.name}' is not a .py file."
                    )
                
                # Auto-enable concise mode for large Python files
                if path.suffix == ".py" and not view_range and not concise:
                    file_text = self._read_file_local(path)
                    if len(file_text.splitlines()) > 110:
                        concise = True
                
                # Get file content
                if path.suffix == ".py" and concise:
                    lines_with_numbers = self._get_elided_lines(path)
                else:
                    file_text = self._read_file_local(path)
                    lines_with_numbers = [(i, line) for i, line in enumerate(file_text.splitlines())]
                
                # Apply view range
                lines_with_numbers = self._apply_view_range(lines_with_numbers, view_range)
                
                # Format output R2E-style
                if concise:
                    output = f"Here is a condensed view for file: {path}; [Note: Useful for understanding file structure in a concise manner.]\n"
                else:
                    output = f"Here's the result of running `cat -n` on the file: {path}:\n"
                
                for i, text in lines_with_numbers:
                    output += f"{i+1:6d} {text}\n"
                
                output = self._maybe_truncate(output)
                
                return ToolResult(
                    success=True,
                    result={
                        "output": output,
                        "type": "file",
                        "total_lines": len(lines_with_numbers)
                    }
                )
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _create(self, path_str: str, file_text: str) -> ToolResult:
        """Create a new file."""
        if self.execution_mode == "k8s":
            return await self._create_k8s(path_str, file_text)
        else:
            return await self._create_local(path_str, file_text)
    
    async def _create_local(self, path_str: str, file_text: str) -> ToolResult:
        """Create file locally."""
        try:
            path = Path(path_str)
            
            if path.exists():
                return ToolResult(success=False, error=f"File already exists at: {path}. Cannot overwrite with 'create'.")
            
            # Lint check for Python files
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(file_text)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed:\n{lint_error}")
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(file_text, encoding="utf-8")
            
            # Save to history
            self.file_history[str(path)].append("")
            self._save_history()
            
            # Format output
            output = f"File created at {path}. "
            output += self._make_output(file_text, str(path))
            output += "Review the file and make sure that it is as expected. Edit the file if necessary."
            
            return ToolResult(success=True, result={"output": output, "path": str(path)})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _str_replace(self, path_str: str, old_str: str, new_str: str) -> ToolResult:
        """Replace string in file."""
        if self.execution_mode == "k8s":
            return await self._str_replace_k8s(path_str, old_str, new_str)
        else:
            return await self._str_replace_local(path_str, old_str, new_str)
    
    async def _str_replace_local(self, path_str: str, old_str: str, new_str: str) -> ToolResult:
        """Replace string locally."""
        try:
            path = Path(path_str)
            
            if not path.exists():
                return ToolResult(success=False, error=f"File does not exist: {path}")
            
            # Read file
            file_content = self._read_file_local(path).expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()
            
            # Check occurrences
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                return ToolResult(success=False, error=f"No occurrences of '{old_str}' found in {path}")
            if occurrences > 1:
                return ToolResult(
                    success=False,
                    error=f"Multiple occurrences of '{old_str}' found in {path}. Please ensure it is unique."
                )
            
            # Replace
            updated_text = file_content.replace(old_str, new_str)
            
            # Lint check
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(updated_text)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed:\n{lint_error}")
            
            # Save history and write
            self.file_history[str(path)].append(file_content)
            self._save_history()
            path.write_text(updated_text, encoding="utf-8")
            
            # Create snippet
            replacement_line = file_content.split(old_str)[0].count("\n")
            snippet = self._create_snippet(updated_text, replacement_line, new_str.count("\n"))
            
            output = f"The file {path} has been edited. "
            output += self._make_output(snippet, f"a snippet of {path}", replacement_line - SNIPPET_LINES + 1)
            output += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            
            return ToolResult(success=True, result={"output": output})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _insert(self, path_str: str, insert_line: int, new_str: str) -> ToolResult:
        """Insert text at line."""
        if self.execution_mode == "k8s":
            return await self._insert_k8s(path_str, insert_line, new_str)
        else:
            return await self._insert_local(path_str, insert_line, new_str)
    
    async def _insert_local(self, path_str: str, insert_line: int, new_str: str) -> ToolResult:
        """Insert text locally."""
        try:
            path = Path(path_str)
            
            if not path.exists():
                return ToolResult(success=False, error=f"File does not exist: {path}")
            
            # Read file
            old_text = self._read_file_local(path).expandtabs()
            new_str = new_str.expandtabs()
            file_lines = old_text.split("\n")
            
            # Validate line number
            if insert_line < 0 or insert_line > len(file_lines):
                return ToolResult(
                    success=False,
                    error=f"Invalid insert_line {insert_line}. Must be in [0, {len(file_lines)}]."
                )
            
            # Insert
            new_lines = new_str.split("\n")
            updated_lines = file_lines[:insert_line] + new_lines + file_lines[insert_line:]
            updated_text = "\n".join(updated_lines)
            
            # Lint check
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(updated_text)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed:\n{lint_error}")
            
            # Save history and write
            self.file_history[str(path)].append(old_text)
            self._save_history()
            path.write_text(updated_text, encoding="utf-8")
            
            # Create snippet
            snippet_lines = (
                file_lines[max(0, insert_line - SNIPPET_LINES):insert_line] +
                new_lines +
                file_lines[insert_line:insert_line + SNIPPET_LINES]
            )
            snippet = "\n".join(snippet_lines)
            
            output = f"The file {path} has been edited. "
            output += self._make_output(snippet, "a snippet of the edited file", max(1, insert_line - SNIPPET_LINES + 1))
            output += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
            
            return ToolResult(success=True, result={"output": output})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _undo_edit(self, path_str: str) -> ToolResult:
        """Undo last edit."""
        if self.execution_mode == "k8s":
            return await self._undo_edit_k8s(path_str)
        else:
            return await self._undo_edit_local(path_str)
    
    async def _undo_edit_local(self, path_str: str) -> ToolResult:
        """Undo edit locally."""
        try:
            path = Path(path_str)
            
            if not self.file_history[str(path)]:
                return ToolResult(success=False, error=f"No previous edits found for {path} to undo.")
            
            # Restore previous content
            old_text = self.file_history[str(path)].pop()
            self._save_history()
            
            path.write_text(old_text, encoding="utf-8")
            
            output = f"Last edit to {path} undone successfully. "
            output += self._make_output(old_text, str(path))
            
            return ToolResult(success=True, result={"output": output})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    # K8s implementations
    async def _view_k8s(self, path_str: str, view_range: Optional[List[int]], concise: bool) -> ToolResult:
        """View file or directory in K8s pod."""
        try:
            # Check if path exists
            exists_check = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if exists_check["stdout"].strip() != "exists":
                return ToolResult(success=False, error=f"Path does not exist: {path_str}")
            
            # Check if directory
            is_dir_check = await self._exec_command(f"test -d '{path_str}' && echo 'dir' || echo 'file'")
            is_dir = is_dir_check["stdout"].strip() == "dir"
            
            if is_dir:
                # List directory
                if self.python_only:
                    cmd = f"find '{path_str}' -maxdepth 2 -not -path '*/.*' \\( -type d -o -name '*.py' \\)"
                else:
                    cmd = f"find '{path_str}' -maxdepth 2 -not -path '*/.*'"
                
                result = await self._exec_command(cmd)
                if not result["success"]:
                    return ToolResult(success=False, error=result["stderr"])
                
                msg = (f"Here's the files and directories up to 2 levels deep in {path_str}, "
                       "excluding hidden:\n" + result["stdout"])
                msg = self._maybe_truncate(msg)
                
                return ToolResult(success=True, result={"output": msg, "type": "directory"})
            
            else:
                # View file
                if self.python_only and not path_str.endswith(".py"):
                    return ToolResult(
                        success=False,
                        error=f"Viewing non-Python files is disallowed. File '{path_str}' is not a .py file."
                    )
                
                # Read file content
                cat_result = await self._exec_command(f"cat '{path_str}'")
                if not cat_result["success"]:
                    return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
                
                file_content = cat_result["stdout"]
                lines = file_content.splitlines()
                
                # Apply view range and format output with correct line numbers
                output = f"Here's the result of running `cat -n` on the file: {path_str}:\n"
                
                if view_range and len(view_range) == 2:
                    start_line = view_range[0] - 1  # Convert to 0-based index
                    end_line = view_range[1] if view_range[1] != -1 else len(lines)
                    
                    # Validate range
                    if 0 <= start_line < len(lines):
                        for i in range(start_line, min(end_line, len(lines))):
                            output += f"{i+1:6d} {lines[i]}\n"
                    else:
                        output += "[Invalid view range]\n"
                else:
                    # No view range, show all lines
                    for i, line in enumerate(lines):
                        output += f"{i+1:6d} {line}\n"
                
                if not lines:
                    output += "[Empty file]\n"
                
                output = self._maybe_truncate(output)
                
                # Calculate actual displayed lines count
                if view_range and len(view_range) == 2:
                    start_line = view_range[0] - 1
                    end_line = view_range[1] if view_range[1] != -1 else len(lines)
                    displayed_lines = max(0, min(end_line, len(lines)) - max(0, start_line))
                else:
                    displayed_lines = len(lines)
                
                return ToolResult(
                    success=True,
                    result={
                        "output": output,
                        "type": "file",
                        "total_lines": displayed_lines,
                        "file_total_lines": len(lines)
                    }
                )
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _create_k8s(self, path_str: str, file_text: str) -> ToolResult:
        """Create file in K8s pod."""
        try:
            # Check if exists
            exists_check = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if exists_check["stdout"].strip() == "exists":
                return ToolResult(success=False, error=f"File already exists at: {path_str}")
            
            # Create parent directories
            parent_dir = os.path.dirname(path_str)
            if parent_dir:
                mkdir_result = await self._exec_command(f"mkdir -p '{parent_dir}'")
                if not mkdir_result["success"]:
                    return ToolResult(success=False, error=f"Failed to create parent directories: {mkdir_result['stderr']}")
            
            # Write file using base64 to handle special characters safely
            encoded_text = base64.b64encode(file_text.encode('utf-8')).decode('ascii')
            write_result = await self._exec_command(f"echo '{encoded_text}' | base64 -d > '{path_str}'")
            if not write_result["success"]:
                # Fallback: try with printf
                escaped_text = file_text.replace("'", "'\"'\"'").replace("\\", "\\\\")
                write_result = await self._exec_command(f"printf '%s' '{escaped_text}' > '{path_str}'")
                if not write_result["success"]:
                    return ToolResult(success=False, error=f"Failed to create file: {write_result['stderr']}")
            
            # Format output
            output = f"File created at {path_str}. "
            output += self._make_output(file_text, str(path_str))
            output += "Review the file and make sure that it is as expected. Edit the file if necessary."
            
            return ToolResult(success=True, result={"output": output, "path": path_str})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _str_replace_k8s(self, path_str: str, old_str: str, new_str: str) -> ToolResult:
        """Replace string in K8s pod."""
        try:
            # Read file
            cat_result = await self._exec_command(f"cat '{path_str}'")
            if not cat_result["success"]:
                return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
            
            file_content = cat_result["stdout"].expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()
            
            # Check occurrences
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                return ToolResult(success=False, error=f"No occurrences of '{old_str}' found in {path_str}")
            if occurrences > 1:
                return ToolResult(
                    success=False,
                    error=f"Multiple occurrences of '{old_str}' found in {path_str}. Please ensure it is unique."
                )
            
            # Replace
            updated_text = file_content.replace(old_str, new_str)
            
            # Write back using base64 to handle special characters
            encoded_text = base64.b64encode(updated_text.encode('utf-8')).decode('ascii')
            write_result = await self._exec_command(f"echo '{encoded_text}' | base64 -d > '{path_str}'")
            if not write_result["success"]:
                # Fallback: try with printf
                escaped_text = updated_text.replace("'", "'\"'\"'").replace("\\", "\\\\")
                write_result = await self._exec_command(f"printf '%s' '{escaped_text}' > '{path_str}'")
                if not write_result["success"]:
                    return ToolResult(success=False, error=f"Failed to write file: {write_result['stderr']}")
            
            # Create snippet
            replacement_line = file_content.split(old_str)[0].count("\n")
            snippet = self._create_snippet(updated_text, replacement_line, new_str.count("\n"))
            
            output = f"The file {path_str} has been edited. "
            output += self._make_output(snippet, f"a snippet of {path_str}", replacement_line - SNIPPET_LINES + 1)
            output += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            
            return ToolResult(success=True, result={"output": output})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _insert_k8s(self, path_str: str, insert_line: int, new_str: str) -> ToolResult:
        """Insert text in K8s pod."""
        try:
            # Read file
            cat_result = await self._exec_command(f"cat '{path_str}'")
            if not cat_result["success"]:
                return ToolResult(success=False, error=f"Failed to read file: {cat_result['stderr']}")
            
            old_text = cat_result["stdout"].expandtabs()
            new_str = new_str.expandtabs()
            file_lines = old_text.split("\n")
            
            # Validate line number
            if insert_line < 0 or insert_line > len(file_lines):
                return ToolResult(
                    success=False,
                    error=f"Invalid insert_line {insert_line}. Must be in [0, {len(file_lines)}]."
                )
            
            # Insert
            new_lines = new_str.split("\n")
            updated_lines = file_lines[:insert_line] + new_lines + file_lines[insert_line:]
            updated_text = "\n".join(updated_lines)
            
            # Write back using base64 to handle special characters
            encoded_text = base64.b64encode(updated_text.encode('utf-8')).decode('ascii')
            write_result = await self._exec_command(f"echo '{encoded_text}' | base64 -d > '{path_str}'")
            if not write_result["success"]:
                # Fallback: try with printf
                escaped_text = updated_text.replace("'", "'\"'\"'").replace("\\", "\\\\")
                write_result = await self._exec_command(f"printf '%s' '{escaped_text}' > '{path_str}'")
                if not write_result["success"]:
                    return ToolResult(success=False, error=f"Failed to write file: {write_result['stderr']}")
            
            # Create snippet
            snippet_lines = (
                file_lines[max(0, insert_line - SNIPPET_LINES):insert_line] +
                new_lines +
                file_lines[insert_line:insert_line + SNIPPET_LINES]
            )
            snippet = "\n".join(snippet_lines)
            
            output = f"The file {path_str} has been edited. "
            output += self._make_output(snippet, "a snippet of the edited file", max(1, insert_line - SNIPPET_LINES + 1))
            output += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            
            return ToolResult(success=True, result={"output": output})
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _undo_edit_k8s(self, path_str: str) -> ToolResult:
        """Undo not supported in K8s mode."""
        return ToolResult(
            success=False,
            error="Undo functionality is not available in K8s mode. File history is only maintained locally."
        )
    
    # Helper methods
    def _read_file_local(self, path: Path) -> str:
        """Read file with encoding detection."""
        try:
            encoding = chardet.detect(path.read_bytes())["encoding"]
            if encoding is None:
                encoding = "utf-8"
            return path.read_text(encoding=encoding)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
    
    def _maybe_truncate(self, content: str) -> str:
        """Truncate content if too long."""
        if len(content) <= self.max_response_len:
            return content
        return content[:self.max_response_len] + TRUNCATED_MESSAGE
    
    def _make_output(self, file_content: str, file_descriptor: str, init_line: int = 1) -> str:
        """Format output R2E-style."""
        file_content = self._maybe_truncate(file_content)
        file_content = file_content.expandtabs()
        
        lines = file_content.split("\n")
        numbered = "\n".join(f"{i + init_line:6}\t{line}" for i, line in enumerate(lines))
        return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + numbered + "\n"
    
    def _create_snippet(self, text: str, center_line: int, extra_lines: int = 0) -> str:
        """Create snippet around a line."""
        lines = text.split("\n")
        start = max(0, center_line - SNIPPET_LINES)
        end = center_line + SNIPPET_LINES + extra_lines + 1
        return "\n".join(lines[start:end])
    
    def _apply_view_range(self, lines_with_numbers: List[Tuple[int, str]], 
                         view_range: Optional[List[int]]) -> List[Tuple[int, str]]:
        """Apply view range to lines."""
        if not view_range or len(view_range) != 2:
            return lines_with_numbers
        
        start, end = view_range
        total_lines = len(lines_with_numbers)
        
        if not (1 <= start <= total_lines):
            return lines_with_numbers
        
        if end == -1:
            end = total_lines
        elif end < start or end > total_lines:
            return lines_with_numbers
        
        # Filter by 1-based index
        result = []
        for i, text in lines_with_numbers:
            one_based = i + 1
            if start <= one_based <= end:
                result.append((i, text))
        
        return result
    
    
    def _lint_check(self, content: str) -> Optional[str]:
        """Check Python syntax."""
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)
    
    def _get_elided_lines(self, path: Path) -> List[Tuple[int, str]]:
        """Get condensed view of Python file (R2E-style)."""
        import ast
        
        file_text = self._read_file_local(path)
        try:
            tree = ast.parse(file_text, filename=str(path))
        except SyntaxError as e:
            raise Exception(f"Syntax error in file {path}: {e}")
        
        def max_lineno_in_subtree(n: ast.AST) -> int:
            m = getattr(n, "lineno", 0)
            for child in ast.iter_child_nodes(n):
                m = max(m, max_lineno_in_subtree(child))
            return m
        
        # Gather line ranges for large function bodies
        elide_line_ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                last_stmt = node.body[-1]
                if hasattr(last_stmt, "end_lineno") and last_stmt.end_lineno:
                    body_start = node.body[0].lineno - 1
                    body_end = last_stmt.end_lineno - 1
                else:
                    body_start = node.body[0].lineno - 1
                    body_end = max_lineno_in_subtree(last_stmt) - 1
                
                if (body_end - body_start) >= 3:
                    elide_line_ranges.append((body_start, body_end))
        
        # Build elided view
        elide_lines = {
            line for (start, end) in elide_line_ranges 
            for line in range(start, end + 1)
        }
        
        elide_messages = [
            (start, f"... eliding lines {start+1}-{end+1} ...")
            for (start, end) in elide_line_ranges
        ]
        
        all_lines = file_text.splitlines()
        keep_lines = [
            (i, line) for i, line in enumerate(all_lines) 
            if i not in elide_lines
        ]
        
        combined = elide_messages + keep_lines
        combined.sort(key=lambda x: x[0])
        
        return combined
    
    def _load_history(self):
        """Load file edit history."""
        try:
            if self.execution_mode == "local" and Path(self.state_file).exists():
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.file_history = defaultdict(list, data)
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
    
    def _save_history(self):
        """Save file edit history."""
        try:
            if self.execution_mode == "local":
                with open(self.state_file, "w") as f:
                    json.dump(dict(self.file_history), f)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
    
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
            
            # Return consistent format
            return {
                "success": exit_code_int == 0,
                "stdout": output if output else "",
                "stderr": "" if exit_code_int == 0 else f"Command failed with exit code {exit_code_int}",
                "exit_code": exit_code_int
            }
        except Exception as e:
            logger.error(f"K8s command execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get execution environment info."""
        info = {
            "execution_mode": self.execution_mode,
            "enable_linting": self.enable_linting,
            "max_response_len": self.max_response_len,
            "python_only": self.python_only,
            "tool_style": "R2E"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info