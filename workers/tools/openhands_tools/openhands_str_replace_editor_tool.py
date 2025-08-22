#!/usr/bin/env python3
"""
OpenHands String Replace Editor tool converted to official tool implementation.
Supports both local file operations and remote K8s pod file operations.
Based on the original OpenHands str_replace_editor tool.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

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


class OpenHandsStrReplaceEditorTool(AgenticBaseTool):
    """OpenHands-style string replace editor tool with K8s support."""
    
    def __init__(self, config: Dict = None):
        """Initialize OpenHands string replace editor tool."""
        # Set execution mode and K8s config FIRST, before calling super().__init__
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        self.working_dir = config.get("working_dir", "/workspace")  # OpenHands default
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode. Please install it from https://github.com/baidubce/kodo.git")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
        
        super().__init__(config)
        
        # OpenHands-specific settings
        self.use_short_description = self.config.get("use_short_description", False)
        self.state_file = self.config.get("state_file", "/var/tmp/openhands_editor_state.json")
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB default
        self.k8s_manager = None
        
        # Initialize state
        self._init_state()
    
    def _init_state(self):
        """Initialize editor state."""
        self.file_states = {}  # Track file edit history
        
    def get_description(self) -> str:
        """Override to provide custom description for OpenHands editor."""
        # Check if we want to use custom OpenHands-style description
        if self.config.get("use_custom_description", False):
            if self.use_short_description:
                return """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`"""
            else:
                return """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"]. IT DOES NOT HANDLE IMAGES.
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* This tool can be used for creating and editing files in plain-text format.

Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each."""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for OpenHands str replace editor."""
        execution_context = f" (operating in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="openhands_str_replace_editor",
            description=f"Custom editing tool for viewing, creating and editing files{execution_context}. OpenHands-style tool with persistent state.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"]
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`."
                },
                "file_text": {
                    "type": "string",
                    "description": "Required parameter of `create` command, with the content of the file to be created."
                },
                "old_str": {
                    "type": "string",
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace."
                },
                "new_str": {
                    "type": "string",
                    "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert."
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`."
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file."
                }
            },
            required=["command", "path"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute str replace editor command."""
        try:
            command = parameters.get("command")
            path = parameters.get("path")
            
            if not command or not path:
                return ToolResult(
                    success=False,
                    error="Missing required parameters 'command' and 'path'."
                )
            
            # Normalize path to absolute path within working directory
            if not path.startswith("/"):
                path = os.path.join(self.working_dir, path)
            
            # Execute command based on type
            if command == "view":
                return await self._handle_view(path, parameters.get("view_range"))
            elif command == "create":
                return await self._handle_create(path, parameters.get("file_text", ""))
            elif command == "str_replace":
                return await self._handle_str_replace(
                    path, 
                    parameters.get("old_str"), 
                    parameters.get("new_str", "")
                )
            elif command == "insert":
                return await self._handle_insert(
                    path, 
                    parameters.get("new_str"), 
                    parameters.get("insert_line")
                )
            elif command == "undo_edit":
                return await self._handle_undo_edit(path)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown command: {command}. Allowed commands are: view, create, str_replace, insert, undo_edit"
                )
                
        except Exception as e:
            logger.error(f"OpenHands editor execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False, 
                error=f"Tool execution error: {str(e)}",
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    async def _handle_view(self, path: str, view_range: Optional[List[int]] = None) -> ToolResult:
        """Handle view command."""
        try:
            if self.execution_mode == "k8s":
                return await self._k8s_view(path, view_range)
            else:
                return await self._local_view(path, view_range)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"View operation failed: {str(e)}"
            )
    
    async def _local_view(self, path: str, view_range: Optional[List[int]] = None) -> ToolResult:
        """View file or directory locally."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"Path does not exist: {path}"
            )
        
        if os.path.isdir(path):
            # List directory contents
            try:
                entries = []
                for root, dirs, files in os.walk(path):
                    level = root.replace(path, '').count(os.sep)
                    if level >= 2:  # Limit to 2 levels deep
                        dirs[:] = []  # Don't go deeper
                        continue
                    
                    indent = ' ' * 2 * level
                    entries.append(f'{indent}{os.path.basename(root)}/')
                    
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        if not file.startswith('.'):  # Skip hidden files
                            entries.append(f'{subindent}{file}')
                
                content = '\n'.join(entries[:100])  # Limit output
                if len(entries) > 100:
                    content += '\n... (truncated)'
                
                return ToolResult(
                    success=True,
                    result={
                        "content": content,
                        "type": "directory",
                        "path": path
                    }
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to list directory: {str(e)}"
                )
        else:
            # View file content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if view_range:
                    start_line = max(1, view_range[0]) - 1  # Convert to 0-based
                    end_line = view_range[1] if view_range[1] != -1 else len(lines)
                    lines = lines[start_line:end_line]
                    start_num = start_line + 1
                else:
                    start_num = 1
                
                # Add line numbers
                numbered_lines = []
                for i, line in enumerate(lines):
                    line_num = start_num + i
                    numbered_lines.append(f"{line_num:6d}|{line.rstrip()}")
                
                content = '\n'.join(numbered_lines)
                
                # Truncate if too long
                if len(content) > 50000:  # 50KB limit
                    content = content[:50000] + '\n<response clipped>'
                
                return ToolResult(
                    success=True,
                    result={
                        "content": content,
                        "type": "file",
                        "path": path,
                        "line_count": len(lines)
                    }
                )
                
            except UnicodeDecodeError:
                return ToolResult(
                    success=False,
                    error=f"Cannot view binary file: {path}"
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to read file: {str(e)}"
                )
    
    async def _k8s_view(self, path: str, view_range: Optional[List[int]] = None) -> ToolResult:
        """View file or directory in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Check if path exists and is file or directory
            test_cmd = f"test -e {path} && echo 'exists' || echo 'not_found'"
            output, exit_code = k8s_mgr.execute_command(self.pod_name, test_cmd)
            
            if "not_found" in output:
                return ToolResult(
                    success=False,
                    error=f"Path does not exist: {path}"
                )
            
            # Check if it's a directory
            dir_test_cmd = f"test -d {path} && echo 'directory' || echo 'file'"
            output, _ = k8s_mgr.execute_command(self.pod_name, dir_test_cmd)
            
            if "directory" in output:
                # List directory
                list_cmd = f"find {path} -maxdepth 2 -type f -o -type d | head -100"
                output, exit_code = k8s_mgr.execute_command(self.pod_name, list_cmd)
                
                if exit_code == 0:
                    return ToolResult(
                        success=True,
                        result={
                            "content": output.strip(),
                            "type": "directory",
                            "path": path
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=f"Failed to list directory: {output}"
                    )
            else:
                # View file
                if view_range:
                    start_line = max(1, view_range[0])
                    if view_range[1] == -1:
                        view_cmd = f"cat -n {path} | tail -n +{start_line}"
                    else:
                        end_line = view_range[1]
                        line_count = end_line - start_line + 1
                        view_cmd = f"cat -n {path} | tail -n +{start_line} | head -n {line_count}"
                else:
                    view_cmd = f"cat -n {path}"
                
                output, exit_code = k8s_mgr.execute_command(self.pod_name, view_cmd)
                
                if exit_code == 0:
                    # Truncate if too long
                    if len(output) > 50000:
                        output = output[:50000] + '\n<response clipped>'
                    
                    return ToolResult(
                        success=True,
                        result={
                            "content": output.strip(),
                            "type": "file",
                            "path": path
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=f"Failed to read file: {output}"
                    )
                    
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"K8s view operation failed: {str(e)}"
            )
    
    async def _handle_create(self, path: str, file_text: str) -> ToolResult:
        """Handle create command."""
        try:
            if self.execution_mode == "k8s":
                return await self._k8s_create(path, file_text)
            else:
                return await self._local_create(path, file_text)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Create operation failed: {str(e)}"
            )
    
    async def _local_create(self, path: str, file_text: str) -> ToolResult:
        """Create file locally."""
        if os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"File already exists: {path}"
            )
        
        try:
            # Create parent directories if needed
            parent_dir = os.path.dirname(path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(file_text)
            
            return ToolResult(
                success=True,
                result={
                    "message": f"File created successfully: {path}",
                    "path": path,
                    "size": len(file_text)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create file: {str(e)}"
            )
    
    async def _k8s_create(self, path: str, file_text: str) -> ToolResult:
        """Create file in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Check if file already exists
            test_cmd = f"test -f {path} && echo 'exists' || echo 'not_found'"
            output, _ = k8s_mgr.execute_command(self.pod_name, test_cmd)
            
            if "exists" in output:
                return ToolResult(
                    success=False,
                    error=f"File already exists: {path}"
                )
            
            # Create parent directories
            parent_dir = os.path.dirname(path)
            mkdir_cmd = f"mkdir -p {parent_dir}"
            k8s_mgr.execute_command(self.pod_name, mkdir_cmd)
            
            # Create file with content
            # Use base64 encoding to handle special characters
            import base64
            encoded_content = base64.b64encode(file_text.encode('utf-8')).decode('ascii')
            create_cmd = f"echo '{encoded_content}' | base64 -d > {path}"
            
            output, exit_code = k8s_mgr.execute_command(self.pod_name, create_cmd)
            
            if exit_code == 0:
                return ToolResult(
                    success=True,
                    result={
                        "message": f"File created successfully: {path}",
                        "path": path,
                        "size": len(file_text)
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Failed to create file: {output}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"K8s create operation failed: {str(e)}"
            )
    
    async def _handle_str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """Handle str_replace command."""
        if not old_str:
            return ToolResult(
                success=False,
                error="Missing required parameter 'old_str' for str_replace command."
            )
        
        try:
            if self.execution_mode == "k8s":
                return await self._k8s_str_replace(path, old_str, new_str)
            else:
                return await self._local_str_replace(path, old_str, new_str)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"String replace operation failed: {str(e)}"
            )
    
    async def _local_str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """String replace locally."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"File does not exist: {path}"
            )
        
        try:
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Save backup for undo
            if path not in self.file_states:
                self.file_states[path] = []
            self.file_states[path].append(content)
            
            # Count occurrences
            count = content.count(old_str)
            if count == 0:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {old_str}"
                )
            elif count > 1:
                return ToolResult(
                    success=False,
                    error=f"String found {count} times in file. Make old_str more specific to uniquely identify the target."
                )
            
            # Perform replacement
            new_content = content.replace(old_str, new_str)
            
            # Write back to file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return ToolResult(
                success=True,
                result={
                    "message": f"String replaced successfully in {path}",
                    "path": path,
                    "old_str": old_str[:100] + "..." if len(old_str) > 100 else old_str,
                    "new_str": new_str[:100] + "..." if len(new_str) > 100 else new_str,
                    "changes": 1
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to replace string: {str(e)}"
            )
    
    async def _k8s_str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """String replace in K8s pod using a more robust approach."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Read file content
            read_cmd = f"cat {path}"
            content, exit_code = k8s_mgr.execute_command(self.pod_name, read_cmd)
            
            if exit_code != 0:
                return ToolResult(
                    success=False,
                    error=f"Failed to read file: {content}"
                )
            
            # Count occurrences
            count = content.count(old_str)
            if count == 0:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {old_str}"
                )
            elif count > 1:
                return ToolResult(
                    success=False,
                    error=f"String found {count} times in file. Make old_str more specific to uniquely identify the target."
                )
            
            # Perform replacement
            new_content = content.replace(old_str, new_str)
            
            # Write back using base64 encoding to handle special characters
            import base64
            encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
            write_cmd = f"echo '{encoded_content}' | base64 -d > {path}"
            
            output, exit_code = k8s_mgr.execute_command(self.pod_name, write_cmd)
            
            if exit_code == 0:
                return ToolResult(
                    success=True,
                    result={
                        "message": f"String replaced successfully in {path}",
                        "path": path,
                        "old_str": old_str[:100] + "..." if len(old_str) > 100 else old_str,
                        "new_str": new_str[:100] + "..." if len(new_str) > 100 else new_str,
                        "changes": 1
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Failed to write file: {output}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"K8s string replace operation failed: {str(e)}"
            )
    
    async def _handle_insert(self, path: str, new_str: str, insert_line: int) -> ToolResult:
        """Handle insert command."""
        if new_str is None:
            return ToolResult(
                success=False,
                error="Missing required parameter 'new_str' for insert command."
            )
        
        if insert_line is None:
            return ToolResult(
                success=False,
                error="Missing required parameter 'insert_line' for insert command."
            )
        
        try:
            if self.execution_mode == "k8s":
                return await self._k8s_insert(path, new_str, insert_line)
            else:
                return await self._local_insert(path, new_str, insert_line)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Insert operation failed: {str(e)}"
            )
    
    async def _local_insert(self, path: str, new_str: str, insert_line: int) -> ToolResult:
        """Insert text locally."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"File does not exist: {path}"
            )
        
        try:
            # Read file lines
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Save backup for undo
            if path not in self.file_states:
                self.file_states[path] = []
            self.file_states[path].append(''.join(lines))
            
            # Validate insert line
            if insert_line < 0 or insert_line > len(lines):
                return ToolResult(
                    success=False,
                    error=f"Invalid insert_line {insert_line}. File has {len(lines)} lines."
                )
            
            # Insert new text
            if not new_str.endswith('\n'):
                new_str += '\n'
            
            lines.insert(insert_line, new_str)
            
            # Write back to file
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Text inserted successfully at line {insert_line} in {path}",
                    "path": path,
                    "insert_line": insert_line,
                    "inserted_text": new_str[:100] + "..." if len(new_str) > 100 else new_str
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to insert text: {str(e)}"
            )
    
    async def _k8s_insert(self, path: str, new_str: str, insert_line: int) -> ToolResult:
        """Insert text in K8s pod."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Read file content
            read_cmd = f"cat {path}"
            content, exit_code = k8s_mgr.execute_command(self.pod_name, read_cmd)
            
            if exit_code != 0:
                return ToolResult(
                    success=False,
                    error=f"Failed to read file: {content}"
                )
            
            lines = content.splitlines(keepends=True)
            
            # Validate insert line
            if insert_line < 0 or insert_line > len(lines):
                return ToolResult(
                    success=False,
                    error=f"Invalid insert_line {insert_line}. File has {len(lines)} lines."
                )
            
            # Insert new text
            if not new_str.endswith('\n'):
                new_str += '\n'
            
            lines.insert(insert_line, new_str)
            new_content = ''.join(lines)
            
            # Write back using base64 encoding
            import base64
            encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
            write_cmd = f"echo '{encoded_content}' | base64 -d > {path}"
            
            output, exit_code = k8s_mgr.execute_command(self.pod_name, write_cmd)
            
            if exit_code == 0:
                return ToolResult(
                    success=True,
                    result={
                        "message": f"Text inserted successfully at line {insert_line} in {path}",
                        "path": path,
                        "insert_line": insert_line,
                        "inserted_text": new_str[:100] + "..." if len(new_str) > 100 else new_str
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Failed to write file: {output}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"K8s insert operation failed: {str(e)}"
            )
    
    async def _handle_undo_edit(self, path: str) -> ToolResult:
        """Handle undo_edit command."""
        if path not in self.file_states or not self.file_states[path]:
            return ToolResult(
                success=False,
                error=f"No edit history available for {path}"
            )
        
        try:
            if self.execution_mode == "k8s":
                return await self._k8s_undo_edit(path)
            else:
                return await self._local_undo_edit(path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Undo operation failed: {str(e)}"
            )
    
    async def _local_undo_edit(self, path: str) -> ToolResult:
        """Undo last edit locally."""
        try:
            # Restore previous content
            previous_content = self.file_states[path].pop()
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(previous_content)
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Successfully undid last edit to {path}",
                    "path": path
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to undo edit: {str(e)}"
            )
    
    async def _k8s_undo_edit(self, path: str) -> ToolResult:
        """Undo last edit in K8s pod."""
        try:
            # For K8s, we would need to implement a more sophisticated backup system
            # This is a simplified implementation
            return ToolResult(
                success=False,
                error="Undo operation is not fully implemented for K8s mode. Please manually revert changes."
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"K8s undo operation failed: {str(e)}"
            )
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        info = {
            "execution_mode": self.execution_mode,
            "working_dir": self.working_dir,
            "max_file_size": self.max_file_size,
            "tool_style": "OpenHands"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info
