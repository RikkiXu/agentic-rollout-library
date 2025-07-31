#!/usr/bin/env python3
"""
File editor tool for viewing, creating, and editing files.
Based on R2E's str_replace_editor but adapted for the core tool framework.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chardet

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class FileEditorTool(AgenticBaseTool):
    """Tool for file operations: view, create, edit, and search."""
    
    def __init__(self, config: Dict = None):
        """Initialize file editor tool."""
        super().__init__(config)
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB default
        self.max_response_length = self.config.get("max_response_length", 10000)
        self.allowed_extensions = set(self.config.get("allowed_extensions", [
            ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".sh", ".js", ".ts", 
            ".html", ".css", ".xml", ".csv", ".log"
        ]))
        self.enable_linting = self.config.get("enable_linting", False)
        self.file_history = {}  # instance_id -> {file_path -> [history]}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for file editor."""
        return create_openai_tool_schema(
            name="file_editor",
            description="File operations tool for viewing, creating, editing files and directories. Supports string replacement, insertion, and file management.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"]
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory"
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
        """Execute file operation."""
        try:
            command = parameters["command"]
            path_str = parameters["path"]
            
            # Validate path
            path = Path(path_str)
            
            if command == "view":
                return await self._view_file(path, parameters.get("view_range"))
            elif command == "create":
                file_text = parameters.get("file_text")
                if file_text is None:
                    return ToolResult(success=False, error="file_text parameter required for create command")
                return await self._create_file(instance_id, path, file_text)
            elif command == "str_replace":
                old_str = parameters.get("old_str")
                new_str = parameters.get("new_str", "")
                if old_str is None:
                    return ToolResult(success=False, error="old_str parameter required for str_replace command")
                return await self._str_replace(instance_id, path, old_str, new_str)
            elif command == "insert":
                insert_line = parameters.get("insert_line")
                new_str = parameters.get("new_str")
                if insert_line is None or new_str is None:
                    return ToolResult(success=False, error="insert_line and new_str parameters required for insert command")
                return await self._insert_text(instance_id, path, insert_line, new_str)
            elif command == "undo_edit":
                return await self._undo_edit(instance_id, path)
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(f"File editor execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _view_file(self, path: Path, view_range: Optional[List[int]] = None) -> ToolResult:
        """View file or directory contents."""
        try:
            if not path.exists():
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            if path.is_dir():
                # List directory contents
                try:
                    files = []
                    for item in path.iterdir():
                        if not item.name.startswith('.'):  # Skip hidden files
                            item_type = "DIR" if item.is_dir() else "FILE"
                            files.append(f"{item_type}: {item.name}")
                    
                    content = f"Directory contents of {path}:\n" + "\n".join(sorted(files))
                    return ToolResult(success=True, result={"content": content, "type": "directory"})
                    
                except Exception as e:
                    return ToolResult(success=False, error=f"Failed to list directory: {e}")
            
            else:
                # View file
                if not self._is_allowed_file(path):
                    return ToolResult(success=False, error=f"File type not allowed: {path.suffix}")
                
                # Check file size
                if path.stat().st_size > self.max_file_size:
                    return ToolResult(success=False, error=f"File too large: {path.stat().st_size} bytes")
                
                # Read file content
                content = self._read_file(path)
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
                        "displayed_lines": len(lines)
                    }
                )
                
        except Exception as e:
            return ToolResult(success=False, error=f"View operation failed: {e}")
    
    async def _create_file(self, instance_id: str, path: Path, file_text: str) -> ToolResult:
        """Create a new file."""
        try:
            if path.exists():
                return ToolResult(success=False, error=f"File already exists: {path}")
            
            if not self._is_allowed_file(path):
                return ToolResult(success=False, error=f"File type not allowed: {path.suffix}")
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lint check for Python files
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(file_text)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Write file
            path.write_text(file_text, encoding="utf-8")
            
            # Initialize history
            if instance_id in self.file_history:
                self.file_history[instance_id][str(path)] = [""]
            
            return ToolResult(
                success=True,
                result={
                    "message": f"File created: {path}",
                    "path": str(path),
                    "size": len(file_text)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Create operation failed: {e}")
    
    async def _str_replace(self, instance_id: str, path: Path, old_str: str, new_str: str) -> ToolResult:
        """Replace string in file."""
        try:
            if not path.exists():
                return ToolResult(success=False, error=f"File does not exist: {path}")
            
            if not self._is_allowed_file(path):
                return ToolResult(success=False, error=f"File type not allowed: {path.suffix}")
            
            # Read current content
            old_content = self._read_file(path)
            
            # Check for occurrences
            occurrences = old_content.count(old_str)
            if occurrences == 0:
                return ToolResult(success=False, error=f"String not found in file: {old_str}")
            elif occurrences > 1:
                return ToolResult(success=False, error=f"Multiple occurrences found ({occurrences}). String must be unique.")
            
            # Perform replacement
            new_content = old_content.replace(old_str, new_str)
            
            # Lint check for Python files
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(new_content)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Save history
            if instance_id in self.file_history:
                if str(path) not in self.file_history[instance_id]:
                    self.file_history[instance_id][str(path)] = []
                self.file_history[instance_id][str(path)].append(old_content)
            
            # Write new content
            path.write_text(new_content, encoding="utf-8")
            
            # Find replacement location for context
            replacement_line = old_content.split(old_str)[0].count('\n')
            context_start = max(0, replacement_line - 2)
            context_end = min(len(new_content.splitlines()), replacement_line + new_str.count('\n') + 3)
            
            context_lines = new_content.splitlines()[context_start:context_end]
            context = "\n".join(f"{context_start + i + 1:6d}  {line}" for i, line in enumerate(context_lines))
            
            return ToolResult(
                success=True,
                result={
                    "message": f"String replaced in {path}",
                    "path": str(path),
                    "context": context,
                    "replacement_line": replacement_line + 1
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"String replace operation failed: {e}")
    
    async def _insert_text(self, instance_id: str, path: Path, insert_line: int, new_str: str) -> ToolResult:
        """Insert text at specified line."""
        try:
            if not path.exists():
                return ToolResult(success=False, error=f"File does not exist: {path}")
            
            if not self._is_allowed_file(path):
                return ToolResult(success=False, error=f"File type not allowed: {path.suffix}")
            
            # Read current content
            old_content = self._read_file(path)
            lines = old_content.splitlines()
            
            if insert_line < 0 or insert_line > len(lines):
                return ToolResult(success=False, error=f"Invalid insert line {insert_line}. File has {len(lines)} lines.")
            
            # Insert text
            new_lines = new_str.splitlines()
            updated_lines = lines[:insert_line] + new_lines + lines[insert_line:]
            new_content = "\n".join(updated_lines)
            
            # Lint check for Python files
            if self.enable_linting and path.suffix == ".py":
                lint_error = self._lint_check(new_content)
                if lint_error:
                    return ToolResult(success=False, error=f"Linting failed: {lint_error}")
            
            # Save history
            if instance_id in self.file_history:
                if str(path) not in self.file_history[instance_id]:
                    self.file_history[instance_id][str(path)] = []
                self.file_history[instance_id][str(path)].append(old_content)
            
            # Write new content
            path.write_text(new_content, encoding="utf-8")
            
            # Generate context
            context_start = max(0, insert_line - 2)
            context_end = min(len(updated_lines), insert_line + len(new_lines) + 3)
            context_lines = updated_lines[context_start:context_end]
            context = "\n".join(f"{context_start + i + 1:6d}  {line}" for i, line in enumerate(context_lines))
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Text inserted in {path} at line {insert_line}",
                    "path": str(path),
                    "context": context,
                    "lines_inserted": len(new_lines)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Insert operation failed: {e}")
    
    async def _undo_edit(self, instance_id: str, path: Path) -> ToolResult:
        """Undo last edit operation."""
        try:
            if instance_id not in self.file_history:
                return ToolResult(success=False, error="No edit history available")
            
            path_str = str(path)
            if path_str not in self.file_history[instance_id] or not self.file_history[instance_id][path_str]:
                return ToolResult(success=False, error=f"No edit history for file: {path}")
            
            # Restore previous content
            previous_content = self.file_history[instance_id][path_str].pop()
            path.write_text(previous_content, encoding="utf-8")
            
            return ToolResult(
                success=True,
                result={
                    "message": f"Undid last edit to {path}",
                    "path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Undo operation failed: {e}")
    
    def _read_file(self, path: Path) -> str:
        """Read file with encoding detection."""
        try:
            # Try to detect encoding
            raw_data = path.read_bytes()
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result["encoding"] or "utf-8"
            
            return path.read_text(encoding=encoding)
        except Exception:
            # Fallback to UTF-8
            return path.read_text(encoding="utf-8", errors="replace")
    
    def _is_allowed_file(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        if not self.allowed_extensions:
            return True  # No restrictions
        return path.suffix.lower() in self.allowed_extensions
    
    def _lint_check(self, content: str) -> Optional[str]:
        """Check Python syntax."""
        try:
            import ast
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """Clean up instance-specific data."""
        if instance_id in self.file_history:
            del self.file_history[instance_id]