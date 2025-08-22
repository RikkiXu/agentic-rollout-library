#!/usr/bin/env python3
"""
Search tool for finding text in files and directories.
Provides grep-like functionality and directory search capabilities.
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional K8s support
try:
    from kodo import KubernetesManager
    K8S_AVAILABLE = True
except ImportError:
    KubernetesManager = None
    K8S_AVAILABLE = False

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class SearchTool(AgenticBaseTool):
    """Tool for searching text in files and directories."""
    
    def __init__(self, config: Dict = None):
        """Initialize search tool."""
        # Set execution mode and K8s config FIRST, before calling super().__init__
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        self.max_results = self.config.get("max_results", 100)
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB
        self.search_extensions = set(self.config.get("search_extensions", [
            ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".sh", ".js", ".ts",
            ".html", ".css", ".xml", ".csv", ".log", ".c", ".cpp", ".h", ".java"
        ]))
        self.exclude_dirs = set(self.config.get("exclude_dirs", [
            ".git", ".svn", "__pycache__", "node_modules", ".pytest_cache",
            ".mypy_cache", ".tox", "venv", ".venv", "env", ".env"
        ]))
        self.k8s_manager = None
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode. Please install it from https://github.com/baidubce/kodo.git")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for search tool."""
        # Description is the same regardless of execution mode
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="search",
            description=f"Search for text patterns in files and directories. Supports regex patterns and various search modes{execution_context}.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Search command type",
                    "enum": ["search_text", "search_files", "search_dir"]
                },
                "pattern": {
                    "type": "string",
                    "description": "Text pattern or regex to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search should be case sensitive (default: false)"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regular expression (default: false)"
                },
                "whole_words": {
                    "type": "boolean",
                    "description": "Match whole words only (default: false)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to search (e.g., ['.py', '.txt'])"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches (default: 2)"
                }
            },
            required=["command", "pattern", "path"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute search operation."""
        try:
            command = parameters["command"]
            pattern = parameters["pattern"]
            path_str = parameters["path"]
            
            # Validate path based on execution mode
            path = Path(path_str)
            if self.execution_mode == "local":
                if not path.exists():
                    return ToolResult(success=False, error=f"Path does not exist: {path}")
            else:
                # For K8s mode, we'll check existence during execution
                pass
            
            # Parse search options
            case_sensitive = parameters.get("case_sensitive", False)
            is_regex = parameters.get("regex", False)
            whole_words = parameters.get("whole_words", False)
            max_results = parameters.get("max_results", self.max_results)
            file_extensions = parameters.get("file_extensions", list(self.search_extensions))
            context_lines = parameters.get("context_lines", 2)
            
            if command == "search_text":
                if self.execution_mode == "k8s":
                    return await self._search_text_in_files_k8s(
                        pattern, path, case_sensitive, is_regex, whole_words,
                        max_results, file_extensions, context_lines
                    )
                else:
                    return await self._search_text_in_files(
                        pattern, path, case_sensitive, is_regex, whole_words,
                        max_results, file_extensions, context_lines
                    )
            elif command == "search_files":
                if self.execution_mode == "k8s":
                    return await self._search_file_names_k8s(
                        pattern, path, case_sensitive, is_regex, max_results
                    )
                else:
                    return await self._search_file_names(
                        pattern, path, case_sensitive, is_regex, max_results
                    )
            elif command == "search_dir":
                if self.execution_mode == "k8s":
                    return await self._search_directory_structure_k8s(
                        pattern, path, case_sensitive, is_regex, max_results
                    )
                else:
                    return await self._search_directory_structure(
                        pattern, path, case_sensitive, is_regex, max_results
                    )
            else:
                return ToolResult(success=False, error=f"Unknown search command: {command}")
                
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_text_in_files(self, pattern: str, path: Path, case_sensitive: bool,
                                   is_regex: bool, whole_words: bool, max_results: int,
                                   file_extensions: List[str], context_lines: int) -> ToolResult:
        """Search for text pattern in files."""
        try:
            # Compile search pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            elif whole_words:
                search_pattern = re.compile(r'\b' + re.escape(pattern) + r'\b', flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            files_searched = 0
            
            if path.is_file():
                # Search single file
                if self._should_search_file(path, file_extensions):
                    file_results = self._search_in_file(path, search_pattern, context_lines)
                    results.extend(file_results)
                    files_searched = 1
            else:
                # Search directory
                for file_path in self._walk_directory(path, file_extensions):
                    if len(results) >= max_results:
                        break
                    
                    file_results = self._search_in_file(file_path, search_pattern, context_lines)
                    results.extend(file_results[:max_results - len(results)])
                    files_searched += 1
            
            return ToolResult(
                success=True,
                result={
                    "matches": results[:max_results],
                    "total_matches": len(results),
                    "files_searched": files_searched,
                    "pattern": pattern,
                    "search_path": str(path)
                },
                metrics={
                    "matches_found": len(results),
                    "files_searched": files_searched
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Text search failed: {e}")
    
    async def _search_file_names(self, pattern: str, path: Path, case_sensitive: bool,
                                is_regex: bool, max_results: int) -> ToolResult:
        """Search for files matching name pattern."""
        try:
            # Compile pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            
            if path.is_file():
                # Check single file
                if search_pattern.search(path.name):
                    results.append({
                        "path": str(path),
                        "name": path.name,
                        "type": "file",
                        "size": path.stat().st_size
                    })
            else:
                # Search directory
                for item in self._walk_all_items(path):
                    if len(results) >= max_results:
                        break
                    
                    if search_pattern.search(item.name):
                        results.append({
                            "path": str(item),
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None
                        })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"File name search failed: {e}")
    
    async def _search_directory_structure(self, pattern: str, path: Path, case_sensitive: bool,
                                         is_regex: bool, max_results: int) -> ToolResult:
        """Search directory structure and list matching paths."""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            
            for item in self._walk_all_items(path):
                if len(results) >= max_results:
                    break
                
                # Search in full path
                if search_pattern.search(str(item)):
                    results.append({
                        "path": str(item),
                        "relative_path": str(item.relative_to(path)),
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "depth": len(item.relative_to(path).parts)
                    })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Directory search failed: {e}")
    
    def _search_in_file(self, file_path: Path, pattern: re.Pattern, context_lines: int) -> List[Dict]:
        """Search for pattern in a single file."""
        results = []
        
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return results
            
            # Read file content
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='latin-1')
                except:
                    return results  # Skip binary files
            
            lines = content.splitlines()
            
            # Search each line
            for line_num, line in enumerate(lines, 1):
                matches = list(pattern.finditer(line))
                
                for match in matches:
                    # Get context lines
                    context_start = max(0, line_num - context_lines - 1)
                    context_end = min(len(lines), line_num + context_lines)
                    
                    context = []
                    for i in range(context_start, context_end):
                        prefix = ">>> " if i == line_num - 1 else "    "
                        context.append(f"{prefix}{i + 1:4d}: {lines[i]}")
                    
                    results.append({
                        "file": str(file_path),
                        "line_number": line_num,
                        "column": match.start() + 1,
                        "match_text": match.group(),
                        "line_content": line,
                        "context": "\n".join(context)
                    })
            
        except Exception as e:
            logger.warning(f"Failed to search in file {file_path}: {e}")
        
        return results
    
    def _walk_directory(self, path: Path, file_extensions: List[str]):
        """Walk directory and yield files to search."""
        for root, dirs, files in os.walk(path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                if self._should_search_file(file_path, file_extensions):
                    yield file_path
    
    def _walk_all_items(self, path: Path):
        """Walk directory and yield all items (files and directories)."""
        for root, dirs, files in os.walk(path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            root_path = Path(root)
            
            # Yield directories
            for dir_name in dirs:
                yield root_path / dir_name
            
            # Yield files
            for file_name in files:
                yield root_path / file_name
    
    def _should_search_file(self, file_path: Path, file_extensions: List[str]) -> bool:
        """Check if file should be searched based on extension and size."""
        if not file_extensions:
            return True
        
        if file_path.suffix.lower() in file_extensions:
            try:
                return file_path.stat().st_size <= self.max_file_size
            except:
                return False
        
        return False
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager

    async def _exec_command(self, command: str) -> Dict[str, Any]:
        """Execute command in K8s pod and return results."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Execute command in pod using kodo API
            output, exit_code = k8s_mgr.execute_command(self.pod_name, command)
            
            return {
                "success": exit_code == 0,
                "stdout": output,
                "stderr": "",  # kodo API doesn't separate stderr
                "return_code": exit_code
            }
            
        except Exception as e:
            logger.error(f"K8s command execution failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }

    async def _search_text_in_files_k8s(self, pattern: str, path: Path, case_sensitive: bool,
                                       is_regex: bool, whole_words: bool, max_results: int,
                                       file_extensions: List[str], context_lines: int) -> ToolResult:
        """Search for text pattern in files using K8s."""
        try:
            path_str = str(path)
            
            # Check if path exists
            exists_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if not exists_result["success"] or exists_result["stdout"].strip() != "exists":
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            # Build grep command
            grep_flags = []
            if not case_sensitive:
                grep_flags.append("-i")
            if not is_regex:
                grep_flags.append("-F")  # Fixed strings
            if whole_words:
                grep_flags.append("-w")
            
            grep_flags.extend(["-n", "-H"])  # Line numbers and filenames
            if context_lines > 0:
                grep_flags.append(f"-C{context_lines}")
            
            # Build file extension filter
            if file_extensions:
                ext_patterns = " -o ".join([f"-name '*{ext}'" for ext in file_extensions])
                find_cmd = f"find '{path_str}' -type f \\( {ext_patterns} \\)"
            else:
                find_cmd = f"find '{path_str}' -type f"
            
            # Exclude directories
            for exclude_dir in self.exclude_dirs:
                find_cmd += f" -not -path '*/{exclude_dir}/*'"
            
            # Escape pattern for shell
            escaped_pattern = pattern.replace("'", "'\"'\"'")
            
            # Combine find and grep
            grep_cmd = f"{find_cmd} -exec grep {' '.join(grep_flags)} '{escaped_pattern}' {{}} \\; 2>/dev/null | head -{max_results}"
            
            # Execute search
            search_result = await self._exec_command(grep_cmd)
            
            if not search_result["success"]:
                return ToolResult(
                    success=True,
                    result={
                        "matches": [],
                        "total_matches": 0,
                        "files_searched": 0,
                        "pattern": pattern,
                        "search_path": str(path)
                    }
                )
            
            # Parse grep output
            matches = []
            current_file = None
            files_searched = set()
            
            for line in search_result["stdout"].splitlines():
                if not line.strip():
                    continue
                
                # Parse grep output format: filename:line_number:content
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path, line_num_str, content = parts
                    try:
                        line_num = int(line_num_str)
                    except ValueError:
                        continue
                    
                    files_searched.add(file_path)
                    
                    matches.append({
                        "file": file_path,
                        "line_number": line_num,
                        "column": 1,  # grep doesn't provide column info easily
                        "match_text": pattern,  # Simplified
                        "line_content": content,
                        "context": content  # Simplified context
                    })
            
            return ToolResult(
                success=True,
                result={
                    "matches": matches[:max_results],
                    "total_matches": len(matches),
                    "files_searched": len(files_searched),
                    "pattern": pattern,
                    "search_path": str(path)
                },
                metrics={
                    "matches_found": len(matches),
                    "files_searched": len(files_searched)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"K8s text search failed: {e}")

    async def _search_file_names_k8s(self, pattern: str, path: Path, case_sensitive: bool,
                                    is_regex: bool, max_results: int) -> ToolResult:
        """Search for files matching name pattern using K8s."""
        try:
            path_str = str(path)
            
            # Check if path exists
            exists_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if not exists_result["success"] or exists_result["stdout"].strip() != "exists":
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            # Build find command
            find_flags = []
            if case_sensitive:
                find_flags.append("-name")
            else:
                find_flags.append("-iname")
            
            # Escape pattern for shell
            if is_regex:
                # For regex, use grep on find output
                escaped_pattern = pattern.replace("'", "'\"'\"'")
                if case_sensitive:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep '{escaped_pattern}' | head -{max_results}"
                else:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep -i '{escaped_pattern}' | head -{max_results}"
            else:
                # For literal strings, use find's name matching
                escaped_pattern = pattern.replace("'", "'\"'\"'")
                find_cmd = f"find '{path_str}' {' '.join(find_flags)} '*{escaped_pattern}*' | head -{max_results}"
            
            # Execute search
            search_result = await self._exec_command(find_cmd)
            
            if not search_result["success"]:
                return ToolResult(
                    success=True,
                    result={
                        "matches": [],
                        "total_matches": 0,
                        "pattern": pattern,
                        "search_path": str(path)
                    }
                )
            
            # Parse find output
            matches = []
            for line in search_result["stdout"].splitlines():
                if not line.strip():
                    continue
                
                item_path = Path(line.strip())
                
                # Get file info
                stat_result = await self._exec_command(f"stat -c '%F %s' '{line}'")
                if stat_result["success"]:
                    stat_parts = stat_result["stdout"].strip().split(' ', 1)
                    file_type = "directory" if "directory" in stat_parts[0] else "file"
                    size = int(stat_parts[1]) if len(stat_parts) > 1 and file_type == "file" else None
                else:
                    file_type = "file"
                    size = None
                
                matches.append({
                    "path": line.strip(),
                    "name": item_path.name,
                    "type": file_type,
                    "size": size
                })
            
            return ToolResult(
                success=True,
                result={
                    "matches": matches,
                    "total_matches": len(matches),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"K8s file name search failed: {e}")

    async def _search_directory_structure_k8s(self, pattern: str, path: Path, case_sensitive: bool,
                                             is_regex: bool, max_results: int) -> ToolResult:
        """Search directory structure using K8s."""
        try:
            path_str = str(path)
            
            # Check if path exists
            exists_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if not exists_result["success"] or exists_result["stdout"].strip() != "exists":
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            # Build find command for directory structure
            escaped_pattern = pattern.replace("'", "'\"'\"'")
            
            if is_regex:
                if case_sensitive:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep '{escaped_pattern}' | head -{max_results}"
                else:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep -i '{escaped_pattern}' | head -{max_results}"
            else:
                # Search in full path
                if case_sensitive:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep '{escaped_pattern}' | head -{max_results}"
                else:
                    find_cmd = f"find '{path_str}' -type f -o -type d | grep -i '{escaped_pattern}' | head -{max_results}"
            
            # Execute search
            search_result = await self._exec_command(find_cmd)
            
            if not search_result["success"]:
                return ToolResult(
                    success=True,
                    result={
                        "matches": [],
                        "total_matches": 0,
                        "pattern": pattern,
                        "search_path": str(path)
                    }
                )
            
            # Parse find output
            matches = []
            base_path = Path(path_str)
            
            for line in search_result["stdout"].splitlines():
                if not line.strip():
                    continue
                
                item_path = Path(line.strip())
                
                try:
                    relative_path = item_path.relative_to(base_path)
                    depth = len(relative_path.parts)
                except ValueError:
                    relative_path = item_path
                    depth = 0
                
                # Get file type
                type_result = await self._exec_command(f"test -d '{line}' && echo 'directory' || echo 'file'")
                file_type = type_result["stdout"].strip() if type_result["success"] else "file"
                
                matches.append({
                    "path": line.strip(),
                    "relative_path": str(relative_path),
                    "name": item_path.name,
                    "type": file_type,
                    "depth": depth
                })
            
            return ToolResult(
                success=True,
                result={
                    "matches": matches,
                    "total_matches": len(matches),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"K8s directory search failed: {e}")
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        info = {
            "execution_mode": self.execution_mode,
            "max_results": self.max_results,
            "max_file_size": self.max_file_size,
            "search_extensions": list(self.search_extensions),
            "exclude_dirs": list(self.exclude_dirs)
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info