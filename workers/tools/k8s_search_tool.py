#!/usr/bin/env python3
"""
K8s Search tool for finding text in files and directories within Kubernetes pods.
Based on search_tool.py but adapted to search within K8s pods using kodo.
"""

import asyncio
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from kodo import KubernetesManager
except ImportError:
    KubernetesManager = None

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class K8sSearchTool(AgenticBaseTool):
    """Tool for searching text in files and directories within Kubernetes pods."""
    
    def __init__(self, config: Dict = None):
        """Initialize K8s search tool."""
        if KubernetesManager is None:
            raise ImportError("kodo library is required for K8s tools. Please install it from https://github.com/baidubce/kodo.git")
        
        # Set K8s configuration first before calling super().__init__
        config = config or {}
        self.pod_name = config.get("pod_name", "swebench-xarray-pod")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        # Search settings
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
        
        # Initialize K8s manager
        self.k8s_manager = None
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for K8s search tool."""
        return create_openai_tool_schema(
            name="k8s_search",
            description=f"Search for text patterns in files and directories within K8s pod '{self.pod_name}'. Supports regex patterns and various search modes.",
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
                    "description": "File or directory path to search in within the pod"
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
        """Execute search operation in K8s pod."""
        try:
            command = parameters["command"]
            pattern = parameters["pattern"]
            path_str = parameters["path"]
            
            # Validate path exists in pod
            check_result = await self._exec_command(f"test -e '{path_str}' && echo 'exists' || echo 'not_exists'")
            if check_result["stdout"].strip() == "not_exists":
                return ToolResult(success=False, error=f"Path does not exist in pod: {path_str}")
            
            # Parse search options
            case_sensitive = parameters.get("case_sensitive", False)
            is_regex = parameters.get("regex", False)
            whole_words = parameters.get("whole_words", False)
            max_results = parameters.get("max_results", self.max_results)
            file_extensions = parameters.get("file_extensions", list(self.search_extensions))
            context_lines = parameters.get("context_lines", 2)
            
            if command == "search_text":
                return await self._search_text_in_files(
                    pattern, path_str, case_sensitive, is_regex, whole_words,
                    max_results, file_extensions, context_lines
                )
            elif command == "search_files":
                return await self._search_file_names(
                    pattern, path_str, case_sensitive, is_regex, max_results
                )
            elif command == "search_dir":
                return await self._search_directory_structure(
                    pattern, path_str, case_sensitive, is_regex, max_results
                )
            else:
                return ToolResult(success=False, error=f"Unknown search command: {command}")
                
        except Exception as e:
            logger.error(f"K8s search execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_text_in_files(self, pattern: str, path_str: str, case_sensitive: bool,
                                   is_regex: bool, whole_words: bool, max_results: int,
                                   file_extensions: List[str], context_lines: int) -> ToolResult:
        """Search for text pattern in files within K8s pod."""
        try:
            # Build grep command
            grep_flags = []
            
            if not case_sensitive:
                grep_flags.append("-i")
            
            if is_regex:
                grep_flags.append("-E")
            else:
                # Escape pattern for fixed string search unless it's regex
                if whole_words:
                    grep_flags.extend(["-w", "-F"])
                else:
                    grep_flags.append("-F")
            
            # Add context lines
            if context_lines > 0:
                grep_flags.append(f"-C {context_lines}")
            
            # Add line numbers and recursive search
            grep_flags.extend(["-n", "-r"])
            
            # Build file extension filter
            include_args = []
            if file_extensions:
                for ext in file_extensions:
                    include_args.extend(["--include", f"*{ext}"])
            
            # Build exclude directory filters
            exclude_args = []
            for exclude_dir in self.exclude_dirs:
                exclude_args.extend(["--exclude-dir", exclude_dir])
            
            # Construct full grep command
            grep_cmd_parts = ["grep"] + grep_flags + include_args + exclude_args + [f"'{pattern}'", f"'{path_str}'"]
            grep_command = " ".join(grep_cmd_parts)
            
            # Execute grep command
            result = await self._exec_command(grep_command)
            
            # Parse grep output
            results = []
            files_searched = 0
            
            if result["return_code"] == 0 and result["stdout"]:
                # Parse grep output
                lines = result["stdout"].strip().split('\n')
                current_file = None
                files_found = set()
                
                for line in lines:
                    if ':' in line:
                        # Extract file, line number, and content
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            try:
                                line_number = int(parts[1])
                                match_content = parts[2]
                                
                                files_found.add(file_path)
                                
                                results.append({
                                    "file": file_path,
                                    "line_number": line_number,
                                    "match_text": match_content.strip(),
                                    "context": line,  # Full grep output line with context
                                    "pod_location": f"{self.namespace}/{self.pod_name}:{file_path}"
                                })
                                
                                if len(results) >= max_results:
                                    break
                            except ValueError:
                                continue  # Skip lines that don't have valid line numbers
                
                files_searched = len(files_found)
            
            elif result["return_code"] == 1:
                # No matches found (normal grep behavior)
                pass
            else:
                # Error occurred
                return ToolResult(success=False, error=f"Search failed: {result['stderr']}")
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "files_searched": files_searched,
                    "pattern": pattern,
                    "search_path": path_str,
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                },
                metrics={
                    "matches_found": len(results),
                    "files_searched": files_searched
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Text search failed: {e}")
    
    async def _search_file_names(self, pattern: str, path_str: str, case_sensitive: bool,
                                is_regex: bool, max_results: int) -> ToolResult:
        """Search for files matching name pattern in K8s pod."""
        try:
            # Build find command
            find_flags = []
            
            if is_regex:
                if case_sensitive:
                    find_flags.extend(["-regex", f"'{pattern}'"])
                else:
                    find_flags.extend(["-iregex", f"'{pattern}'"])
            else:
                if case_sensitive:
                    find_flags.extend(["-name", f"'*{pattern}*'"])
                else:
                    find_flags.extend(["-iname", f"'*{pattern}*'"])
            
            # Add type filter and exclude directories
            exclude_args = []
            for exclude_dir in self.exclude_dirs:
                exclude_args.extend(["-path", f"'*/{exclude_dir}/*'", "-prune", "-o"])
            
            # Build find command
            find_cmd_parts = ["find", f"'{path_str}'"] + exclude_args + find_flags + ["-type", "f", "-print"]
            find_command = " ".join(find_cmd_parts)
            
            # Execute find command
            result = await self._exec_command(find_command)
            
            if result["return_code"] != 0:
                return ToolResult(success=False, error=f"File search failed: {result['stderr']}")
            
            results = []
            if result["stdout"]:
                file_paths = result["stdout"].strip().split('\n')
                
                for file_path in file_paths[:max_results]:
                    if file_path:  # Skip empty lines
                        # Get file info
                        stat_result = await self._exec_command(f"stat -c '%s %Y' '{file_path}' 2>/dev/null || echo '0 0'")
                        try:
                            size_str, mtime_str = stat_result["stdout"].strip().split()
                            file_size = int(size_str)
                        except (ValueError, IndexError):
                            file_size = 0
                        
                        results.append({
                            "path": file_path,
                            "name": Path(file_path).name,
                            "type": "file",
                            "size": file_size,
                            "pod_location": f"{self.namespace}/{self.pod_name}:{file_path}"
                        })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": path_str,
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"File name search failed: {e}")
    
    async def _search_directory_structure(self, pattern: str, path_str: str, case_sensitive: bool,
                                         is_regex: bool, max_results: int) -> ToolResult:
        """Search directory structure and list matching paths in K8s pod."""
        try:
            # Build find command for directory structure search
            find_flags = []
            
            if is_regex:
                if case_sensitive:
                    find_flags.extend(["-regex", f"'.*{pattern}.*'"])
                else:
                    find_flags.extend(["-iregex", f"'.*{pattern}.*'"])
            else:
                if case_sensitive:
                    find_flags.extend(["-path", f"'*{pattern}*'"])
                else:
                    find_flags.extend(["-ipath", f"'*{pattern}*'"])
            
            # Exclude certain directories
            exclude_args = []
            for exclude_dir in self.exclude_dirs:
                exclude_args.extend(["-path", f"'*/{exclude_dir}/*'", "-prune", "-o"])
            
            # Build find command
            find_cmd_parts = ["find", f"'{path_str}'"] + exclude_args + find_flags + ["-print"]
            find_command = " ".join(find_cmd_parts)
            
            # Execute find command
            result = await self._exec_command(find_command)
            
            if result["return_code"] != 0:
                return ToolResult(success=False, error=f"Directory search failed: {result['stderr']}")
            
            results = []
            if result["stdout"]:
                paths = result["stdout"].strip().split('\n')
                
                for full_path in paths[:max_results]:
                    if full_path and full_path != path_str:  # Skip empty lines and root path
                        # Check if it's directory or file
                        type_result = await self._exec_command(f"test -d '{full_path}' && echo 'directory' || echo 'file'")
                        path_type = type_result["stdout"].strip()
                        
                        # Calculate relative path and depth
                        try:
                            relative_path = Path(full_path).relative_to(Path(path_str))
                            depth = len(relative_path.parts)
                        except ValueError:
                            relative_path = Path(full_path)
                            depth = len(relative_path.parts)
                        
                        results.append({
                            "path": full_path,
                            "relative_path": str(relative_path),
                            "name": Path(full_path).name,
                            "type": path_type,
                            "depth": depth,
                            "pod_location": f"{self.namespace}/{self.pod_name}:{full_path}"
                        })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": path_str,
                    "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Directory search failed: {e}")
    
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
    
    def get_pod_info(self) -> Dict[str, str]:
        """Get information about the target pod."""
        return {
            "pod_name": self.pod_name,
            "namespace": self.namespace,
            "kubeconfig_path": self.kubeconfig_path or "default"
        }
    
    async def get_search_statistics(self, path_str: str) -> Dict[str, Any]:
        """Get statistics about the search target directory."""
        try:
            # Count files by extension
            count_result = await self._exec_command(f"find '{path_str}' -type f | wc -l")
            total_files = int(count_result["stdout"].strip()) if count_result["return_code"] == 0 else 0
            
            # Estimate directory size
            size_result = await self._exec_command(f"du -sh '{path_str}' 2>/dev/null | cut -f1")
            total_size = size_result["stdout"].strip() if size_result["return_code"] == 0 else "unknown"
            
            return {
                "total_files": total_files,
                "total_size": total_size,
                "search_path": path_str,
                "pod_location": f"{self.namespace}/{self.pod_name}:{path_str}"
            }
        except Exception as e:
            logger.warning(f"Failed to get search statistics: {e}")
            return {
                "total_files": 0,
                "total_size": "unknown",
                "search_path": path_str,
                "error": str(e)
            }