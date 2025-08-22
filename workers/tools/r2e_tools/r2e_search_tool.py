#!/usr/bin/env python3
"""
R2E Search tool converted to official tool implementation.
Supports both local search and remote K8s pod search.
Based on the original R2E search.py.
"""

import os
import fnmatch
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

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class R2ESearchTool(AgenticBaseTool):
    """R2E-style search tool for finding text in files."""
    
    def __init__(self, config: Dict = None):
        """Initialize R2E search tool."""
        # Set execution mode and K8s config FIRST
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        self.working_dir = config.get("working_dir", "/testbed")  # R2E default working directory
        
        super().__init__(config)
        
        # R2E-specific settings
        self.max_results = self.config.get("max_results", 100)
        self.max_files_threshold = self.config.get("max_files_threshold", 100)
        self.python_only = self.config.get("python_only", False)
        self.exclude_hidden = self.config.get("exclude_hidden", True)
        self.ignore_patterns = set(self.config.get("ignore_patterns", [
            ".git", "__pycache__", "*.pyc", ".pytest_cache"
        ]))
        
        self.k8s_manager = None
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode.")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
    
    def get_description(self) -> str:
        """Override to provide custom description for R2E search."""
        if self.config.get("use_custom_description", False):
            return """–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
  •    If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files
  •    If path points to a file, it runs a grep -n in that file to show line numbers
  •    If more than 100 files match, results are truncated

Parameters:
  1.    search_term (string, required)
The term or string to search for in files.
  2.    path (string, optional)
The file or directory to search in. Defaults to . if not specified.

–– END FUNCTION #3 ––"""
        
        # Default: return JSON schema
        return super().get_description()
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E search."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="r2e_search",
            description=f"Search for a term in files or directories{execution_context}. R2E-style recursive search tool.",
            parameters={
                "search_term": {
                    "type": "string",
                    "description": "The term or string to search for in files"
                },
                "path": {
                    "type": "string", 
                    "description": "The file or directory to search in. Defaults to current directory if not specified",
                    "default": "."
                }
            },
            required=["search_term"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute search."""
        try:
            search_term = parameters["search_term"]
            path = parameters.get("path", ".")
            
            # Safety check: prevent searching from root or system directories
            dangerous_paths = ["/", "/etc", "/usr", "/bin", "/sbin", "/var", "/sys", "/proc"]
            if path in dangerous_paths:
                # Redirect to working directory instead
                logger.warning(f"Search from {path} is not allowed, redirecting to working directory")
                path = self.working_dir
                return ToolResult(
                    success=False,
                    error=f"Searching from '{parameters.get('path', '.')}' is not allowed as it may timeout. Please search in a specific directory like {self.working_dir} instead."
                )
            
            # Choose execution mode
            if self.execution_mode == "k8s":
                return await self._search_k8s(search_term, path, self.python_only)
            else:
                return await self._search_local(search_term, path, self.python_only)
                
        except Exception as e:
            logger.error(f"R2E search failed: {e}")
            return ToolResult(
                success=False, 
                error=str(e),
                result={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
    
    async def _search_local(self, search_term: str, path_str: str, python_only: bool) -> ToolResult:
        """Search locally."""
        try:
            path = Path(path_str)
            
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"Path does not exist: {path_str}"
                )
            
            if path.is_file():
                return self._search_in_file_local(search_term, path)
            else:
                return self._search_in_directory_local(search_term, path, python_only)
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _search_in_file_local(self, search_term: str, filepath: Path) -> ToolResult:
        """Search in a single file locally."""
        try:
            matches = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if search_term in line:
                        matches.append(f"{line_num}: {line.strip()}")
            
            if not matches:
                output = f'No matches found for "{search_term}" in {filepath}'
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": 0}
                )
            
            # Format output
            output_lines = [f'Found {len(matches)} matches for "{search_term}" in {filepath}:']
            for match in matches[:20]:  # Limit display
                output_lines.append(f"line {match}")
            
            if len(matches) > 20:
                output_lines.append(f"... and {len(matches) - 20} more matches")
            
            output = "\n".join(output_lines)
            
            return ToolResult(
                success=True,
                result={
                    "output": output,
                    "matches": len(matches)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _search_in_directory_local(self, search_term: str, directory: Path, python_only: bool) -> ToolResult:
        """Search in directory locally."""
        try:
            matches = {}
            num_files_matched = 0
            
            # Walk directory
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories
                if self.exclude_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                # Skip ignored patterns
                dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in self.ignore_patterns)]
                
                for filename in files:
                    # Skip hidden files
                    if self.exclude_hidden and filename.startswith('.'):
                        continue
                    
                    # Skip ignored patterns
                    if any(fnmatch.fnmatch(filename, p) for p in self.ignore_patterns):
                        continue
                    
                    # Check if we should search this file
                    if python_only and not filename.endswith('.py'):
                        continue
                    
                    filepath = Path(root) / filename
                    
                    # Search in file
                    try:
                        file_matches = 0
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if search_term in line:
                                    file_matches += 1
                            if file_matches > 0:
                                matches[str(filepath)] = file_matches
                                num_files_matched += 1
                    except (UnicodeDecodeError, PermissionError):
                        # Skip files that can't be read
                        continue
            
            if not matches:
                output = f'No matches found for "{search_term}" in {directory}'
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": 0}
                )
            
            # Check threshold
            if num_files_matched > self.max_files_threshold:
                output = (
                    f'More than {num_files_matched} files matched for "{search_term}" in {directory}. '
                    "Please narrow your search."
                )
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": sum(matches.values())}
                )
            
            # Format output
            num_matches = sum(matches.values())
            output_lines = [f'Found {num_matches} matches for "{search_term}" in {directory}:']
            
            # Print matched files
            for filepath, count in matches.items():
                relative_path = os.path.relpath(filepath, start=os.getcwd())
                if not relative_path.startswith("./"):
                    relative_path = "./" + relative_path
                output_lines.append(f"{relative_path} ({count} matches)")
            
            output_lines.append(f'End of matches for "{search_term}" in {directory}')
            output = "\n".join(output_lines)
            
            return ToolResult(
                success=True,
                result={
                    "output": output,
                    "matches": num_matches,
                    "files_matched": num_files_matched
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
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
            
            # Prepend cd to working directory
            # Don't wrap in bash -c or add extra escaping - let kodo handle it
            full_command = f"cd {self.working_dir} && {command}"
            
            # Execute command in pod using kodo API
            output, exit_code = k8s_mgr.execute_command(self.pod_name, full_command)
            
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
                
            return {
                "success": exit_code_int == 0,
                "stdout": output,
                "stderr": "",  # kodo API doesn't separate stderr
                "return_code": exit_code_int
            }
            
        except Exception as e:
            logger.error(f"K8s command execution failed: {e}")
            # Check if it's an ApiException or connection issue
            error_str = str(e)
            if "ApiException" in error_str or "connection" in error_str.lower():
                logger.error(f"K8s API error - pod may not be ready or connection issue: {error_str}")
            return {
                "success": False,
                "stdout": "",
                "stderr": error_str,
                "return_code": -1
            }
    
    async def _search_k8s(self, search_term: str, path_str: str, python_only: bool) -> ToolResult:
        """Search in K8s pod with optimized command execution."""
        try:
            # First check if we can execute commands in the pod
            test_cmd = "echo 'TEST'"
            test_result = await self._exec_command(test_cmd)
            if not test_result["success"] and "ApiException" in test_result.get("stderr", ""):
                # Pod is not ready or K8s API issue
                return ToolResult(
                    success=False,
                    error=f"K8s pod execution failed. Pod '{self.pod_name}' may not be ready or there's a connection issue. Error: {test_result.get('stderr', '')}"
                )
            # Handle path normalization for K8s environment
            # We need to ensure paths work correctly in the pod's working directory
            logger.debug(f"Original path: {path_str}, working_dir: {self.working_dir}")
            
            if not path_str or path_str == "" or path_str == ".":
                # Empty or current directory
                path_str = "."
            elif path_str.startswith("./"):
                # Already properly formatted relative path
                pass  # Keep as is
            elif path_str.startswith(self.working_dir):
                # Full path starting with working_dir (e.g., /testbed/django/db)
                # This is OK, keep as absolute path
                pass  # Keep as is
            elif path_str.startswith("/"):
                # Absolute path NOT starting with working_dir (e.g., /django/db)
                # This is likely meant to be relative to working_dir
                # Convert to relative: /django/db -> ./django/db
                path_str = "." + path_str
            else:
                # Plain relative path without prefix (e.g., django/db)
                # Add ./ prefix for clarity
                path_str = "./" + path_str
            
            logger.debug(f"Normalized path: {path_str}")
            
            # Escape single quotes in search term and path
            escaped_term = search_term.replace("'", "'\"'\"'")
            escaped_path = path_str.replace("'", "'\"'\"'")
            
            # Combine all operations into a single command to minimize pod executions
            # This command will:
            # 1. Check if path exists
            # 2. Determine if it's a file or directory
            # 3. Execute the appropriate search command
            # 4. Return results in a parseable format
            
            # Simplified approach: check path existence first
            # Use [ ] instead of test for better compatibility
            check_cmd = f"[ -e '{escaped_path}' ] && echo 'EXISTS' || echo 'NOT_EXISTS'"
            check_result = await self._exec_command(check_cmd)
            
            # Check if command execution failed
            if not check_result["success"] and check_result.get("exit_code") == -1:
                return ToolResult(
                    success=False,
                    error=f"Command execution failed: {check_result.get('stderr', 'Unknown error')}"
                )
            
            if check_result.get("stdout", "").strip() == "NOT_EXISTS":
                # Show the actual path in the working directory context
                if path_str == ".":
                    actual_path = self.working_dir
                elif path_str.startswith("./"):
                    actual_path = f"{self.working_dir}/{path_str[2:]}"
                elif path_str.startswith(self.working_dir):
                    actual_path = path_str
                else:
                    actual_path = f"{self.working_dir}/{path_str}"
                return ToolResult(success=False, error=f"Path not found: {actual_path}")
            
            # Check if it's a file or directory
            type_cmd = f"[ -f '{escaped_path}' ] && echo 'FILE' || echo 'DIR'"
            type_result = await self._exec_command(type_cmd)
            
            # Check if command execution failed
            if not type_result["success"] and type_result.get("exit_code") == -1:
                return ToolResult(
                    success=False,
                    error=f"Command execution failed while checking path type: {type_result.get('stderr', 'Unknown error')}"
                )
            is_file = type_result.get("stdout", "").strip() == "FILE"
            
            if is_file:
                # Search in single file
                search_cmd = f"grep -n '{escaped_term}' '{escaped_path}' 2>/dev/null || echo 'NO_MATCHES'"
                combined_cmd = f"echo 'TYPE: file' && {search_cmd}"
            else:
                # Search in directory
                if python_only:
                    # For Python files only
                    search_cmd = (
                        f"find '{escaped_path}' -maxdepth 5 -type f -name '*.py' 2>/dev/null | "
                        f"head -100 | xargs -I{{}} grep -l '{escaped_term}' {{}} 2>/dev/null | head -100"
                    )
                else:
                    # For all files
                    search_cmd = (
                        f"find '{escaped_path}' -maxdepth 5 -type f -size -10M 2>/dev/null | "
                        f"head -200 | xargs -I{{}} grep -l '{escaped_term}' {{}} 2>/dev/null | head -100"
                    )
                combined_cmd = f"echo 'TYPE: directory' && ({search_cmd} || echo 'NO_MATCHES')"
            
            # Execute the combined command
            result = await self._exec_command(combined_cmd)
            
            if not result["success"]:
                error_output = result.get("stdout", "").strip()
                if error_output.startswith("ERROR:"):
                    error_msg = error_output.replace("ERROR:", "").strip()
                    # Show the actual path in the working directory context
                    if path_str == ".":
                        actual_path = self.working_dir
                    elif path_str.startswith("./"):
                        actual_path = f"{self.working_dir}/{path_str[2:]}"
                    elif path_str.startswith(self.working_dir):
                        actual_path = path_str
                    else:
                        actual_path = f"{self.working_dir}/{path_str}"
                    return ToolResult(success=False, error=f"{error_msg}: {actual_path}")
                else:
                    # Include more diagnostic information
                    stderr_msg = result.get('stderr', '').strip()
                    stdout_msg = result.get('stdout', '').strip()[:200]  # First 200 chars for debugging
                    return ToolResult(
                        success=False, 
                        error=f"Command execution failed. stderr: {stderr_msg}, stdout preview: {stdout_msg}"
                    )
            
            # Parse the output
            output_lines = result["stdout"].strip().split('\n')
            if not output_lines:
                return ToolResult(success=False, error="No output from search command")
            
            # First line tells us the type
            type_line = output_lines[0]
            if type_line == "TYPE: file":
                # File search results
                if len(output_lines) == 2 and output_lines[1] == "NO_MATCHES":
                    return ToolResult(
                        success=True,
                        result={
                            "output": f'No matches found for "{search_term}" in {path_str}',
                            "matches": 0
                        }
                    )
                else:
                    # Process grep -n output
                    matches = output_lines[1:]
                    output = f'Found {len(matches)} matches for "{search_term}" in {path_str}:\n'
                    for match in matches[:20]:  # Limit display
                        output += match + '\n'
                    if len(matches) > 20:
                        output += f'... and {len(matches) - 20} more matches'
                    return ToolResult(
                        success=True,
                        result={
                            "output": output.strip(),
                            "matches": len(matches)
                        }
                    )
            elif type_line == "TYPE: directory":
                # Directory search results
                if len(output_lines) == 2 and output_lines[1] == "NO_MATCHES":
                    return ToolResult(
                        success=True,
                        result={
                            "output": f'No matches found for "{search_term}" in {path_str}',
                            "matches": 0
                        }
                    )
                else:
                    # Process file list
                    matched_files = [f for f in output_lines[1:] if f and f != "NO_MATCHES"]
                    
                    if len(matched_files) >= 100:
                        output = f'Found more than 100 files matching "{search_term}" in {path_str}. Please narrow your search.'
                    else:
                        output = f'Found {len(matched_files)} files matching "{search_term}" in {path_str}:\n'
                        for filepath in matched_files:
                            output += filepath + '\n'
                    
                    return ToolResult(
                        success=True,
                        result={
                            "output": output.strip(),
                            "files_matched": len(matched_files)
                        }
                    )
            else:
                return ToolResult(success=False, error=f"Unexpected output format: {type_line}")
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        info = {
            "execution_mode": self.execution_mode,
            "max_results": self.max_results,
            "max_files_threshold": self.max_files_threshold,
            "python_only": self.python_only,
            "exclude_hidden": self.exclude_hidden,
            "ignore_patterns": list(self.ignore_patterns),
            "tool_style": "R2E"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info