#!/usr/bin/env python3
"""
R2E (Reasoning to Execution) tool configurations and utilities.
This module contains custom tool descriptions, parsers, and prompt generators
for R2E-style agent interactions.
"""

import re
from typing import Any, Dict, List, Optional, Union


# Custom tool descriptions for R2E style
CUSTOM_TOOL_DESCRIPTIONS = {
    "r2e_file_editor": """–– BEGIN FUNCTION #1: file_editor ––
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

–– END FUNCTION #1 ––""",
    
    "r2e_bash_executor": """–– BEGIN FUNCTION #2: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
  •    If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
  •    If the bash command returns exit code -1, it means the process is still running. The assistant may:
  •    Call this function again with command as an empty string ("") to retrieve additional logs.
  •    Send more input to STDIN of the running process by calling this function again with command set to the text input.
  •    Send command="ctrl+c" to interrupt the currently running process.
  •    If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
  1.    cmd (string, required)
The bash command (and optional arguments) to execute.
  •    Can be empty ("") to retrieve more logs if the process is still running.
  •    Can be "ctrl+c" to interrupt the running process.

–– END FUNCTION #2 ––""",
    
    "r2e_search": """–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
  •    If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
  •    If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
  •    If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
  •    If no matches are found, it will inform you as well.

Parameters:
  1.    search_term (string, required)
The term or string to search for in files.
  2.    path (string, optional)
The file or directory to search in. Defaults to . if not specified.

–– END FUNCTION #3 ––""",
    
    "r2e_submit": """–– BEGIN FUNCTION #4: finish ––
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  •    The submit command finalizes your output.

Parameters:
  1.    command (string, required)
Currently allowed value: [submit]
  2.    result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

–– END FUNCTION #4 ––"""
}


def parse_xml_action_custom(output: str) -> Optional[Union[Dict[str, Any], List]]:
    """
    Custom XML parser for R2E tools.
    
    Args:
        output: The LLM output containing potential XML function calls
        
    Returns:
        Dict with tool_name and tool_args, or None if no valid function call found
    """
    output = output.strip()
    
    # Check if output contains XML function call
    function_match = re.search(
        r'<function=([^>]+)>(.*?)</function>',
        output,
        re.DOTALL
    )
    
    if not function_match:
        return None
    
    tool_name = function_match.group(1).strip()
    params_content = function_match.group(2).strip()
    
    # Parse parameters
    tool_args = {}
    param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
    param_matches = re.findall(param_pattern, params_content, re.DOTALL)
    
    for param_name, param_value in param_matches:
        tool_args[param_name.strip()] = param_value.strip()
    
    # Map function names to tool names
    tool_name_mapping = {
        "file_editor": "r2e_file_editor",
        "execute_bash": "r2e_bash_executor", 
        "search": "r2e_search",
        "finish": "r2e_submit"
    }
    
    mapped_tool_name = tool_name_mapping.get(tool_name, tool_name)
    
    # Check if there's text before the function call (thought)
    text_before_function = output[:function_match.start()].strip()
    
    if text_before_function:
        # When we have both thought and action, return a single ACTION step
        return {
            "tool_name": mapped_tool_name,
            "tool_args": tool_args,
            "thought_content": text_before_function,
            "has_thought": True
        }
    else:
        return {
            "tool_name": mapped_tool_name,
            "tool_args": tool_args
        }


class CustomDescriptionWrapper:
    """Wrapper to add custom description to existing tools."""
    
    def __init__(self, tool, custom_description):
        self.tool = tool
        self.custom_description = custom_description
        # Copy all attributes from the original tool
        for attr in dir(tool):
            if not attr.startswith('_') and attr != 'get_description':
                setattr(self, attr, getattr(tool, attr))
    
    def get_description(self):
        return self.custom_description
    
    def __getattr__(self, name):
        # Delegate all other attribute access to the wrapped tool
        return getattr(self.tool, name)


def generate_custom_system_prompt(tools, **kwargs):
    """Generate custom system prompt with dynamic variables.
    
    Args:
        tools: Dictionary of tool instances
        **kwargs: Additional variables to inject into the prompt
            - task_description: Description of the task
            - working_directory: Current working directory
            - additional_instructions: Any additional instructions
    """
    task_description = kwargs.get('task_description', 'solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue')
    working_directory = kwargs.get('working_directory', '/testbed')
    additional_instructions = kwargs.get('additional_instructions', '')
    
    prompt = f"""You are a programming agent who is provided a github issue and repository bash environment and is tasked to {task_description}.

We have access to the following functions:

{tools['r2e_file_editor'].get_description()}

{tools['r2e_bash_executor'].get_description()}

{tools['r2e_search'].get_description()}

{tools['r2e_submit'].get_description()}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
{additional_instructions}"""
    
    return prompt


__all__ = [
    'CUSTOM_TOOL_DESCRIPTIONS',
    'parse_xml_action_custom',
    'CustomDescriptionWrapper',
    'generate_custom_system_prompt'
]