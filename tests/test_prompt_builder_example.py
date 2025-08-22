#!/usr/bin/env python3
"""
Example showing how to use PromptBuilder with R2E tools.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client, PromptBuilder, PromptLibrary
from workers.core.trajectory import TrajectoryStep, StepType
import re
from typing import Dict, Any, List, Optional, Union

# Use same tool descriptions as test_r2e_general_agent.py
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


def parse_xml_action_custom(output: str) -> Optional[Union[Dict[str, Any], List[TrajectoryStep]]]:
    """Custom XML parser for R2E tools."""
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
        # Create both thought and action steps
        steps = []
        
        steps.append(TrajectoryStep(
            step_type=StepType.THOUGHT,
            content=text_before_function,
            metadata={"raw_output": output, "xml_parsed": True}
        ))
        
        steps.append(TrajectoryStep(
            step_type=StepType.ACTION,
            content=function_match.group(0),
            metadata={"raw_output": output, "xml_parsed": True},
            tool_name=mapped_tool_name,
            tool_args=tool_args
        ))
        
        return steps
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


def custom_tool_formatter(tool_name: str, tool: Any) -> str:
    """Custom formatter that uses tool's get_description method."""
    return tool.get_description()


async def test_prompt_builder():
    """Test PromptBuilder with R2E tools."""
    print("\n" + "="*80)
    print("🚀 Testing PromptBuilder with R2E Tools")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create R2E tools
    base_tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    # Wrap tools with custom descriptions
    tools = {}
    for tool_name, tool in base_tools.items():
        if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
            tools[tool_name] = CustomDescriptionWrapper(tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name])
        else:
            tools[tool_name] = tool
    
    # Example 1: Using PromptBuilder with template
    print("\n📝 Example 1: Template-based Prompt")
    print("-"*40)
    
    builder1 = PromptBuilder(
        template="""You are a programming agent working on the {repo_name} repository.
Task: {task_description}
Working Directory: {working_directory}"""
    )
    
    prompt1 = (builder1
               .add_variables(
                   repo_name="xarray",
                   task_description="fix the issue with datetime merging",
                   working_directory="/testbed"
               )
               .add_tools(tools, formatter=custom_tool_formatter)
               .add_section("Instructions", """If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>""")
               .build())
    
    print("Generated prompt (first 500 chars):")
    print(prompt1[:500] + "...")
    
    # Example 2: Section-based prompt building
    print("\n\n📝 Example 2: Section-based Prompt")
    print("-"*40)
    
    builder2 = PromptBuilder()
    
    prompt2 = (builder2
               .add_context({
                   "Repository": "pandas",
                   "Issue Number": "#12345",
                   "Python Version": "3.9",
                   "Test Command": "pytest tests/test_merge.py"
               })
               .add_section("Role", "You are an expert Python developer fixing bugs in pandas.")
               .add_tools(tools, formatter=custom_tool_formatter)
               .add_examples([
                   {
                       "input": "Search for merge function",
                       "output": "<function=search>\n<parameter=search_term>def merge</parameter>\n</function>"
                   }
               ])
               .add_timestamp()
               .build())
    
    print("Generated prompt (first 500 chars):")
    print(prompt2[:500] + "...")
    
    # Example 3: Using PromptLibrary with custom tools
    print("\n\n📝 Example 3: Using PromptLibrary")
    print("-"*40)
    
    # Create a custom SWE-bench prompt using the library
    issue_description = """The DataFrame.merge() function fails when merging on datetime columns with different timezones.
    
    Steps to reproduce:
    1. Create two DataFrames with datetime columns in different timezones
    2. Try to merge them using pd.merge()
    3. Observe the error
    
    Expected: Merge should handle timezone conversion automatically
    Actual: TypeError is raised"""
    
    swe_prompt = PromptLibrary.swe_bench_prompt(
        tools,
        issue=issue_description,
        repo="pandas",
        working_dir="/testbed/pandas",
        test_command="pytest tests/test_merge.py::test_timezone_merge"
    )
    
    # Since our tools use custom descriptions, we need to rebuild with custom formatter
    builder3 = PromptBuilder(template=swe_prompt.split("\n\nAvailable Tools:")[0])
    final_prompt = (builder3
                   .add_tools(tools, formatter=custom_tool_formatter)
                   .add_section("Testing", "Run tests using: pytest tests/test_merge.py::test_timezone_merge")
                   .build())
    
    print("Generated SWE-bench style prompt (first 600 chars):")
    print(final_prompt[:600] + "...")
    
    # Example 4: Dynamic prompt in agent
    print("\n\n📝 Example 4: Using with GeneralAgent")
    print("-"*40)
    
    # Create agent
    agent = GeneralAgent(
        max_rounds=5,
        debug=True,
        termination_tool_names=["r2e_submit"],
        action_parser=parse_xml_action_custom
    )
    agent.set_tools(tools)
    
    # Use PromptBuilder to create dynamic prompt
    task_specific_prompt = (PromptBuilder()
                           .add_section("Current Task", "Explore the repository structure")
                           .add_variable("timestamp", "2024-01-01 10:00:00")
                           .add_tools(tools, formatter=custom_tool_formatter)
                           .add_section("XML Format", """<function=example_function_name>
<parameter=param1>value1</parameter>
</function>""")
                           .build())
    
    agent.system_prompt = task_specific_prompt
    
    print("Agent configured with dynamic prompt")
    print(f"Prompt length: {len(task_specific_prompt)} characters")
    
    print("\n" + "="*80)
    print("✅ PromptBuilder test completed!")
    print("\nKey Features Demonstrated:")
    print("1. Template-based prompts with variable substitution")
    print("2. Section-based prompt building")
    print("3. Custom tool formatters for R2E-style descriptions")
    print("4. Integration with PromptLibrary")
    print("5. Dynamic prompt generation for agents")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_prompt_builder())