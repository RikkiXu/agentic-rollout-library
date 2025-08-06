#!/usr/bin/env python3
"""
Test GeneralAgent with R2E tools for K8S execution.
This configures a GeneralAgent instance with R2E tools that execute in K8S.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent, dump_trajectory, save_trajectory_as_messages
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.core.trajectory import TrajectoryStep, StepType
import logging
import re
from typing import Dict, Any, List, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
#MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
#MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")
#MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")
#MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")
print(API_KEY)
print(BASE_URL)
print(MODEL_NAME)

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
  1.    command (string, required)
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
  •    This function takes no parameters.

Parameters:
  None - This function requires no parameters.

–– END FUNCTION #4 ––"""
}


def parse_xml_action_custom(output: str) -> Optional[Union[Dict[str, Any], List[TrajectoryStep]]]:
    """
    Custom XML parser for R2E tools.
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


# Create wrapper classes that override get_description
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

Current working directory: {working_directory}

We have access to the following functions:

{tools['r2e_file_editor'].get_description()}

{tools['r2e_bash_executor'].get_description()}

{tools['r2e_search'].get_description()}

{tools['r2e_submit'].get_description()}

When calling a function, your response should follow this format:

First, explain your reasoning and what you plan to do (as natural text).

Then, make the function call in this exact format:
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
- ALWAYS start with your reasoning/thought process before the function call
- Function calls MUST follow the specified XML format
- Required parameters MUST be specified
- Only call one function at a time
- Your complete response = reasoning (natural text) + function call (XML format)
{additional_instructions}"""
    
    return prompt


async def test_r2e_general_agent_k8s(output_dir: str = "./trajectories"):
    """Test GeneralAgent with R2E tools in K8S execution mode.
    
    Args:
        output_dir: Directory to save trajectory files
    """
    print("\n" + "="*80)
    print("🚀 Testing GeneralAgent with R2E Tools (K8S Execution)")
    print("="*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Trajectory output directory: {output_dir}")
    
    # K8S configuration for swebench-xarray-pod
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": os.getenv("K8S_NAMESPACE", "default"),
        "kubeconfig_path": os.getenv("KUBECONFIG", None)  # Will use default kubeconfig if not set
    }
    
    # Create R2E tools with K8S configuration
    print("\n📦 Creating R2E tools for K8S execution...")
    
    # Create base tools
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
    
    print(f"✅ Created {len(tools)} R2E tools")
    print(f"   Pod: {k8s_config['pod_name']}")
    print(f"   Namespace: {k8s_config['namespace']}")
    print(f"   Kubeconfig: {k8s_config.get('kubeconfig_path') or 'default'}")
    
    # Display tool schemas
    print("\n📋 Tool Schemas:")
    for name, tool in tools.items():
        schema = tool.get_openai_tool_schema()
        if hasattr(schema, 'model_dump'):
            schema_dict = schema.model_dump()
        else:
            schema_dict = schema.dict()
        func = schema_dict.get('function', {})
        print(f"  - {name}: {func.get('description', '')[:60]}...")
    
    # Create GeneralAgent with R2E tools
    print("\n🤖 Creating GeneralAgent with R2E tools...")
    
    # Generate custom system prompt with variables
    custom_system_prompt = generate_custom_system_prompt(
        tools,
        task_description="analyze and fix issues in the repository",
        working_directory="/testbed",
        additional_instructions="\n- Be concise in your responses\n- Focus on the specific issue at hand"
    )
    
    # Create agent instance with termination tool and custom XML parser
    agent = GeneralAgent(
        max_rounds=15,
        debug=True,  # Enable agent debug output
        termination_tool_names=["r2e_submit"],  # Mark r2e_submit as termination tool
        action_parser=parse_xml_action_custom,  # Use custom XML action parser
        system_prompt=custom_system_prompt  # Pass custom prompt in constructor
    )
    agent.set_tools(tools)
    
    # Print the actual system prompt being used
    print("\n" + "="*80)
    print("📋 Agent System Prompt:")
    print("="*80)
    agent.create_system_prompt()
    print("="*80)

    #exit(0)
    print("✅ GeneralAgent created successfully")
    
    # Test task 1: Explore the K8S pod environment
    print("\n" + "="*80)
    print("📝 Test Task 1: Explore K8S Pod Environment")
    print("="*80)
    
    # You can also regenerate the prompt with different variables for different tasks
    # For example:
    # agent.system_prompt = generate_custom_system_prompt(
    #     tools,
    #     task_description="explore and understand the codebase structure",
    #     working_directory="/workspace"
    # )
    
    task1 = """
    Explore the current working directory:
    1. Show the current directory path
    2. List the contents of the current directory
    3. Check if there's a README file and show its first 20 lines if it exists
    4. Submit when done
    """
    
    print(f"\nTask: {task1.strip()}")
    print("\nExecuting agent...")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True
    )
    
    try:
        result = await agent.run_trajectory(
            prompt=task1,
            llm_generate_func=llm_client.generate,
            request_id="test_r2e_k8s_1"
        )
        
        print("\n✅ Task 1 completed")
        print(f"Total steps: {len(result.steps)}")
        print(f"Completed: {result.is_completed}")
        
        # Save trajectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_file = os.path.join(output_dir, f"task1_explore_{timestamp}.jsonl")
        dump_trajectory(result, trajectory_file, format="jsonl")
        print(f"📝 Saved trajectory to: {trajectory_file}")
        
    except Exception as e:
        print(f"\n❌ Task 1 failed: {e}")
    
    # Test task 2: Create and modify a file
    print("\n" + "="*80)
    print("📝 Test Task 2: File Operations in K8S")
    print("="*80)
    
    task2 = """
    Create a test Python file in /tmp directory:
    1. Create a file named /tmp/r2e_test_agent.py with a simple hello world function
    2. Search for 'def' in the file to verify it was created
    3. Use str_replace to change 'Hello' to 'Hi' in the function
    4. View the modified file to confirm the change
    5. Submit when done
    """
    
    print(f"\nTask: {task2.strip()}")
    print("\nExecuting agent...")
    
    try:
        result = await agent.run_trajectory(
            prompt=task2,
            llm_generate_func=llm_client.generate,
            request_id="test_r2e_k8s_2"
        )
        
        print("\n✅ Task 2 completed")
        print(f"Total steps: {len(result.steps)}")
        print(f"Completed: {result.is_completed}")
        
        # Save trajectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_file = os.path.join(output_dir, f"task2_file_ops_{timestamp}.jsonl")
        dump_trajectory(result, trajectory_file, format="jsonl")
        print(f"📝 Saved trajectory to: {trajectory_file}")
        
        # Also save a more detailed version
        detailed_trajectory = os.path.join(output_dir, f"task2_file_ops_{timestamp}_detailed.json")
        dump_trajectory(result, detailed_trajectory, format="json")
        print(f"📝 Saved detailed trajectory to: {detailed_trajectory}")
        
    except Exception as e:
        print(f"\n❌ Task 2 failed: {e}")
    
    print("\n" + "="*80)
    print("🎉 R2E GeneralAgent K8S test completed!")
    print(f"📁 All trajectories saved to: {output_dir}")
    print("="*80)
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "summary.json")
    summary = {
        "test_run": datetime.now().isoformat(),
        "output_directory": output_dir,
        "tasks_executed": 2,
        "trajectories_saved": [
            f"task1_explore_{timestamp}.jsonl",
            f"task2_file_ops_{timestamp}.jsonl"
        ],
        "configuration": {
            "model": MODEL_NAME,
            "max_rounds": agent.max_rounds,
            "k8s_pod": k8s_config["pod_name"],
            "k8s_namespace": k8s_config["namespace"]
        }
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Summary saved to: {summary_file}")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test R2E GeneralAgent with trajectory saving")
    parser.add_argument("--output-dir", default="./trajectories", 
                       help="Directory to save trajectory files (default: ./trajectories)")
    args = parser.parse_args()
    
    print("🧪 R2E GeneralAgent Test Suite")
    print("Testing GeneralAgent with R2E tools in K8S mode only")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Skip local test for now - focus on K8S
    # await test_r2e_general_agent_local()
    
    # Test K8S execution with output directory
    await test_r2e_general_agent_k8s(output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print(f"📁 Check {args.output_dir} for saved trajectories")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
