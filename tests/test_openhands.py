#!/usr/bin/env python3
"""
Test OpenHands tools for K8S execution.
This test demonstrates the OpenHands tools with K8s support.
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
from workers.tools.openhands_tools import (
    OpenHandsBashTool,
    OpenHandsStrReplaceEditorTool, 
    OpenHandsFinishTool
)
import logging
import re
from typing import Dict, Any, List, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")

# K8s configuration
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
K8S_POD_NAME = os.getenv("K8S_POD_NAME", "test-pod")
KUBECONFIG_PATH = os.getenv("KUBECONFIG_PATH", None)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "local")  # or "k8s"

print(f"API_KEY: {API_KEY}")
print(f"BASE_URL: {BASE_URL}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"EXECUTION_MODE: {EXECUTION_MODE}")
print(f"K8S_NAMESPACE: {K8S_NAMESPACE}")
print(f"K8S_POD_NAME: {K8S_POD_NAME}")


def parse_xml_action_openhands(content: str) -> Dict[str, Any]:
    """Parse XML-style action format for OpenHands tools."""
    # Look for action tags like <action type="execute">, <action type="str_replace_editor">, etc.
    action_pattern = r'<action\s+type="([^"]+)"(?:\s+[^>]*)?>([^<]*)</action>'
    
    match = re.search(action_pattern, content, re.DOTALL)
    if not match:
        # Try alternative patterns or fallback
        return {"action": "unknown", "parameters": {"content": content}}
    
    action_type = match.group(1)
    action_content = match.group(2).strip()
    
    # Parse parameters based on action type
    if action_type == "bash" or action_type == "execute":
        return {
            "action": "openhands_bash",
            "parameters": {"command": action_content}
        }
    elif action_type == "str_replace_editor":
        # Parse str_replace_editor parameters
        return parse_str_replace_editor_params(action_content)
    elif action_type == "finish":
        return {
            "action": "openhands_finish", 
            "parameters": {"message": action_content}
        }
    else:
        return {"action": action_type, "parameters": {"content": action_content}}


def parse_str_replace_editor_params(content: str) -> Dict[str, Any]:
    """Parse str_replace_editor action content into parameters."""
    lines = content.strip().split('\n')
    params = {}
    
    for line in lines:
        if line.startswith('COMMAND:'):
            params['command'] = line.replace('COMMAND:', '').strip()
        elif line.startswith('PATH:'):
            params['path'] = line.replace('PATH:', '').strip()
        elif line.startswith('FILE_TEXT:'):
            # Multi-line file text follows
            file_text_lines = []
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].startswith(('OLD_STR:', 'NEW_STR:', 'INSERT_LINE:')):
                file_text_lines.append(lines[idx])
                idx += 1
            params['file_text'] = '\n'.join(file_text_lines)
        elif line.startswith('OLD_STR:'):
            # Multi-line old string follows
            old_str_lines = []
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].startswith(('NEW_STR:', 'INSERT_LINE:')):
                old_str_lines.append(lines[idx])
                idx += 1
            params['old_str'] = '\n'.join(old_str_lines)
        elif line.startswith('NEW_STR:'):
            # Multi-line new string follows
            new_str_lines = []
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].startswith('INSERT_LINE:'):
                new_str_lines.append(lines[idx])
                idx += 1
            params['new_str'] = '\n'.join(new_str_lines)
        elif line.startswith('INSERT_LINE:'):
            params['insert_line'] = int(line.replace('INSERT_LINE:', '').strip())
    
    return {
        "action": "openhands_str_replace_editor",
        "parameters": params
    }


def generate_openhands_system_prompt(tools: List[str]) -> str:
    """Generate OpenHands-style system prompt."""
    return f"""You are a helpful AI assistant that can help with coding, file operations, and other tasks.

You have access to the following tools:
{', '.join(tools)}

For each action, use the following XML format:

For bash commands:
<action type="bash">
command_here
</action>

For file operations:
<action type="str_replace_editor">
COMMAND: view|create|str_replace|insert|undo_edit
PATH: /path/to/file
FILE_TEXT: (for create command)
content here
OLD_STR: (for str_replace command)
old content here
NEW_STR: (for str_replace command)
new content here
INSERT_LINE: (for insert command - line number)
</action>

When finished:
<action type="finish">
Summary of what was accomplished
</action>

Always be precise with file paths and commands. Use absolute paths when possible.
"""


class OpenHandsDescriptionWrapper:
    """Wrapper to provide OpenHands-style descriptions for tools."""
    
    def __init__(self, tool, use_custom_description=True):
        self.tool = tool
        self.use_custom_description = use_custom_description
        
        # Set the custom description flag in tool config
        if hasattr(tool, 'config'):
            tool.config['use_custom_description'] = use_custom_description
    
    def get_openai_tool_schema(self):
        """Get the tool schema."""
        return self.tool.get_openai_tool_schema()
    
    def get_description(self):
        """Get custom description if enabled."""
        if self.use_custom_description and hasattr(self.tool, 'get_description'):
            return self.tool.get_description()
        return None
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs):
        """Execute the wrapped tool."""
        return await self.tool.execute_tool(instance_id, parameters, **kwargs)


async def test_openhands_tools_basic():
    """Test basic OpenHands tools functionality."""
    print("🔧 Testing OpenHands tools - Basic functionality")
    
    # Create tool configurations
    tool_configs = {
        "execution_mode": EXECUTION_MODE,
        "pod_name": K8S_POD_NAME,
        "namespace": K8S_NAMESPACE,
        "kubeconfig_path": KUBECONFIG_PATH,
        "working_dir": "/workspace",
        "use_custom_description": True
    }
    
    # Test bash tool
    print("\n📝 Testing OpenHands Bash Tool...")
    bash_tool = OpenHandsBashTool(tool_configs)
    
    result = await bash_tool.execute_tool(
        instance_id="test_001",
        parameters={"command": "echo 'Hello from OpenHands bash tool!'"}
    )
    
    print(f"Bash tool result: {result.success}")
    if result.result:
        print(f"Output: {result.result.get('output', 'No output')}")
    
    # Test editor tool  
    print("\n📝 Testing OpenHands Editor Tool...")
    editor_tool = OpenHandsStrReplaceEditorTool(tool_configs)
    
    # Test create file
    result = await editor_tool.execute_tool(
        instance_id="test_002",
        parameters={
            "command": "create",
            "path": "/workspace/test_openhands.py",
            "file_text": "# OpenHands test file\nprint('Hello from OpenHands!')\n"
        }
    )
    
    print(f"Editor create result: {result.success}")
    
    # Test view file
    result = await editor_tool.execute_tool(
        instance_id="test_003",
        parameters={
            "command": "view",
            "path": "/workspace/test_openhands.py"
        }
    )
    
    print(f"Editor view result: {result.success}")
    if result.result:
        print(f"File content preview: {result.result.get('content', 'No content')[:200]}...")
    
    # Test finish tool
    print("\n📝 Testing OpenHands Finish Tool...")
    finish_tool = OpenHandsFinishTool(tool_configs)
    
    result = await finish_tool.execute_tool(
        instance_id="test_004",
        parameters={"message": "OpenHands tools testing completed successfully! ✅"}
    )
    
    print(f"Finish tool result: {result.success}")
    if result.result:
        print(f"Finish message: {result.result.get('message', 'No message')}")


async def test_openhands_agent_integration():
    """Test OpenHands tools with GeneralAgent integration."""
    print("\n🤖 Testing OpenHands tools with GeneralAgent...")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME
    )
    
    # Tool configurations
    tool_configs = {
        "execution_mode": EXECUTION_MODE,
        "pod_name": K8S_POD_NAME,
        "namespace": K8S_NAMESPACE,
        "kubeconfig_path": KUBECONFIG_PATH,
        "working_dir": "/workspace",
        "use_custom_description": True
    }
    
    # Create OpenHands tools
    tools = {
        "openhands_bash": OpenHandsDescriptionWrapper(
            OpenHandsBashTool(tool_configs), 
            use_custom_description=True
        ),
        "openhands_str_replace_editor": OpenHandsDescriptionWrapper(
            OpenHandsStrReplaceEditorTool(tool_configs),
            use_custom_description=True
        ),
        "openhands_finish": OpenHandsDescriptionWrapper(
            OpenHandsFinishTool(tool_configs),
            use_custom_description=True
        )
    }
    
    # Create system prompt
    system_prompt = generate_openhands_system_prompt(list(tools.keys()))
    
    # Create agent
    agent = GeneralAgent(
        llm_client=llm_client,
        tools=tools,
        max_iterations=10,
        action_parser=parse_xml_action_openhands,
        system_prompt=system_prompt
    )
    
    # Test task: Create a simple Python script and run it
    task = """Create a Python script that prints the current date and time, then execute it.
    
Steps:
1. Create a file called /workspace/datetime_script.py
2. Add Python code to print the current date and time
3. Run the script
4. Show the output
"""
    
    print(f"🎯 Task: {task}")
    
    # Run agent
    try:
        result = await agent.arun(task)
        
        print(f"\n✅ Agent completed with result: {result}")
        
        # Get trajectory
        trajectory = agent.get_trajectory()
        
        print(f"\n📊 Trajectory Summary:")
        print(f"Total steps: {len(trajectory.steps)}")
        
        for i, step in enumerate(trajectory.steps):
            print(f"Step {i+1}: {step.step_type.value}")
            if step.step_type == StepType.ACTION:
                action_data = step.action_data
                print(f"  Action: {action_data.get('action', 'Unknown')}")
                if step.observation_data:
                    success = step.observation_data.get('success', False)
                    print(f"  Success: {success}")
        
        # Save trajectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"openhands_trajectory_{timestamp}.json"
        
        dump_trajectory(trajectory, output_file)
        print(f"\n💾 Trajectory saved to: {output_file}")
        
        # Save as messages format too
        messages_file = f"openhands_messages_{timestamp}.json"
        save_trajectory_as_messages(trajectory, messages_file)
        print(f"💾 Messages saved to: {messages_file}")
        
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Starting OpenHands Tools Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic tool functionality
        await test_openhands_tools_basic()
        
        print("\n" + "=" * 50)
        
        # Test 2: Agent integration
        await test_openhands_agent_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Handle emoji preservation in logs [[memory:6314335]]
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
