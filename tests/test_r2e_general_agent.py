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
from workers.tools.r2e_configs import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    CustomDescriptionWrapper,
    generate_custom_system_prompt
)
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
