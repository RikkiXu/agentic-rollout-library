#!/usr/bin/env python3
"""
Tool Integration Demo for Agentic Rollout Library

This demo shows how to integrate the new core tools with the agentic framework,
demonstrating both standalone usage and integration with agents.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workers import (
    # Core framework
    AgenticRollout, AgenticRolloutConfig, ReactAgent,
    # Tools
    CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool,
    # Tool management
    get_global_tool_registry, register_tool
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMFunction:
    """Mock LLM function that provides realistic ReAct responses."""
    
    def __init__(self, scenario="math"):
        self.scenario = scenario
        self.call_count = 0
        self.responses = self._get_responses_for_scenario(scenario)
    
    def _get_responses_for_scenario(self, scenario):
        if scenario == "math":
            return [
                "Thought: I need to calculate the factorial of 5.",
                "Action: calculator(expression='factorial(5)')",
                "Thought: I can also calculate it step by step to verify.",
                "Action: calculator(expression='5 * 4 * 3 * 2 * 1')",
                "Action: Final Answer: The factorial of 5 is 120."
            ]
        elif scenario == "file_ops":
            return [
                "Thought: I need to create a Python script file.",
                "Action: file_editor(command='create', path='/tmp/hello.py', file_text='print(\"Hello from Agentic Tools!\")')",
                "Thought: Now let me view the file to confirm it was created.",
                "Action: file_editor(command='view', path='/tmp/hello.py')",
                "Thought: Great! Now I'll run the script to test it.",
                "Action: bash_executor(command='python /tmp/hello.py')",
                "Action: Final Answer: Successfully created and executed the Python script."
            ]
        elif scenario == "search_and_edit":
            return [
                "Thought: I need to search for Python files in the current directory.",
                "Action: search(command='search_files', pattern='*.py', path='/tmp')",
                "Thought: Now let me search for 'import' statements in Python files.",
                "Action: search(command='search_text', pattern='import', path='/tmp', file_extensions=['.py'])",
                "Action: Final Answer: Completed file search operations."
            ]
        else:
            return [
                "Thought: Let me solve this step by step.",
                "Action: Final Answer: Task completed."
            ]
    
    async def __call__(self, messages, **kwargs):
        """Mock LLM generation."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "Action: Final Answer: Task completed."
        
        self.call_count += 1
        logger.info(f"Mock LLM response: {response}")
        return response


async def demo_individual_tools():
    """Demo each tool individually."""
    print("\n" + "="*60)
    print("INDIVIDUAL TOOLS DEMO")
    print("="*60)
    
    # Calculator Tool
    print("\n1. Calculator Tool Demo")
    calc = CalculatorTool({"debug": False})
    instance_id = await calc.create_instance()
    
    # Test various calculations
    calculations = [
        {"expression": "sqrt(16) + 2**3"},
        {"operation": "factorial", "n": 5},
        {"numbers": [1, 2, 3, 4, 5], "operation": "average"}
    ]
    
    for calc_params in calculations:
        result = await calc.execute_tool(instance_id, calc_params)
        if result.success:
            print(f"  {calc_params} = {result.result['result']}")
        else:
            print(f"  Error: {result.error}")
    
    await calc.release_instance(instance_id)
    
    # Bash Executor Tool
    print("\n2. Bash Executor Tool Demo")
    bash = BashExecutorTool({"timeout": 10})
    instance_id = await bash.create_instance()
    
    commands = [
        {"command": "echo 'Hello from bash!'"},
        {"command": "python -c 'import math; print(f\"Pi = {math.pi:.4f}\")'"},
        {"command": "ls /tmp | head -3"}
    ]
    
    for cmd_params in commands:
        result = await bash.execute_tool(instance_id, cmd_params)
        if result.success:
            print(f"  $ {cmd_params['command']}")
            print(f"    Output: {result.result['stdout'].strip()}")
        else:
            print(f"  Command failed: {result.error}")
    
    await bash.release_instance(instance_id)
    
    # File Editor Tool
    print("\n3. File Editor Tool Demo")
    editor = FileEditorTool({"max_file_size": 10240})
    instance_id = await editor.create_instance()
    
    # Create a test file
    test_file = "/tmp/agentic_demo.py"
    result = await editor.execute_tool(instance_id, {
        "command": "create",
        "path": test_file,
        "file_text": "# Agentic Tools Demo\nprint('Hello World!')\n"
    })
    
    if result.success:
        print(f"  Created file: {test_file}")
        
        # View the file
        result = await editor.execute_tool(instance_id, {
            "command": "view",
            "path": test_file
        })
        if result.success:
            print(f"  File contents:\n{result.result['content']}")
    
    await editor.release_instance(instance_id)
    
    # Search Tool
    print("\n4. Search Tool Demo")
    search = SearchTool({"max_results": 5})
    instance_id = await search.create_instance()
    
    # Search for Python files
    result = await search.execute_tool(instance_id, {
        "command": "search_files",
        "pattern": "*.py",
        "path": str(Path(__file__).parent),
        "max_results": 3
    })
    
    if result.success:
        print(f"  Found {result.result['total_matches']} Python files:")
        for match in result.result['matches']:
            print(f"    {match['name']}")
    
    await search.release_instance(instance_id)


async def demo_tool_registry():
    """Demo the global tool registry system."""
    print("\n" + "="*60)
    print("TOOL REGISTRY DEMO")
    print("="*60)
    
    # Register tools with the global registry
    register_tool(CalculatorTool, {"debug": False}, "calc")
    register_tool(BashExecutorTool, {"timeout": 15}, "bash")
    register_tool(FileEditorTool, {"max_file_size": 1024*1024}, "editor")
    register_tool(SearchTool, {"max_results": 10}, "search")
    
    registry = get_global_tool_registry()
    print(f"Registered tools: {registry.get_tool_names()}")
    
    # Create and use tools through registry
    calc_instance = await registry.create_tool_instance("calc")
    bash_instance = await registry.create_tool_instance("bash")
    
    # Execute through registry
    result = await registry.execute_tool("calc", calc_instance, {"expression": "10 + 5 * 2"})
    print(f"Calculator result: {result.result['result'] if result.success else result.error}")
    
    result = await registry.execute_tool("bash", bash_instance, {"command": "date"})
    print(f"Date command: {result.result['stdout'].strip() if result.success else result.error}")
    
    # Show registry info
    tool_info = registry.get_tool_info()
    print("\nRegistry status:")
    for info in tool_info:
        print(f"  {info['name']}: {info['active_instances']} active instances")
    
    # Clean up
    await registry.release_all_instances()
    print("All tool instances released")


async def demo_agent_with_tools():
    """Demo using tools with a ReAct agent."""
    print("\n" + "="*60)
    print("AGENT WITH TOOLS DEMO")
    print("="*60)
    
    # Register tools if not already done
    registry = get_global_tool_registry()
    if not registry.get_tool_names():
        register_tool(CalculatorTool, {"debug": False})
        register_tool(BashExecutorTool, {"timeout": 10})
        register_tool(FileEditorTool)
        register_tool(SearchTool)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Math Calculation",
            "prompt": "Calculate the factorial of 5 and verify it by manual multiplication",
            "llm_scenario": "math"
        },
        {
            "name": "File Operations", 
            "prompt": "Create a Python script that prints 'Hello from Agentic Tools!' and run it",
            "llm_scenario": "file_ops"
        },
        {
            "name": "Search Operations",
            "prompt": "Search for Python files and then search for import statements in them",
            "llm_scenario": "search_and_edit"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Create agent configuration
        config = AgenticRolloutConfig(
            agent_type="react",
            max_steps=8,
            max_tokens_per_step=512,
            temperature=0.7
        )
        
        # Create mock LLM function for this scenario
        llm_func = MockLLMFunction(scenario['llm_scenario'])
        
        # Create agent
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        # Set up tools - create simple tool wrappers for the agent
        class ToolWrapper:
            def __init__(self, tool_name, registry):
                self.tool_name = tool_name
                self.registry = registry
                self.instance_id = None
                
            async def execute(self, **kwargs):
                if not self.instance_id:
                    self.instance_id = await self.registry.create_tool_instance(self.tool_name)
                result = await self.registry.execute_tool(self.tool_name, self.instance_id, kwargs)
                if result.success:
                    return str(result.result)
                else:
                    return f"Error: {result.error}"
        
        tools = {
            "calculator": ToolWrapper("calculator", registry),
            "bash_executor": ToolWrapper("bash_executor", registry), 
            "file_editor": ToolWrapper("file_editor", registry),
            "search": ToolWrapper("search", registry)
        }
        
        rollout.agent.set_tools(tools)
        
        # Run trajectory
        prompt_data = {"content": scenario["prompt"]}
        
        try:
            trajectory = await rollout.agent.run_trajectory(
                prompt=prompt_data,
                llm_generate_func=llm_func,
                request_id=f"demo_{scenario['name'].lower().replace(' ', '_')}"
            )
            
            print(f"  Completed: {trajectory.is_completed}")
            print(f"  Steps: {len(trajectory.steps)}")
            print(f"  Final response: {trajectory.get_final_response()}")
            
            # Show trajectory steps
            print("  Trajectory:")
            for i, step in enumerate(trajectory.steps):
                if step.tool_name:
                    print(f"    {i+1}. TOOL: {step.tool_name}")
                else:
                    print(f"    {i+1}. {step.step_type.value}: {step.content[:60]}...")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Final cleanup
    await registry.release_all_instances()


async def main():
    """Run all demos."""
    print("Agentic Rollout Library - Tool Integration Demo")
    print("=" * 60)
    
    try:
        await demo_individual_tools()
        await demo_tool_registry()
        await demo_agent_with_tools()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nKey Features Demonstrated:")
        print("✓ Individual tool usage with async execution")
        print("✓ Global tool registry for centralized management")
        print("✓ Tool integration with ReAct agents")
        print("✓ VERL compatibility with standalone operation")
        print("✓ Comprehensive error handling and logging")
        print("✓ Flexible configuration and extensibility")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())