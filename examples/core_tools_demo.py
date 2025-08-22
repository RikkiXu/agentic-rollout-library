#!/usr/bin/env python3
"""
Demo script for the new core tool system.
Shows how to use the unified tool framework with or without VERL.
"""

import asyncio
import logging
from pathlib import Path

# Import from the workers module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from workers import (
    CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool,
    get_global_tool_registry, register_tool
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_calculator_tool():
    """Demo calculator tool functionality."""
    print("\n=== Calculator Tool Demo ===")
    
    calc = CalculatorTool({"debug": True})
    instance_id = await calc.create_instance()
    
    test_cases = [
        {"expression": "2 + 3 * 4"},
        {"expression": "sqrt(16) + factorial(4)"},
        {"operation": "power", "base": 2, "exponent": 8},
        {"numbers": [1, 2, 3, 4, 5], "operation": "average"},
        {"a": 10, "b": 3, "operation": "divide"},
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\nTest {i}: {params}")
        result = await calc.execute_tool(instance_id, params)
        
        if result.success:
            print(f"  Result: {result.result['formatted_result']}")
            print(f"  Type: {result.result['calculation_type']}")
        else:
            print(f"  Error: {result.error}")
    
    await calc.release_instance(instance_id)


async def demo_bash_executor_tool():
    """Demo bash executor tool functionality."""
    print("\n=== Bash Executor Tool Demo ===")
    
    bash = BashExecutorTool({"debug": True})
    instance_id = await bash.create_instance()
    
    test_commands = [
        {"command": "echo 'Hello World'"},
        {"command": "ls -la /tmp", "timeout": 10},
        {"command": "python -c 'print(2 + 2)'"},
        {"command": "pwd"},
    ]
    
    for i, params in enumerate(test_commands, 1):
        print(f"\nTest {i}: {params['command']}")
        result = await bash.execute_tool(instance_id, params)
        
        if result.success:
            print(f"  Stdout: {result.result['stdout'][:200]}...")  # Truncate output
            print(f"  Return code: {result.result['return_code']}")
        else:
            print(f"  Error: {result.error}")
    
    await bash.release_instance(instance_id)


async def demo_file_editor_tool():
    """Demo file editor tool functionality."""
    print("\n=== File Editor Tool Demo ===")
    
    editor = FileEditorTool({"debug": True})
    instance_id = await editor.create_instance()
    
    # Create a temporary file
    temp_file = "/tmp/demo_file.py"
    
    # Test create
    print(f"\n1. Creating file: {temp_file}")
    result = await editor.execute_tool(instance_id, {
        "command": "create",
        "path": temp_file,
        "file_text": "def hello():\n    print('Hello World')\n\nif __name__ == '__main__':\n    hello()\n"
    })
    print(f"  Success: {result.success}")
    if not result.success:
        print(f"  Error: {result.error}")
    
    # Test view
    print(f"\n2. Viewing file: {temp_file}")
    result = await editor.execute_tool(instance_id, {
        "command": "view",
        "path": temp_file
    })
    if result.success:
        print(f"  Content preview:\n{result.result['content'][:300]}...")
    else:
        print(f"  Error: {result.error}")
    
    # Test str_replace
    print(f"\n3. Replacing string in file")
    result = await editor.execute_tool(instance_id, {
        "command": "str_replace",
        "path": temp_file,
        "old_str": "Hello World",
        "new_str": "Hello from Agentic Tools!"
    })
    print(f"  Success: {result.success}")
    if not result.success:
        print(f"  Error: {result.error}")
    
    # Clean up
    try:
        Path(temp_file).unlink(missing_ok=True)
    except:
        pass
    
    await editor.release_instance(instance_id)


async def demo_search_tool():
    """Demo search tool functionality."""
    print("\n=== Search Tool Demo ===")
    
    search = SearchTool({"debug": True})
    instance_id = await search.create_instance()
    
    # Search in current directory
    current_dir = str(Path(__file__).parent)
    
    # Test text search
    print(f"\n1. Searching for 'import' in Python files")
    result = await search.execute_tool(instance_id, {
        "command": "search_text",
        "pattern": "import",
        "path": current_dir,
        "file_extensions": [".py"],
        "max_results": 5
    })
    
    if result.success:
        print(f"  Found {result.result['total_matches']} matches in {result.result['files_searched']} files")
        for match in result.result['matches'][:3]:  # Show first 3 matches
            print(f"    {match['file']}:{match['line_number']} - {match['match_text']}")
    else:
        print(f"  Error: {result.error}")
    
    # Test file name search
    print(f"\n2. Searching for files containing 'demo'")
    result = await search.execute_tool(instance_id, {
        "command": "search_files",
        "pattern": "demo",
        "path": current_dir,
        "max_results": 10
    })
    
    if result.success:
        print(f"  Found {result.result['total_matches']} matching files")
        for match in result.result['matches'][:5]:  # Show first 5 matches
            print(f"    {match['name']} ({match['type']})")
    else:
        print(f"  Error: {result.error}")
    
    await search.release_instance(instance_id)


async def demo_tool_registry():
    """Demo global tool registry functionality."""
    print("\n=== Tool Registry Demo ===")
    
    # Get global registry
    registry = get_global_tool_registry()
    
    # Register tools
    print("1. Registering tools in global registry")
    register_tool(CalculatorTool, {"debug": False}, "calc")
    register_tool(BashExecutorTool, {"timeout": 15}, "bash")
    register_tool(FileEditorTool, {"max_file_size": 512*1024}, "editor")
    register_tool(SearchTool, {"max_results": 50}, "search")
    
    print(f"  Registered tools: {registry.get_tool_names()}")
    
    # Create instances and test
    print("\n2. Creating tool instances")
    calc_instance = await registry.create_tool_instance("calc")
    bash_instance = await registry.create_tool_instance("bash")
    
    print(f"  Created calculator instance: {calc_instance}")
    print(f"  Created bash instance: {bash_instance}")
    
    # Execute through registry
    print("\n3. Executing tools through registry")
    result = await registry.execute_tool("calc", calc_instance, {"expression": "5 * 5"})
    print(f"  Calculator result: {result.result['result'] if result.success else result.error}")
    
    result = await registry.execute_tool("bash", bash_instance, {"command": "echo 'Registry test'"})
    print(f"  Bash result: {result.result['stdout'].strip() if result.success else result.error}")
    
    # Get tool info
    print("\n4. Tool information")
    info = registry.get_tool_info()
    for tool_info in info:
        print(f"  {tool_info['name']}: {tool_info['active_instances']} active instances")
    
    # Clean up
    print("\n5. Cleaning up")
    await registry.release_all_instances()
    print("  All instances released")


async def main():
    """Run all demos."""
    print("Agentic Rollout Library - Core Tools Demo")
    print("=" * 50)
    
    try:
        await demo_calculator_tool()
        await demo_bash_executor_tool()
        await demo_file_editor_tool()
        await demo_search_tool()
        await demo_tool_registry()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())