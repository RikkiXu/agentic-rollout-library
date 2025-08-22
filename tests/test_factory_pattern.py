#!/usr/bin/env python3
"""
Test script for factory pattern tool and agent creation.
"""

import asyncio
import sys
import os
import logging

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_tools, create_agent, create_agents
from workers.core import get_global_tool_factory, get_global_agent_factory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_tool_factory():
    """Test the ToolFactory functionality."""
    print("=== Testing ToolFactory ===")
    
    # Get factory instance
    tool_factory = get_global_tool_factory()
    
    # List available tools
    available_tools = tool_factory.list_available_tools()
    print("Available tools:")
    for name, path in available_tools.items():
        print(f"  - {name}: {path}")
    
    # Test single tool creation
    print("\n--- Single Tool Creation ---")
    
    # Create Calculator tool with config
    calculator = create_tool("Calculator", {"debug": True, "precision": 10})
    print(f"Created Calculator: {type(calculator).__name__}")
    
    # Create Finish tool
    finish = create_tool("Finish")
    print(f"Created Finish: {type(finish).__name__}")
    
    # Test multiple tools creation
    print("\n--- Multiple Tools Creation ---")
    
    tool_configs = {
        "Calculator": {"debug": True, "max_results": 100},
        "Search": {"max_results": 50, "max_file_size": 1024000},
        "Finish": {}
    }
    
    tools = create_tools(tool_configs)
    print("Created tools:")
    for name, tool in tools.items():
        print(f"  - {name}: {type(tool).__name__}")
    
    # Test tool execution
    print("\n--- Tool Execution Test ---")
    
    try:
        # Test calculator
        import uuid
        instance_id = str(uuid.uuid4())
        result = await calculator.execute_tool(instance_id, {"expression": "2+3*4"})
        print(f"Calculator result: {result.success}, {result.result}")
        
        # Test finish tool
        instance_id = str(uuid.uuid4())
        result = await finish.execute_tool(instance_id, {"answer": "Test completed", "status": "success"})
        print(f"Finish result: {result.success}, {result.result}")
        
    except Exception as e:
        print(f"Tool execution error: {e}")
    
    return tools


async def test_agent_factory():
    """Test the AgentFactory functionality."""
    print("\n=== Testing AgentFactory ===")
    
    # Get factory instance
    agent_factory = get_global_agent_factory()
    
    # List available agents
    available_agents = agent_factory.list_available_agents()
    print("Available agents:")
    for name, path in available_agents.items():
        print(f"  - {name}: {path}")
    
    # Test single agent creation
    print("\n--- Single Agent Creation ---")
    
    # Create General agent with config
    general_agent = create_agent("General", {
        "max_rounds": 5,
        "termination_tool_names": ["finish"],
        "system_prompt": "You are a helpful assistant."
    })
    print(f"Created General agent: {type(general_agent).__name__}")
    print(f"Max rounds: {general_agent.max_rounds}")
    
    # Create React agent
    react_agent = create_agent("React", {"max_steps": 10})
    print(f"Created React agent: {type(react_agent).__name__}")
    
    # Test multiple agents creation
    print("\n--- Multiple Agents Creation ---")
    
    agent_configs = {
        "General": {
            "max_rounds": 3,
            "system_prompt": "You are a math tutor."
        },
        "React": {
            "max_steps": 8,
            "temperature": 0.8
        }
    }
    
    agents = create_agents(agent_configs)
    print("Created agents:")
    for name, agent in agents.items():
        print(f"  - {name}: {type(agent).__name__}")
    
    return agents


async def test_integrated_workflow():
    """Test integrated workflow with factory-created tools and agents."""
    print("\n=== Testing Integrated Workflow ===")
    
    # Create tools using factory
    tools = create_tools({
        "Calculator": {"debug": False},
        "Finish": {}
    })
    
    # Create agent using factory
    agent = create_agent("General", {
        "max_rounds": 3,
        "termination_tool_names": ["finish"]
    })
    
    # Set tools to agent
    agent.set_tools(tools)
    
    print(f"Agent has {len(agent.tools)} tools: {list(agent.tools.keys())}")
    
    # Mock LLM for testing
    class MockLLM:
        def __init__(self):
            self.responses = [
                "Thought: I need to calculate 5+3.",
                "Action: calculator(expression=5+3)",
                "Action: finish(answer=The result is 8)"
            ]
            self.call_count = 0
        
        async def __call__(self, messages, **kwargs):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            return "Action: finish(answer=Done)"
    
    # Test trajectory execution
    try:
        trajectory = await agent.run_trajectory(
            prompt="Calculate 5 + 3",
            llm_generate_func=MockLLM(),
            request_id="factory_test_001"
        )
        
        print(f"Trajectory completed: {trajectory.is_completed}")
        print(f"Steps: {len(trajectory.steps)}")
        
        for i, step in enumerate(trajectory.steps):
            print(f"  Step {i+1}: {step.step_type.value} - {step.content[:50]}...")
    
    except Exception as e:
        print(f"Workflow execution error: {e}")
        import traceback
        traceback.print_exc()


def test_tool_info():
    """Test tool information retrieval."""
    print("\n=== Testing Tool Info ===")
    
    factory = get_global_tool_factory()
    
    # Get info for available tools
    for tool_name in ["Calculator", "Search", "Finish"]:
        try:
            info = factory.get_tool_info(tool_name)
            print(f"\n{tool_name} Info:")
            print(f"  Class: {info.get('class', 'N/A')}")
            print(f"  Description: {info.get('description', 'N/A')[:100]}...")
        except Exception as e:
            print(f"  Error getting info for {tool_name}: {e}")


def test_agent_info():
    """Test agent information retrieval."""
    print("\n=== Testing Agent Info ===")
    
    factory = get_global_agent_factory()
    
    # Get info for available agents
    for agent_name in ["General", "React"]:
        try:
            info = factory.get_agent_info(agent_name)
            print(f"\n{agent_name} Info:")
            print(f"  Class: {info.get('class', 'N/A')}")
            print(f"  Description: {info.get('doc', 'N/A')[:100]}...")
        except Exception as e:
            print(f"  Error getting info for {agent_name}: {e}")


async def main():
    """Run all tests."""
    print("Factory Pattern Testing")
    print("======================")
    
    try:
        await test_tool_factory()
        await test_agent_factory()
        await test_integrated_workflow()
        test_tool_info()
        test_agent_info()
        
        print("\n✅ All factory pattern tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("""
    Factory Pattern Features:
    
    🏭 **ToolFactory**:
    - Create tools by class name: create_tool("Calculator", config)
    - Support configuration arguments
    - Automatic module loading and caching
    - Built-in tool registration
    
    🏭 **AgentFactory**:
    - Create agents by class name: create_agent("General", config)
    - Support configuration arguments
    - Automatic module loading and caching
    - Built-in agent registration
    
    📝 **Configuration Examples**:
    
    Tools:
    ```python
    calculator = create_tool("Calculator", {"debug": True, "precision": 10})
    search = create_tool("Search", {"max_results": 100})
    ```
    
    Agents:
    ```python
    agent = create_agent("General", {
        "max_rounds": 5,
        "system_prompt": "You are a helper.",
        "termination_tool_names": ["finish"]
    })
    ```
    
    Batch Creation:
    ```python
    tools = create_tools({
        "Calculator": {"debug": True},
        "Search": {"max_results": 50}
    })
    
    agents = create_agents({
        "General": {"max_rounds": 3},
        "React": {"temperature": 0.8}
    })
    ```
    """)
    
    asyncio.run(main())