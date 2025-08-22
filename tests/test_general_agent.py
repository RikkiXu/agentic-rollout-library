#!/usr/bin/env python3
"""
Test script for GeneralAgent functionality with modern tool system.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent
from workers.agents.general_agent import dump_trajectory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLM:
    """Mock LLM for testing purposes with JSON format responses."""
    
    def __init__(self):
        self.call_count = 0
        # Predefined responses for testing with JSON format
        self.responses = [
            '''Thought: I need to solve this math problem step by step.

Action:
{
  "tool_name": "bash_executor",
  "parameters": {
    "command": "echo 'Calculating 2+3*4 = 14'"
  }
}''',
            '''Thought: The calculation is complete. Let me finish.

Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "The result of 2+3*4 is 14, using order of operations: 3*4=12, then 2+12=14"
  }
}'''
        ]
    
    async def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Mock LLM generation function."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            logger.info(f"Mock LLM response {self.call_count}: {response[:100]}...")
            return response
        else:
            # Fallback response
            return '''Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "Task completed"
  }
}'''


async def test_basic_functionality():
    """Test basic GeneralAgent functionality."""
    logger.info("Testing basic GeneralAgent functionality...")
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 10
        }),
        "finish": create_tool("Finish")
    }
    
    # Create agent
    agent = create_agent("General", {
        "max_rounds": 5,
        "termination_tool_names": ["finish"]
    })
    
    # Set tools
    agent.set_tools(tools)
    
    # Create mock LLM
    mock_llm = MockLLM()
    
    # Test trajectory
    prompt = "Calculate 2 + 3 * 4"
    trajectory = await agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=mock_llm,
        request_id="test_001"
    )
    
    # Verify trajectory
    logger.info(f"Trajectory completed: {trajectory.is_completed}")
    logger.info(f"Number of steps: {len(trajectory.steps)}")
    
    # Dump trajectory to file
    dump_trajectory(trajectory, "test_trajectory.json", "json")
    dump_trajectory(trajectory, "test_trajectory.txt", "txt")
    
    logger.info("Basic functionality test completed!")
    return trajectory


async def test_custom_system_prompt():
    """Test GeneralAgent with custom system prompt."""
    logger.info("Testing GeneralAgent with custom system prompt...")
    
    custom_prompt = """You are a helpful assistant that uses JSON-formatted actions.
    
    Always use this format:
    
    Thought: [Your reasoning]
    
    Action:
    {
      "tool_name": "tool_name", 
      "parameters": {
        "param": "value"
      }
    }
    
    Available tools include bash_executor for running commands and finish for completion."""
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 10
        }),
        "finish": create_tool("Finish")
    }
    
    agent = create_agent("General", {
        "system_prompt": custom_prompt,
        "max_rounds": 3,
        "termination_tool_names": ["finish"]
    })
    
    agent.set_tools(tools)
    
    # Test with different responses
    class CustomMockLLM:
        def __init__(self):
            self.responses = [
                '''Thought: This is a simple calculation problem.

Action:
{
  "tool_name": "bash_executor",
  "parameters": {
    "command": "echo 'Result: 5*6+2 = 32'"
  }
}''',
                '''Thought: The calculation is done, time to finish.

Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "The result is 32, calculated as 5*6=30, then 30+2=32"
  }
}'''
            ]
            self.call_count = 0
        
        async def __call__(self, messages, **kwargs):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            return '''Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "Done"
  }
}'''
    
    trajectory = await agent.run_trajectory(
        prompt="What is 5 * 6 + 2?",
        llm_generate_func=CustomMockLLM(),
        request_id="test_002"
    )
    
    logger.info(f"Custom prompt test completed: {trajectory.is_completed}")
    return trajectory


async def test_error_handling():
    """Test error handling in GeneralAgent."""
    logger.info("Testing error handling...")
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 10
        }),
        "finish": create_tool("Finish")
    }
    
    agent = create_agent("General", {
        "max_rounds": 4,
        "termination_tool_names": ["finish"]
    })
    
    agent.set_tools(tools)
    
    class ErrorMockLLM:
        def __init__(self):
            self.responses = [
                '''Thought: I'll try to use a non-existent tool.

Action:
{
  "tool_name": "nonexistent_tool",
  "parameters": {
    "param": "value"
  }
}''',
                '''Thought: That failed, let me try the bash executor.

Action:
{
  "tool_name": "bash_executor", 
  "parameters": {
    "command": "echo 'Error recovery successful: 1+1=2'"
  }
}''',
                '''Thought: Command executed successfully.

Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "After handling error, successfully executed command"
  }
}'''
            ]
            self.call_count = 0
        
        async def __call__(self, messages, **kwargs):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            return '''Action:
{
  "tool_name": "finish",
  "parameters": {
    "answer": "Done"
  }
}'''
    
    trajectory = await agent.run_trajectory(
        prompt="Test error handling",
        llm_generate_func=ErrorMockLLM(),
        request_id="test_003"
    )
    
    logger.info(f"Error handling test completed: {trajectory.is_completed}")
    
    # Check if error was handled properly
    error_steps = [step for step in trajectory.steps if step.metadata.get("action_failed")]
    logger.info(f"Found {len(error_steps)} error steps")
    
    return trajectory


async def main():
    """Run all tests."""
    logger.info("Starting GeneralAgent tests...")
    
    try:
        # Run tests
        await test_basic_functionality()
        await test_custom_system_prompt()
        await test_error_handling()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())