#!/usr/bin/env python3
"""
Test script for AgenticRollout functionality.

This script demonstrates how to use AgenticRollout independently 
for testing and development.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any

# Import the agentic rollout library
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from workers import AgenticRollout, AgenticRolloutConfig

# Try to import VERL DataProto if available, otherwise use mock
try:
    from verl.protocol import DataProto
except ImportError:
    # Mock DataProto for standalone usage
    class DataProto:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMGenerator:
    """Mock LLM generator for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    async def __call__(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Mock LLM generation."""
        self.call_count += 1
        
        # Get the last message content
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple rule-based responses for testing
        if "what is 2+3" in last_message.lower():
            if self.call_count == 1:
                return "Thought: I need to calculate 2+3. This is a simple addition problem."
            elif self.call_count == 2:
                return "Action: calculate_answer(answer='5')"
            else:
                return "Action: Final Answer: The answer is 5."
        
        elif "solve" in last_message.lower() and "math" in last_message.lower():
            if self.call_count == 1:
                return "Thought: I need to solve this math problem step by step."
            elif self.call_count == 2:
                return "Action: execute_calculation(code='result = 2 + 3; print(result)')"
            else:
                return "Action: Final Answer: Based on my calculation, the answer is 5."
        
        else:
            return f"Thought: I'm processing this request (call {self.call_count}). Let me think about what to do next."


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.description = f"Mock tool: {name}"
    
    async def execute(self, **kwargs) -> str:
        """Mock tool execution."""
        logger.info(f"Executing mock tool {self.name} with args: {kwargs}")
        
        if self.name == "calculate_answer":
            answer = kwargs.get("answer", "unknown")
            return f"Calculated answer: {answer}"
        
        elif self.name == "execute_calculation":
            code = kwargs.get("code", "")
            return f"Executed code: {code}\nOutput: 5"
        
        else:
            return f"Mock result from {self.name}"


async def test_react_agent():
    """Test ReAct agent with a simple math problem."""
    logger.info("Testing ReAct agent...")
    
    # Create configuration
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=5,
        max_tokens_per_step=256,
        temperature=0.3,
        include_trajectory_in_output=True
    )
    
    # Create mock LLM generator
    llm_generator = MockLLMGenerator()
    
    # Create rollout
    rollout = AgenticRollout(
        config=config,
        llm_generate_func=llm_generator
    )
    
    # Set up mock tools
    rollout.tools = {
        "calculate_answer": MockTool("calculate_answer"),
        "execute_calculation": MockTool("execute_calculation")
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Create test input
    test_prompts = DataProto(
        batch={
            "input_ids": [[1, 2, 3, 4]],  # Mock token IDs
        },
        non_tensor_batch={
            "raw_prompt": ["What is 2+3? Please solve this step by step."],
            "messages": [[
                {"role": "user", "content": "What is 2+3? Please solve this step by step."}
            ]]
        },
        meta_info={"validate": False}
    )
    
    # Run rollout
    try:
        output = await rollout.generate_sequences(test_prompts)
        
        # Print results
        logger.info("Rollout completed successfully!")
        logger.info(f"Generated responses: {output.non_tensor_batch.get('responses', [])}")
        
        if "trajectories" in output.non_tensor_batch:
            trajectory = output.non_tensor_batch["trajectories"][0]
            logger.info(f"Trajectory steps: {len(trajectory['steps'])}")
            for i, step in enumerate(trajectory["steps"]):
                logger.info(f"Step {i+1}: {step['step_type']} - {step['content'][:100]}...")
        
        return output
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


async def test_tool_agent():
    """Test Tool agent with a simple problem."""
    logger.info("Testing Tool agent...")
    
    # Create configuration
    config = AgenticRolloutConfig(
        agent_type="tool",
        max_steps=3,
        max_tokens_per_step=256,
        temperature=0.3,
        include_trajectory_in_output=True
    )
    
    # Create mock LLM generator
    llm_generator = MockLLMGenerator()
    
    # Create rollout
    rollout = AgenticRollout(
        config=config,
        llm_generate_func=llm_generator
    )
    
    # Set up mock tools
    rollout.tools = {
        "calculate_answer": MockTool("calculate_answer")
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Create test input
    test_prompts = DataProto(
        batch={
            "input_ids": [[1, 2, 3, 4]],
        },
        non_tensor_batch={
            "raw_prompt": ["Calculate 2+3"],
            "messages": [[
                {"role": "user", "content": "Calculate 2+3"}
            ]]
        },
        meta_info={"validate": False}
    )
    
    # Run rollout
    try:
        output = await rollout.generate_sequences(test_prompts)
        logger.info("Tool agent test completed successfully!")
        return output
        
    except Exception as e:
        logger.error(f"Tool agent test failed: {e}")
        raise


def save_test_results(results: Dict[str, Any], filename: str = "test_results.json"):
    """Save test results to file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting AgenticRollout tests...")
    
    results = {}
    
    try:
        # Test ReAct agent
        react_output = await test_react_agent()
        results["react_agent"] = {
            "success": True,
            "output": react_output.non_tensor_batch if react_output else None
        }
        
    except Exception as e:
        logger.error(f"ReAct agent test failed: {e}")
        results["react_agent"] = {
            "success": False,
            "error": str(e)
        }
    
    try:
        # Test Tool agent
        tool_output = await test_tool_agent()
        results["tool_agent"] = {
            "success": True,
            "output": tool_output.non_tensor_batch if tool_output else None
        }
        
    except Exception as e:
        logger.error(f"Tool agent test failed: {e}")
        results["tool_agent"] = {
            "success": False,
            "error": str(e)
        }
    
    # Save results
    save_test_results(results)
    
    # Print summary
    logger.info("Test Summary:")
    for test_name, result in results.items():
        status = "PASSED" if result["success"] else "FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())