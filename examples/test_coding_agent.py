#!/usr/bin/env python3
"""
Test script for CodingAgent functionality.

This script demonstrates the CodingAgent's capabilities for SWE tasks.
"""

import asyncio
import logging
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

# Import the agentic rollout library
import sys
sys.path.append(str(Path(__file__).parent.parent))

from workers import AgenticRollout, AgenticRolloutConfig, CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool

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
    """Mock LLM generator for testing coding tasks."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = []
    
    async def __call__(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """Mock LLM generation with realistic coding responses."""
        self.call_count += 1
        
        # Get context from messages
        last_message = messages[-1]["content"].lower() if messages else ""
        conversation_context = " ".join([msg["content"] for msg in messages[-3:]])
        
        # Generate responses based on phase and context
        if "understanding" in last_message or self.call_count == 1:
            return self._generate_understanding_response(conversation_context)
        elif "exploration" in last_message or "search" in conversation_context:
            return self._generate_exploration_response(conversation_context)
        elif "planning" in last_message or "plan" in conversation_context:
            return self._generate_planning_response(conversation_context)
        elif "implementation" in last_message or "edit" in conversation_context:
            return self._generate_implementation_response(conversation_context)
        elif "testing" in last_message or "test" in conversation_context:
            return self._generate_testing_response(conversation_context)
        else:
            return self._generate_default_response(conversation_context)
    
    def _generate_understanding_response(self, context: str) -> str:
        """Generate understanding phase response."""
        return """Thought: I need to understand this problem clearly. Let me analyze the requirements.

From the problem description, I need to:
1. Understand what needs to be implemented
2. Identify the expected functionality
3. Determine what files might be involved

Let me start by exploring the codebase to understand the current structure."""
    
    def _generate_exploration_response(self, context: str) -> str:
        """Generate exploration phase response."""
        if "search_files" not in context:
            return "Action: search_files(pattern='*.py', directory='.')"
        elif "view_file" not in context:
            return "Action: view_file(file_path='main.py')"
        else:
            return "Action: search_code(query='def main', file_pattern='*.py')"
    
    def _generate_planning_response(self, context: str) -> str:
        """Generate planning phase response."""
        return """Thought: Based on my exploration, I can see the codebase structure. Here's my implementation plan:

1. Modify the main function to add the required functionality
2. Create helper functions if needed
3. Update any configuration or imports
4. Write tests to validate the changes
5. Run tests to ensure everything works

Let me start implementing these changes."""
    
    def _generate_implementation_response(self, context: str) -> str:
        """Generate implementation phase response."""
        if "edit_file" not in context:
            return """Action: edit_file(file_path='main.py', old_content='def main():', new_content='def main():
    # Added new functionality
    print("Hello, World!")
    return "success"')"""
        else:
            return "Action: create_file(file_path='utils.py', content='def helper_function():\\n    return True\\n')"
    
    def _generate_testing_response(self, context: str) -> str:
        """Generate testing phase response."""
        if "run_tests" not in context:
            return "Action: run_tests(test_command='python -m pytest test_main.py -v')"
        else:
            return "Thought: Tests are passing! The implementation is working correctly."
    
    def _generate_default_response(self, context: str) -> str:
        """Generate default response."""
        return f"Thought: Let me think about what to do next based on the current state (call {self.call_count})."


async def setup_test_environment():
    """Set up a test environment with sample files."""
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="coding_agent_test_")
    logger.info(f"Created test environment at: {temp_dir}")
    
    # Create sample files
    main_py_content = """def main():
    print("Original implementation")
    return "original"

if __name__ == "__main__":
    main()
"""
    
    test_py_content = """import unittest
from main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        result = main()
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
"""
    
    # Write sample files
    with open(Path(temp_dir) / "main.py", "w") as f:
        f.write(main_py_content)
    
    with open(Path(temp_dir) / "test_main.py", "w") as f:
        f.write(test_py_content)
    
    return temp_dir


async def test_coding_agent():
    """Test CodingAgent with a simple coding problem."""
    logger.info("Testing CodingAgent...")
    
    # Set up test environment
    sandbox_path = await setup_test_environment()
    
    try:
        # Create configuration
        config = AgenticRolloutConfig(
            agent_type="coding",
            max_steps=15,
            max_tokens_per_step=512,
            temperature=0.1,
            max_files_to_explore=10,
            enable_bash_execution=True,
            include_trajectory_in_output=True
        )
        
        # Create mock LLM generator
        llm_generator = MockLLMGenerator()
        
        # Create rollout
        rollout = AgenticRollout(
            config=config,
            llm_generate_func=llm_generator
        )
        
        # Set up SWE tools
        swe_tools = create_swe_tools(sandbox_path)
        rollout.tools = swe_tools
        rollout.agent.set_tools(swe_tools)
        
        # Create test input
        problem_description = """
        Problem: Modify the main.py file to implement a greeting function that:
        1. Takes a name parameter
        2. Returns a personalized greeting message
        3. Update the main() function to use this new greeting function
        4. Ensure all tests pass
        
        The expected output should be a greeting like "Hello, [name]!"
        """
        
        test_prompts = DataProto(
            batch={
                "input_ids": [[1, 2, 3, 4]],  # Mock token IDs
            },
            non_tensor_batch={
                "raw_prompt": [problem_description],
                "problem_description": [problem_description],
                "messages": [[
                    {"role": "user", "content": problem_description}
                ]]
            },
            meta_info={"validate": False}
        )
        
        # Run rollout
        output = await rollout.generate_sequences(test_prompts)
        
        # Analyze results
        logger.info("CodingAgent test completed successfully!")
        
        if "trajectories" in output.non_tensor_batch:
            trajectory = output.non_tensor_batch["trajectories"][0]
            
            logger.info(f"Trajectory completed with {len(trajectory['steps'])} steps")
            logger.info(f"Final phase: {trajectory['metadata'].get('final_phase', 'unknown')}")
            logger.info(f"Files explored: {trajectory['metadata'].get('files_explored', 0)}")
            logger.info(f"Files modified: {trajectory['metadata'].get('files_modified', 0)}")
            
            # Print step summary
            for i, step in enumerate(trajectory["steps"]):
                step_type = step["step_type"]
                content_preview = step["content"][:100] + "..." if len(step["content"]) > 100 else step["content"]
                logger.info(f"Step {i+1} ({step_type}): {content_preview}")
                
                if step.get("tool_name"):
                    logger.info(f"  Tool: {step['tool_name']} with args: {step.get('tool_args', {})}")
        
        return output
        
    except Exception as e:
        logger.error(f"CodingAgent test failed: {e}")
        raise
    
    finally:
        # Cleanup test environment
        import shutil
        shutil.rmtree(sandbox_path, ignore_errors=True)
        logger.info(f"Cleaned up test environment: {sandbox_path}")


async def test_swe_tools():
    """Test individual SWE tools."""
    logger.info("Testing SWE tools...")
    
    # Set up test environment
    sandbox_path = await setup_test_environment()
    
    try:
        # Create tools
        tools = create_swe_tools(sandbox_path)
        
        # Test file viewing
        view_result = await tools["view_file"].execute(file_path="main.py")
        logger.info(f"View file result: {view_result[:100]}...")
        
        # Test file search
        search_result = await tools["search_files"].execute(pattern="*.py")
        logger.info(f"File search result: {search_result}")
        
        # Test code search
        code_search_result = await tools["search_code"].execute(query="def main")
        logger.info(f"Code search result: {code_search_result}")
        
        # Test file editing
        edit_result = await tools["edit_file"].execute(
            file_path="main.py",
            old_content='print("Original implementation")',
            new_content='print("Modified implementation")'
        )
        logger.info(f"Edit result: {edit_result}")
        
        # Test bash execution
        bash_result = await tools["execute_bash"].execute(command="ls -la")
        logger.info(f"Bash result: {bash_result}")
        
        # Test running tests
        test_result = await tools["run_tests"].execute(
            test_command="python -m unittest test_main.py"
        )
        logger.info(f"Test result: {test_result}")
        
        logger.info("SWE tools test completed successfully!")
        
    except Exception as e:
        logger.error(f"SWE tools test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(sandbox_path, ignore_errors=True)


def save_test_results(results: Dict[str, Any], filename: str = "coding_agent_test_results.json"):
    """Save test results to file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting CodingAgent tests...")
    
    results = {}
    
    try:
        # Test SWE tools
        await test_swe_tools()
        results["swe_tools"] = {"success": True}
        
    except Exception as e:
        logger.error(f"SWE tools test failed: {e}")
        results["swe_tools"] = {"success": False, "error": str(e)}
    
    try:
        # Test CodingAgent
        coding_output = await test_coding_agent()
        results["coding_agent"] = {
            "success": True,
            "output": coding_output.non_tensor_batch if coding_output else None
        }
        
    except Exception as e:
        logger.error(f"CodingAgent test failed: {e}")
        results["coding_agent"] = {"success": False, "error": str(e)}
    
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