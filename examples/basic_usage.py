#!/usr/bin/env python3
"""
Basic usage examples for the AgenticRollout library.

This script demonstrates how to use the agentic rollout library independently
from VERL for various agentic workflows.
"""

import asyncio
import logging
from typing import Dict, List, Any

# Import the agentic rollout library
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from workers import (
    AgenticRollout,
    AgenticRolloutConfig,
    BaseAgent,
    ReactAgent,
    Trajectory,
    TrajectoryStep,
    StepType,
    create_agentic_rollout
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTool:
    """Mock tool for demonstration purposes."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> str:
        """Mock execution that returns a formatted result."""
        return f"Tool {self.name} executed with args: {kwargs}"


class MockLLMFunction:
    """Mock LLM function for demonstration."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or [
            "Thought: I need to solve this step by step.",
            "Action: calculator(expression='2+2')",
            "Thought: The result is 4. Now I can provide the final answer.",
            "Action: Final Answer: The result of 2+2 is 4."
        ]
        self.call_count = 0
    
    async def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Mock LLM generation."""
        logger.info(f"LLM called with {len(messages)} messages")
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "Action: Final Answer: I've completed the task."
        
        self.call_count += 1
        logger.info(f"LLM response: {response}")
        return response


async def example_1_basic_react_agent():
    """Example 1: Basic ReAct agent usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic ReAct Agent Usage")
    print("="*60)
    
    # Create configuration
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=5,
        max_tokens_per_step=256,
        temperature=0.7
    )
    
    # Create mock LLM function
    llm_func = MockLLMFunction([
        "Thought: I need to calculate 2+2.",
        "Action: calculator(expression='2+2')",
        "Thought: The calculator returned 4. This is correct.",
        "Action: Final Answer: The result of 2+2 is 4."
    ])
    
    # Create agentic rollout
    rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
    
    # Add mock tools
    rollout.tools = {
        "calculator": MockTool("calculator", "Performs mathematical calculations"),
        "search": MockTool("search", "Searches for information")
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Create a simple prompt
    prompt_data = {
        "content": "What is 2 + 2?",
        "task_type": "math"
    }
    
    # Run trajectory
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt_data,
        llm_generate_func=llm_func,
        request_id="example_1"
    )
    
    # Display results
    print(f"Trajectory completed: {trajectory.is_completed}")
    print(f"Total steps: {len(trajectory.steps)}")
    print(f"Final response: {trajectory.get_final_response()}")
    print(f"Total reward: {trajectory.get_total_reward()}")
    
    print("\nTrajectory steps:")
    for i, step in enumerate(trajectory.steps):
        print(f"  {i+1}. {step.step_type.value}: {step.content[:100]}")
    
    return trajectory


async def example_2_custom_agent():
    """Example 2: Custom agent implementation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Agent Implementation")
    print("="*60)
    
    class SimpleAgent(BaseAgent):
        """Simple custom agent for demonstration."""
        
        async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
            trajectory = Trajectory(request_id=request_id)
            
            # Add initial observation
            initial_step = TrajectoryStep(
                step_type=StepType.OBSERVATION,
                content=str(prompt.get('content', prompt)),
                metadata={"prompt": prompt}
            )
            trajectory.add_step(initial_step)
            
            # Simple two-step process: think then answer
            # Step 1: Think
            think_step = TrajectoryStep(
                step_type=StepType.THOUGHT,
                content="Let me think about this question carefully.",
                metadata={}
            )
            trajectory.add_step(think_step)
            
            # Step 2: Final answer
            answer_step = TrajectoryStep(
                step_type=StepType.FINAL_ANSWER,
                content="Based on my analysis, here is my response to your question.",
                metadata={}
            )
            trajectory.add_step(answer_step)
            
            self.finalize_trajectory(trajectory)
            return trajectory
    
    # Create custom agent
    agent = SimpleAgent(max_steps=3)
    
    # Run trajectory
    prompt_data = {"content": "How does photosynthesis work?"}
    trajectory = await agent.run_trajectory(
        prompt=prompt_data,
        llm_generate_func=None,  # Not needed for this simple agent
        request_id="custom_agent_example"
    )
    
    print(f"Custom agent trajectory:")
    print(f"  Completed: {trajectory.is_completed}")
    print(f"  Steps: {len(trajectory.steps)}")
    for step in trajectory.steps:
        print(f"    {step.step_type.value}: {step.content}")
    
    return trajectory


async def example_3_batch_processing():
    """Example 3: Batch processing multiple prompts."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Create configuration with higher concurrency
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=3,
        concurrent_requests=3,
        include_trajectory_in_output=True
    )
    
    # Create different LLM responses for variety
    def create_llm_func(task_type: str):
        if task_type == "math":
            responses = [
                "Thought: This is a math problem.",
                "Action: Final Answer: The answer is 42."
            ]
        elif task_type == "search":
            responses = [
                "Thought: I need to search for information.",
                "Action: search(query='information')",
                "Action: Final Answer: Here's what I found."
            ]
        else:
            responses = [
                "Thought: Let me think about this.",
                "Action: Final Answer: Here's my response."
            ]
        return MockLLMFunction(responses)
    
    # Create multiple prompts
    prompts = [
        {"content": "What is 6 * 7?", "task_type": "math"},
        {"content": "Search for information about AI", "task_type": "search"},
        {"content": "Explain machine learning", "task_type": "general"}
    ]
    
    # Process each prompt
    trajectories = []
    for i, prompt in enumerate(prompts):
        llm_func = create_llm_func(prompt["task_type"])
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        # Add mock search tool for search tasks
        if prompt["task_type"] == "search":
            rollout.tools = {"search": MockTool("search", "Searches for information")}
            rollout.agent.set_tools(rollout.tools)
        
        trajectory = await rollout.agent.run_trajectory(
            prompt=prompt,
            llm_generate_func=llm_func,
            request_id=f"batch_{i}"
        )
        trajectories.append(trajectory)
    
    # Display batch results
    print(f"Processed {len(trajectories)} trajectories:")
    for i, traj in enumerate(trajectories):
        print(f"  {i+1}. {traj.request_id}: {traj.get_final_response()[:50]}...")
    
    return trajectories


async def example_4_tool_integration():
    """Example 4: Advanced tool integration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Tool Integration")
    print("="*60)
    
    class AdvancedCalculator:
        """Advanced calculator tool with multiple operations."""
        
        def __init__(self):
            self.description = "Advanced calculator that can perform various mathematical operations"
        
        async def execute(self, operation: str, **kwargs) -> str:
            if operation == "add":
                result = kwargs.get("a", 0) + kwargs.get("b", 0)
                return f"Addition result: {result}"
            elif operation == "multiply":
                result = kwargs.get("a", 1) * kwargs.get("b", 1)
                return f"Multiplication result: {result}"
            elif operation == "factorial":
                n = kwargs.get("n", 1)
                if n <= 1:
                    result = 1
                else:
                    result = 1
                    for i in range(2, n + 1):
                        result *= i
                return f"Factorial of {n} is {result}"
            else:
                return f"Unknown operation: {operation}"
    
    class DataStore:
        """Simple data storage tool."""
        
        def __init__(self):
            self.description = "Stores and retrieves data"
            self.data = {}
        
        async def execute(self, action: str, key: str = None, value: str = None) -> str:
            if action == "store":
                self.data[key] = value
                return f"Stored {key} = {value}"
            elif action == "retrieve":
                return f"Retrieved {key} = {self.data.get(key, 'Not found')}"
            elif action == "list":
                return f"Stored keys: {list(self.data.keys())}"
            else:
                return f"Unknown action: {action}"
    
    # Create configuration
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=8,
        max_tokens_per_step=512
    )
    
    # Create LLM function with tool-aware responses
    llm_func = MockLLMFunction([
        "Thought: I need to calculate the factorial of 5 and store the result.",
        "Action: calculator(operation='factorial', n=5)",
        "Thought: Great! Now I'll store this result.",
        "Action: datastore(action='store', key='factorial_5', value='120')",
        "Thought: Let me verify what I've stored.",
        "Action: datastore(action='list')",
        "Action: Final Answer: I calculated factorial(5) = 120 and stored it successfully."
    ])
    
    # Create rollout with advanced tools
    rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
    rollout.tools = {
        "calculator": AdvancedCalculator(),
        "datastore": DataStore()
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Run complex task
    prompt_data = {
        "content": "Calculate the factorial of 5 and store the result for later use.",
        "complexity": "high"
    }
    
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt_data,
        llm_generate_func=llm_func,
        request_id="advanced_tools"
    )
    
    print(f"Advanced tool usage trajectory:")
    print(f"  Steps: {len(trajectory.steps)}")
    print(f"  Tool calls: {len(trajectory.get_tool_calls())}")
    print(f"  Final response: {trajectory.get_final_response()}")
    
    print("\nDetailed trajectory:")
    for i, step in enumerate(trajectory.steps):
        if step.tool_name:
            print(f"  {i+1}. {step.step_type.value}: {step.tool_name}({step.tool_args})")
        else:
            print(f"  {i+1}. {step.step_type.value}: {step.content[:80]}...")
    
    return trajectory


async def example_5_configuration_variations():
    """Example 5: Different configuration scenarios."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Configuration Variations")
    print("="*60)
    
    configs = [
        {
            "name": "High-step detailed",
            "config": AgenticRolloutConfig(
                agent_type="react",
                max_steps=10,
                max_tokens_per_step=1024,
                temperature=0.1,  # Low temperature for consistency
                concurrent_requests=1
            )
        },
        {
            "name": "Fast parallel",
            "config": AgenticRolloutConfig(
                agent_type="react", 
                max_steps=3,
                max_tokens_per_step=256,
                temperature=0.9,  # High temperature for variety
                concurrent_requests=5
            )
        },
        {
            "name": "Balanced",
            "config": AgenticRolloutConfig(
                agent_type="react",
                max_steps=5,
                max_tokens_per_step=512,
                temperature=0.7,
                concurrent_requests=2,
                include_trajectory_in_output=True,
                save_trajectories=False
            )
        }
    ]
    
    prompt_data = {"content": "Explain the concept of recursion in programming."}
    
    for config_info in configs:
        print(f"\n--- {config_info['name']} Configuration ---")
        config = config_info['config']
        
        llm_func = MockLLMFunction([
            "Thought: Let me explain recursion clearly.",
            "Action: Final Answer: Recursion is when a function calls itself to solve smaller subproblems."
        ])
        
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        trajectory = await rollout.agent.run_trajectory(
            prompt=prompt_data,
            llm_generate_func=llm_func,
            request_id=f"config_{config_info['name'].lower().replace(' ', '_')}"
        )
        
        print(f"  Max steps: {config.max_steps}, Actual steps: {len(trajectory.steps)}")
        print(f"  Temperature: {config.temperature}, Tokens per step: {config.max_tokens_per_step}")
        print(f"  Result: {trajectory.get_final_response()[:60]}...")
    
    return configs


async def main():
    """Run all examples."""
    print("AgenticRollout Library - Independent Usage Examples")
    print("=" * 60)
    
    # Run all examples
    examples = [
        example_1_basic_react_agent,
        example_2_custom_agent,
        example_3_batch_processing,
        example_4_tool_integration,
        example_5_configuration_variations
    ]
    
    results = {}
    for example_func in examples:
        try:
            result = await example_func()
            results[example_func.__name__] = result
        except Exception as e:
            logger.error(f"Error in {example_func.__name__}: {e}")
            results[example_func.__name__] = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Completed {len([r for r in results.values() if r is not None])}/{len(examples)} examples successfully")
    
    for name, result in results.items():
        status = "✓ SUCCESS" if result is not None else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print("\nThe AgenticRollout library can be used independently for:")
    print("  • Multi-step reasoning with ReAct agents")
    print("  • Custom agent implementations")
    print("  • Tool integration and execution")
    print("  • Batch processing of multiple tasks")
    print("  • Flexible configuration scenarios")
    print("  • Trajectory tracking and analysis")


if __name__ == "__main__":
    asyncio.run(main())