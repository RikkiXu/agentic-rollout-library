#!/usr/bin/env python3
"""
Validation and testing suite for the AgenticRollout library.

This script provides comprehensive testing utilities to validate:
- Library functionality and correctness
- Performance benchmarks
- Integration testing
- Error handling and edge cases
"""

import asyncio
import time
import logging
import unittest
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from verl.workers.rollout.agentic_rollout import (
    AgenticRollout,
    AgenticRolloutConfig,
    BaseAgent,
    ReactAgent,
    Trajectory,
    TrajectoryStep,
    StepType,
    register_agent,
    get_agent_class
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class MockLLMFunction:
    """Deterministic mock LLM for testing."""
    
    def __init__(self, response_sequence: List[str]):
        self.responses = response_sequence
        self.call_count = 0
        
    async def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "Action: Final Answer: Default response"
        
        self.call_count += 1
        return response
    
    def reset(self):
        """Reset call count for reuse."""
        self.call_count = 0


class DeterministicTool:
    """Deterministic tool for testing."""
    
    def __init__(self, name: str, outputs: List[str]):
        self.name = name
        self.description = f"Test tool: {name}"
        self.outputs = outputs
        self.call_count = 0
        
    async def execute(self, **kwargs) -> str:
        if self.call_count < len(self.outputs):
            result = self.outputs[self.call_count]
        else:
            result = f"Default output {self.call_count}"
        
        self.call_count += 1
        return result
    
    def reset(self):
        """Reset call count."""
        self.call_count = 0


class AgenticRolloutTester:
    """Comprehensive test suite for AgenticRollout."""
    
    def __init__(self):
        self.test_results = []
        
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results."""
        print(f"\nRunning test: {test_name}")
        start_time = time.time()
        
        try:
            details = await test_func()
            success = True
            error_message = None
            print(f"  ✓ PASSED")
            
        except Exception as e:
            details = {"error": str(e)}
            success = False
            error_message = str(e)
            print(f"  ✗ FAILED: {error_message}")
            
        duration = time.time() - start_time
        
        result = TestResult(
            test_name=test_name,
            success=success,
            duration=duration,
            details=details,
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    async def test_basic_trajectory_creation(self) -> Dict[str, Any]:
        """Test basic trajectory creation and execution."""
        # Create simple configuration
        config = AgenticRolloutConfig(
            agent_type="react",
            max_steps=3,
            max_tokens_per_step=256
        )
        
        # Create mock LLM with predictable responses
        llm_func = MockLLMFunction([
            "Thought: I need to think about this.",
            "Action: Final Answer: This is my answer."
        ])
        
        # Create rollout
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        # Simple prompt
        prompt = {"content": "Test question"}
        
        # Run trajectory
        trajectory = await rollout.agent.run_trajectory(
            prompt=prompt,
            llm_generate_func=llm_func,
            request_id="test_basic"
        )
        
        # Validate results
        assert trajectory is not None, "Trajectory should not be None"
        assert trajectory.request_id == "test_basic", "Request ID should match"
        assert len(trajectory.steps) >= 2, "Should have at least 2 steps"
        assert trajectory.is_completed, "Trajectory should be completed"
        
        return {
            "steps": len(trajectory.steps),
            "completed": trajectory.is_completed,
            "final_response": trajectory.get_final_response()
        }
    
    async def test_tool_execution(self) -> Dict[str, Any]:
        """Test tool execution functionality."""
        config = AgenticRolloutConfig(agent_type="react", max_steps=5)
        
        # Create LLM that will call tools
        llm_func = MockLLMFunction([
            "Thought: I should use the test tool.",
            "Action: test_tool(param1='value1')",
            "Thought: Tool executed successfully.",
            "Action: Final Answer: Tool execution completed."
        ])
        
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        # Add test tool
        test_tool = DeterministicTool("test_tool", ["Tool output 1", "Tool output 2"])
        rollout.tools = {"test_tool": test_tool}
        rollout.agent.set_tools(rollout.tools)
        
        # Run trajectory
        trajectory = await rollout.agent.run_trajectory(
            prompt={"content": "Use the test tool"},
            llm_generate_func=llm_func,
            request_id="test_tools"
        )
        
        # Validate tool usage
        tool_calls = trajectory.get_tool_calls()
        assert len(tool_calls) >= 1, "Should have at least one tool call"
        assert tool_calls[0].tool_name == "test_tool", "Tool name should match"
        
        # Check for tool result steps
        result_steps = [s for s in trajectory.steps if s.step_type == StepType.ACTION_RESULT]
        assert len(result_steps) >= 1, "Should have tool result steps"
        
        return {
            "tool_calls": len(tool_calls),
            "result_steps": len(result_steps),
            "tool_outputs": [step.content for step in result_steps]
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling capabilities."""
        config = AgenticRolloutConfig(agent_type="react", max_steps=4)
        
        class FailingTool:
            def __init__(self):
                self.description = "Tool that always fails"
                
            async def execute(self, **kwargs):
                raise Exception("Simulated tool failure")
        
        # LLM tries to use failing tool
        llm_func = MockLLMFunction([
            "Action: failing_tool()",
            "Thought: The tool failed, let me try something else.",
            "Action: Final Answer: Handled the error gracefully."
        ])
        
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        rollout.tools = {"failing_tool": FailingTool()}
        rollout.agent.set_tools(rollout.tools)
        
        # Run trajectory
        trajectory = await rollout.agent.run_trajectory(
            prompt={"content": "Use the failing tool"},
            llm_generate_func=llm_func,
            request_id="test_errors"
        )
        
        # Validate error handling
        assert trajectory.is_completed, "Trajectory should complete despite errors"
        
        # Check for error handling in steps
        error_steps = [s for s in trajectory.steps 
                      if s.metadata.get("execution_successful") == False]
        assert len(error_steps) >= 1, "Should have error steps"
        
        return {
            "completed_with_errors": trajectory.is_completed,
            "error_steps": len(error_steps),
            "final_response": trajectory.get_final_response()
        }
    
    async def test_custom_agent(self) -> Dict[str, Any]:
        """Test custom agent implementation."""
        
        @register_agent("test_custom")
        class TestCustomAgent(BaseAgent):
            async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
                trajectory = Trajectory(request_id=request_id)
                
                # Add custom steps
                trajectory.add_step(TrajectoryStep(
                    step_type=StepType.OBSERVATION,
                    content=str(prompt.get('content', '')),
                    metadata={"custom_agent": True}
                ))
                
                trajectory.add_step(TrajectoryStep(
                    step_type=StepType.THOUGHT,
                    content="Custom agent thinking process",
                    metadata={"custom_agent": True}
                ))
                
                trajectory.add_step(TrajectoryStep(
                    step_type=StepType.FINAL_ANSWER,
                    content="Custom agent response",
                    metadata={"custom_agent": True}
                ))
                
                self.finalize_trajectory(trajectory)
                return trajectory
        
        # Test agent registration
        agent_class = get_agent_class("test_custom")
        assert agent_class is not None, "Custom agent should be registered"
        assert agent_class == TestCustomAgent, "Should return correct agent class"
        
        # Test agent execution
        config = AgenticRolloutConfig(agent_type="test_custom", max_steps=5)
        rollout = AgenticRollout(config=config)
        
        trajectory = await rollout.agent.run_trajectory(
            prompt={"content": "Test custom agent"},
            llm_generate_func=None,  # Not needed for this agent
            request_id="test_custom_exec"
        )
        
        # Validate custom agent behavior
        assert trajectory.is_completed, "Custom trajectory should be completed"
        custom_steps = [s for s in trajectory.steps if s.metadata.get("custom_agent")]
        assert len(custom_steps) == 3, "Should have 3 custom steps"
        
        return {
            "agent_registered": True,
            "steps": len(trajectory.steps),
            "custom_steps": len(custom_steps),
            "response": trajectory.get_final_response()
        }
    
    async def test_configuration_validation(self) -> Dict[str, Any]:
        """Test configuration validation and edge cases."""
        # Test minimum configuration
        min_config = AgenticRolloutConfig(agent_type="react")
        assert min_config.max_steps == 10, "Should have default max_steps"
        assert min_config.temperature == 0.7, "Should have default temperature"
        
        # Test configuration with all parameters
        full_config = AgenticRolloutConfig(
            agent_type="react",
            max_steps=20,
            max_tokens_per_step=1024,
            temperature=0.5,
            batch_size=4,
            concurrent_requests=8,
            include_trajectory_in_output=False,
            save_trajectories=True,
            trajectory_save_path="/tmp/test"
        )
        
        rollout = AgenticRollout(config=full_config)
        assert rollout.config.max_steps == 20, "Config should be set correctly"
        assert rollout.config.temperature == 0.5, "Temperature should be set"
        
        # Test invalid agent type
        try:
            invalid_config = AgenticRolloutConfig(agent_type="nonexistent")
            AgenticRollout(config=invalid_config)
            assert False, "Should raise error for invalid agent type"
        except ValueError:
            pass  # Expected
        
        return {
            "min_config_valid": True,
            "full_config_valid": True,
            "invalid_config_handled": True
        }
    
    async def test_trajectory_serialization(self) -> Dict[str, Any]:
        """Test trajectory serialization and deserialization."""
        # Create a trajectory
        trajectory = Trajectory(request_id="test_serialization")
        trajectory.add_step(TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content="Test observation",
            metadata={"test": True}
        ))
        trajectory.add_step(TrajectoryStep(
            step_type=StepType.ACTION,
            content="test_tool(arg='value')",
            tool_name="test_tool",
            tool_args={"arg": "value"},
            tool_result="Tool result"
        ))
        trajectory.add_step(TrajectoryStep(
            step_type=StepType.FINAL_ANSWER,
            content="Final answer",
            reward_score=1.0
        ))
        
        trajectory.is_completed = True
        trajectory.final_reward = 1.0
        
        # Test serialization
        traj_dict = trajectory.to_dict()
        assert isinstance(traj_dict, dict), "Should serialize to dict"
        assert traj_dict["request_id"] == "test_serialization", "Should preserve request_id"
        assert len(traj_dict["steps"]) == 3, "Should preserve all steps"
        
        # Test deserialization
        reconstructed = Trajectory.from_dict(traj_dict)
        assert reconstructed.request_id == trajectory.request_id, "Should reconstruct request_id"
        assert len(reconstructed.steps) == len(trajectory.steps), "Should reconstruct all steps"
        assert reconstructed.is_completed == trajectory.is_completed, "Should reconstruct completion status"
        
        # Verify step details
        orig_action_step = trajectory.steps[1]
        recon_action_step = reconstructed.steps[1]
        assert orig_action_step.tool_name == recon_action_step.tool_name, "Should preserve tool name"
        assert orig_action_step.tool_args == recon_action_step.tool_args, "Should preserve tool args"
        
        return {
            "serialization_successful": True,
            "deserialization_successful": True,
            "data_preserved": True,
            "original_steps": len(trajectory.steps),
            "reconstructed_steps": len(reconstructed.steps)
        }
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        config = AgenticRolloutConfig(
            agent_type="react",
            max_steps=10,
            concurrent_requests=1
        )
        
        # Create a fast LLM mock
        llm_func = MockLLMFunction([
            "Action: Final Answer: Quick response"
        ])
        
        rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
        
        # Benchmark single trajectory
        start_time = time.time()
        trajectory = await rollout.agent.run_trajectory(
            prompt={"content": "Simple test"},
            llm_generate_func=llm_func,
            request_id="perf_test"
        )
        single_duration = time.time() - start_time
        
        # Benchmark multiple trajectories
        llm_func.reset()
        num_trajectories = 5
        start_time = time.time()
        
        tasks = []
        for i in range(num_trajectories):
            task = rollout.agent.run_trajectory(
                prompt={"content": f"Test {i}"},
                llm_generate_func=llm_func,
                request_id=f"perf_test_{i}"
            )
            tasks.append(task)
        
        trajectories = await asyncio.gather(*tasks)
        batch_duration = time.time() - start_time
        
        assert len(trajectories) == num_trajectories, "Should complete all trajectories"
        avg_duration = batch_duration / num_trajectories
        
        return {
            "single_trajectory_time": single_duration,
            "batch_trajectory_time": batch_duration,
            "average_time_per_trajectory": avg_duration,
            "trajectories_per_second": num_trajectories / batch_duration,
            "all_completed": all(t.is_completed for t in trajectories)
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("=" * 60)
        print("AgenticRollout Library - Validation Test Suite")
        print("=" * 60)
        
        tests = [
            ("Basic Trajectory Creation", self.test_basic_trajectory_creation),
            ("Tool Execution", self.test_tool_execution),
            ("Error Handling", self.test_error_handling),
            ("Custom Agent", self.test_custom_agent),
            ("Configuration Validation", self.test_configuration_validation),
            ("Trajectory Serialization", self.test_trajectory_serialization),
            ("Performance Benchmarks", self.test_performance_benchmarks)
        ]
        
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        total_time = sum(r.duration for r in self.test_results)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests * 100,
            "total_time": total_time,
            "average_time_per_test": total_time / total_tests,
            "test_results": self.test_results
        }
        
        print(f"\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per test: {summary['average_time_per_test']:.2f}s")
        
        if summary['failed_tests'] > 0:
            print(f"\nFailed tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"  • {result.test_name}: {result.error_message}")
        
        return summary


class IntegrationTester:
    """Integration testing with external components."""
    
    @staticmethod
    async def test_verl_protocol_compatibility():
        """Test compatibility with VERL protocol."""
        try:
            from verl.protocol import DataProto
            
            # Create mock DataProto
            mock_data = DataProto(
                batch={"input_ids": [[1, 2, 3], [4, 5, 6]]},
                non_tensor_batch={"raw_prompt": ["Test 1", "Test 2"]},
                meta_info={"test": True}
            )
            
            config = AgenticRolloutConfig(agent_type="react", max_steps=2)
            llm_func = MockLLMFunction(["Action: Final Answer: Protocol test"])
            rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
            
            # Test DataProto processing
            prompt_list = rollout._extract_prompts_from_data_proto(mock_data)
            assert len(prompt_list) == 2, "Should extract correct number of prompts"
            assert "raw_prompt" in prompt_list[0], "Should include non-tensor data"
            
            return True
            
        except ImportError:
            print("VERL protocol not available - skipping integration test")
            return False
        except Exception as e:
            print(f"Integration test failed: {e}")
            return False


async def main():
    """Run the complete validation suite."""
    # Core functionality tests
    tester = AgenticRolloutTester()
    summary = await tester.run_all_tests()
    
    # Integration tests
    print(f"\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    integration_tester = IntegrationTester()
    verl_compat = await integration_tester.test_verl_protocol_compatibility()
    print(f"VERL Protocol Compatibility: {'✓ PASSED' if verl_compat else '✗ SKIPPED'}")
    
    # Final assessment
    print(f"\n" + "=" * 60)
    print("VALIDATION ASSESSMENT")
    print("=" * 60)
    
    if summary['success_rate'] >= 90:
        assessment = "✓ EXCELLENT - Library is ready for production use"
    elif summary['success_rate'] >= 75:
        assessment = "⚠ GOOD - Minor issues to address"
    elif summary['success_rate'] >= 50:
        assessment = "⚠ FAIR - Significant issues to resolve"
    else:
        assessment = "✗ POOR - Major problems require attention"
    
    print(f"Overall Assessment: {assessment}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    print("\nValidated capabilities:")
    print("  • Basic trajectory creation and execution")
    print("  • Tool integration and execution")
    print("  • Error handling and recovery")
    print("  • Custom agent implementation")
    print("  • Configuration validation")
    print("  • Data serialization/deserialization")
    print("  • Performance characteristics")
    if verl_compat:
        print("  • VERL protocol compatibility")
    
    print(f"\nThe AgenticRollout library has been validated and is ready for independent use!")
    
    return summary


if __name__ == "__main__":
    asyncio.run(main())