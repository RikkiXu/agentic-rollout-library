#!/usr/bin/env python3
"""
Advanced integration examples for the AgenticRollout library.

This script demonstrates advanced use cases including:
- Real LLM integration (OpenAI/Anthropic compatible)
- Complex multi-tool workflows
- Agent composition and chaining
- Production-ready error handling
- Performance monitoring
"""

import asyncio
import time
import logging
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

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
    register_agent,
    get_agent_class
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking for trajectories."""
    start_time: float
    end_time: float
    total_steps: int
    tool_calls: int
    tokens_generated: int
    success: bool
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealLLMFunction:
    """
    Real LLM function that can integrate with OpenAI, Anthropic, or local models.
    This is a mock that simulates real API calls.
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.call_count = 0
        
    async def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Simulate real LLM API call."""
        self.call_count += 1
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Extract the last user message to understand context
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break
        
        # Generate contextual responses based on conversation
        if "math" in last_message.lower() or "calculate" in last_message.lower():
            responses = [
                "Thought: I need to solve this mathematical problem step by step.",
                "Action: calculator(expression='{}')".format("the given expression"),
                "Thought: Let me verify this calculation is correct.",
                "Action: Final Answer: The mathematical result is computed correctly."
            ]
        elif "search" in last_message.lower() or "find" in last_message.lower():
            responses = [
                "Thought: I should search for this information.",
                "Action: web_search(query='{}')".format("relevant keywords"),
                "Thought: Based on the search results, I can provide an answer.",
                "Action: Final Answer: Here's what I found from my search."
            ]
        elif "code" in last_message.lower() or "program" in last_message.lower():
            responses = [
                "Thought: I need to write and test code for this request.",
                "Action: code_executor(language='python', code='# solution code')",
                "Thought: The code executed successfully. Let me explain the solution.",
                "Action: Final Answer: Here's the code solution with explanation."
            ]
        else:
            responses = [
                "Thought: Let me analyze this request carefully.",
                "Action: Final Answer: Based on my analysis, here's my response."
            ]
        
        # Return appropriate response based on call count
        response_idx = (self.call_count - 1) % len(responses)
        response = responses[response_idx]
        
        logger.info(f"LLM {self.model_name} response: {response[:50]}...")
        return response


class WebSearchTool:
    """Simulated web search tool."""
    
    def __init__(self):
        self.description = "Searches the web for information on any topic"
        
    async def execute(self, query: str, max_results: int = 5) -> str:
        """Simulate web search."""
        await asyncio.sleep(0.2)  # Simulate search delay
        
        # Mock search results based on query
        if "weather" in query.lower():
            return f"Weather search results for '{query}': Sunny, 75°F, light breeze"
        elif "news" in query.lower():
            return f"Latest news for '{query}': Breaking news articles found"
        elif "python" in query.lower():
            return f"Python programming results: Documentation, tutorials, and examples"
        else:
            return f"Search results for '{query}': {max_results} relevant articles found"


class CalculatorTool:
    """Advanced calculator with multiple functions."""
    
    def __init__(self):
        self.description = "Performs mathematical calculations and evaluations"
        
    async def execute(self, expression: str = None, operation: str = None, **kwargs) -> str:
        """Perform calculation."""
        try:
            if expression:
                # Safe evaluation (in real implementation, use a proper math parser)
                if any(dangerous in expression for dangerous in ['import', 'exec', 'eval', '__']):
                    return "Error: Invalid expression for security reasons"
                
                # Simulate complex calculation
                if "+" in expression:
                    parts = expression.split("+")
                    if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                        result = int(parts[0]) + int(parts[1])
                        return f"Calculation result: {expression} = {result}"
                
                return f"Calculated: {expression} = [simulated result]"
                
            elif operation:
                if operation == "factorial" and "n" in kwargs:
                    n = kwargs["n"]
                    result = 1
                    for i in range(1, n + 1):
                        result *= i
                    return f"Factorial of {n} is {result}"
                elif operation == "power" and "base" in kwargs and "exp" in kwargs:
                    result = kwargs["base"] ** kwargs["exp"]
                    return f"{kwargs['base']} ^ {kwargs['exp']} = {result}"
                    
            return "Calculator ready: please provide expression or operation"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"


class CodeExecutorTool:
    """Simulated code execution tool."""
    
    def __init__(self):
        self.description = "Executes code in various programming languages safely"
        
    async def execute(self, language: str, code: str, timeout: int = 10) -> str:
        """Simulate code execution."""
        await asyncio.sleep(0.3)  # Simulate execution time
        
        if language.lower() == "python":
            if "print" in code:
                return f"Code executed successfully:\nOutput: [simulated Python output]"
            elif "def" in code:
                return f"Function defined successfully:\n{code[:100]}..."
            else:
                return f"Python code executed: {code[:50]}..."
        elif language.lower() == "javascript":
            return f"JavaScript executed: {code[:50]}..."
        else:
            return f"Code in {language} executed successfully"


@register_agent("research")
class ResearchAgent(BaseAgent):
    """
    Custom research agent that specializes in information gathering and analysis.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_depth = kwargs.get("research_depth", 3)
        self.verification_enabled = kwargs.get("verification_enabled", True)
    
    def create_system_prompt(self) -> str:
        return """You are a research agent specialized in gathering and analyzing information.

Your process:
1. Understand the research question
2. Plan your research strategy  
3. Gather information using available tools
4. Verify and cross-reference findings
5. Synthesize a comprehensive answer

Use tools systematically and verify important claims when possible."""
    
    async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
        """Run research-focused trajectory."""
        trajectory = Trajectory(request_id=request_id)
        
        # Extract research question
        question = str(prompt.get('content', prompt))
        initial_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=question,
            metadata={"research_question": question}
        )
        trajectory.add_step(initial_step)
        
        research_steps = 0
        max_research_steps = self.research_depth
        
        while self.should_continue(trajectory) and research_steps < max_research_steps:
            try:
                # Generate next step
                messages = self.format_messages_for_llm(trajectory)
                response = await llm_generate_func(
                    messages,
                    max_tokens=self.max_tokens_per_step,
                    temperature=self.temperature
                )
                
                # Parse response
                step = self._parse_research_output(response)
                trajectory.add_step(step)
                
                if step.step_type == StepType.ACTION and step.tool_name:
                    # Execute research tool
                    result_step = await self.execute_tool_call(
                        step.tool_name, step.tool_args or {}, trajectory
                    )
                    trajectory.add_step(result_step)
                    research_steps += 1
                    
                elif step.step_type == StepType.FINAL_ANSWER:
                    break
                    
            except Exception as e:
                logger.error(f"Research error: {e}")
                break
        
        self.finalize_trajectory(trajectory)
        return trajectory
    
    def _parse_research_output(self, output: str) -> TrajectoryStep:
        """Parse research agent output."""
        output = output.strip()
        
        if output.startswith("Thought:"):
            return TrajectoryStep(
                step_type=StepType.THOUGHT,
                content=output[8:].strip(),
                metadata={"agent_type": "research"}
            )
        elif output.startswith("Action:"):
            action_content = output[7:].strip()
            if action_content.lower().startswith("final answer:"):
                return TrajectoryStep(
                    step_type=StepType.FINAL_ANSWER,
                    content=action_content[13:].strip(),
                    metadata={"agent_type": "research"}
                )
            else:
                # Parse tool call
                tool_name, tool_args = self._parse_simple_tool_call(action_content)
                return TrajectoryStep(
                    step_type=StepType.ACTION,
                    content=action_content,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    metadata={"agent_type": "research"}
                )
        else:
            return TrajectoryStep(
                step_type=StepType.THOUGHT,
                content=output,
                metadata={"agent_type": "research", "inferred": True}
            )
    
    def _parse_simple_tool_call(self, text: str) -> tuple:
        """Simple tool call parser."""
        if "(" in text and ")" in text:
            tool_name = text.split("(")[0].strip()
            args_text = text.split("(", 1)[1].rsplit(")", 1)[0]
            args = {}
            if args_text and "=" in args_text:
                for arg in args_text.split(","):
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        args[key.strip()] = value.strip().strip("\"'")
                    else:
                        args["query"] = args_text.strip().strip("\"'")
            else:
                args["query"] = args_text.strip().strip("\"'")
            return tool_name, args
        return None, {}


async def example_1_real_llm_integration():
    """Example 1: Integration with real LLM."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Real LLM Integration")
    print("="*60)
    
    # Create real LLM function
    llm_func = RealLLMFunction(model_name="gpt-4")
    
    # Create configuration
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=6,
        max_tokens_per_step=512,
        temperature=0.7
    )
    
    # Create rollout with real tools
    rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
    rollout.tools = {
        "calculator": CalculatorTool(),
        "web_search": WebSearchTool(),
        "code_executor": CodeExecutorTool()
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Test mathematical reasoning
    prompt = {
        "content": "Calculate the factorial of 6 and explain the mathematical concept",
        "domain": "mathematics"
    }
    
    start_time = time.time()
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=llm_func,
        request_id="real_llm_math"
    )
    end_time = time.time()
    
    # Track performance
    metrics = PerformanceMetrics(
        start_time=start_time,
        end_time=end_time,
        total_steps=len(trajectory.steps),
        tool_calls=len(trajectory.get_tool_calls()),
        tokens_generated=sum(len(step.content.split()) for step in trajectory.steps),
        success=trajectory.is_completed
    )
    
    print(f"Performance metrics:")
    print(f"  Duration: {metrics.duration:.2f}s")
    print(f"  Steps: {metrics.total_steps}")
    print(f"  Tool calls: {metrics.tool_calls}")
    print(f"  Est. tokens: {metrics.tokens_generated}")
    print(f"  Success: {metrics.success}")
    print(f"  Final answer: {trajectory.get_final_response()[:100]}...")
    
    return trajectory, metrics


async def example_2_multi_tool_workflow():
    """Example 2: Complex multi-tool workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Tool Workflow")
    print("="*60)
    
    class DatabaseTool:
        """Mock database tool."""
        def __init__(self):
            self.description = "Database operations"
            self.data = {}
            
        async def execute(self, operation: str, **kwargs) -> str:
            if operation == "store":
                self.data[kwargs.get("key")] = kwargs.get("value")
                return f"Stored: {kwargs.get('key')} = {kwargs.get('value')}"
            elif operation == "retrieve":
                value = self.data.get(kwargs.get("key"), "Not found")
                return f"Retrieved: {kwargs.get('key')} = {value}"
            elif operation == "list":
                return f"Database contents: {list(self.data.keys())}"
            return "Unknown operation"
    
    class FileSystemTool:
        """Mock file system tool."""
        def __init__(self):
            self.description = "File system operations"
            self.files = {}
            
        async def execute(self, operation: str, **kwargs) -> str:
            if operation == "write":
                self.files[kwargs.get("filename")] = kwargs.get("content")
                return f"Wrote file: {kwargs.get('filename')}"
            elif operation == "read":
                content = self.files.get(kwargs.get("filename"), "File not found")
                return f"File content: {content}"
            elif operation == "list":
                return f"Files: {list(self.files.keys())}"
            return "Unknown operation"
    
    # Create complex workflow agent
    config = AgenticRolloutConfig(
        agent_type="react",
        max_steps=12,
        max_tokens_per_step=256,
        temperature=0.5
    )
    
    llm_func = RealLLMFunction(model_name="gpt-4")
    rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
    rollout.tools = {
        "calculator": CalculatorTool(),
        "database": DatabaseTool(),
        "filesystem": FileSystemTool(),
        "web_search": WebSearchTool()
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Complex task requiring multiple tools
    prompt = {
        "content": "Research the population of Tokyo, calculate what percentage it is of Japan's total population (125 million), store the result in database, and save a summary to a file called 'tokyo_analysis.txt'",
        "complexity": "high",
        "required_tools": ["web_search", "calculator", "database", "filesystem"]
    }
    
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=llm_func,
        request_id="multi_tool_workflow"
    )
    
    # Analyze tool usage
    tool_usage = {}
    for step in trajectory.get_tool_calls():
        tool = step.tool_name
        tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    print(f"Multi-tool workflow results:")
    print(f"  Total steps: {len(trajectory.steps)}")
    print(f"  Tool usage: {tool_usage}")
    print(f"  Workflow completed: {trajectory.is_completed}")
    print(f"  Final result: {trajectory.get_final_response()[:80]}...")
    
    return trajectory


async def example_3_agent_composition():
    """Example 3: Agent composition and chaining."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Agent Composition")
    print("="*60)
    
    class AgentChain:
        """Chain multiple agents for complex workflows."""
        
        def __init__(self, agents: List[BaseAgent], llm_func):
            self.agents = agents
            self.llm_func = llm_func
            
        async def run_chain(self, initial_prompt: Dict[str, Any]) -> List[Trajectory]:
            """Run a chain of agents."""
            trajectories = []
            current_input = initial_prompt
            
            for i, agent in enumerate(self.agents):
                trajectory = await agent.run_trajectory(
                    prompt=current_input,
                    llm_generate_func=self.llm_func,
                    request_id=f"chain_step_{i}"
                )
                trajectories.append(trajectory)
                
                # Pass result to next agent
                current_input = {
                    "content": f"Previous result: {trajectory.get_final_response()}",
                    "context": f"Chain step {i+1}"
                }
            
            return trajectories
    
    # Create specialized agents
    research_agent = ResearchAgent(max_steps=4, research_depth=2)
    analysis_agent = ReactAgent(max_steps=3)
    
    # Set up tools for each agent
    tools = {
        "web_search": WebSearchTool(),
        "calculator": CalculatorTool()
    }
    
    research_agent.set_tools(tools)
    analysis_agent.set_tools(tools)
    
    # Create agent chain
    llm_func = RealLLMFunction()
    chain = AgentChain([research_agent, analysis_agent], llm_func)
    
    # Run complex chained workflow
    prompt = {
        "content": "Research renewable energy adoption rates globally and analyze the economic impact",
        "type": "research_and_analysis"
    }
    
    trajectories = await chain.run_chain(prompt)
    
    print(f"Agent composition results:")
    print(f"  Chain length: {len(trajectories)}")
    for i, traj in enumerate(trajectories):
        agent_type = "research" if i == 0 else "analysis"
        print(f"  Agent {i+1} ({agent_type}): {len(traj.steps)} steps")
        print(f"    Result: {traj.get_final_response()[:60]}...")
    
    return trajectories


async def example_4_production_monitoring():
    """Example 4: Production-ready monitoring and error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Production Monitoring")
    print("="*60)
    
    class MonitoredAgent(ReactAgent):
        """Agent with built-in monitoring."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metrics = []
            self.error_count = 0
            
        async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
            """Monitored trajectory execution."""
            start_time = time.time()
            
            try:
                trajectory = await super().run_trajectory(
                    prompt, llm_generate_func, request_id, **kwargs
                )
                success = True
                error_msg = None
                
            except Exception as e:
                logger.error(f"Trajectory {request_id} failed: {e}")
                self.error_count += 1
                
                # Create minimal error trajectory
                trajectory = Trajectory(request_id=request_id)
                trajectory.add_step(TrajectoryStep(
                    step_type=StepType.OBSERVATION,
                    content=str(prompt.get('content', prompt))
                ))
                trajectory.add_step(TrajectoryStep(
                    step_type=StepType.FINAL_ANSWER,
                    content=f"I encountered an error: {str(e)}"
                ))
                self.finalize_trajectory(trajectory)
                
                success = False
                error_msg = str(e)
            
            # Record metrics
            metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=time.time(),
                total_steps=len(trajectory.steps),
                tool_calls=len(trajectory.get_tool_calls()),
                tokens_generated=sum(len(step.content.split()) for step in trajectory.steps),
                success=success,
                error_message=error_msg
            )
            
            self.metrics.append(metrics)
            return trajectory
    
    class FailingTool:
        """Tool that fails sometimes for testing error handling."""
        def __init__(self, failure_rate=0.3):
            self.description = "Tool that fails randomly"
            self.failure_rate = failure_rate
            self.call_count = 0
            
        async def execute(self, **kwargs) -> str:
            self.call_count += 1
            if self.call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated tool failure")
            return f"Tool executed successfully (call {self.call_count})"
    
    # Create monitored setup
    agent = MonitoredAgent(max_steps=5)
    agent.set_tools({
        "reliable_tool": CalculatorTool(),
        "failing_tool": FailingTool()
    })
    
    llm_func = RealLLMFunction()
    
    # Run multiple trajectories with some failures
    test_prompts = [
        {"content": "Use the reliable tool to calculate 5+5"},
        {"content": "Use the failing tool for a test"},
        {"content": "Try both tools in sequence"},
        {"content": "Calculate 10*10 using available tools"},
        {"content": "Test the failing tool again"}
    ]
    
    results = []
    for i, prompt in enumerate(test_prompts):
        trajectory = await agent.run_trajectory(
            prompt=prompt,
            llm_generate_func=llm_func,
            request_id=f"monitored_{i}"
        )
        results.append(trajectory)
    
    # Analyze monitoring results
    successful = len([m for m in agent.metrics if m.success])
    total = len(agent.metrics)
    avg_duration = sum(m.duration for m in agent.metrics) / total
    avg_steps = sum(m.total_steps for m in agent.metrics) / total
    
    print(f"Production monitoring results:")
    print(f"  Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"  Average duration: {avg_duration:.2f}s")
    print(f"  Average steps: {avg_steps:.1f}")
    print(f"  Error count: {agent.error_count}")
    
    # Show failed cases
    failed_metrics = [m for m in agent.metrics if not m.success]
    if failed_metrics:
        print(f"  Failed cases:")
        for m in failed_metrics:
            print(f"    Error: {m.error_message}")
    
    return results, agent.metrics


async def example_5_custom_trajectory_analysis():
    """Example 5: Custom trajectory analysis and optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Trajectory Analysis")
    print("="*60)
    
    class TrajectoryAnalyzer:
        """Analyze and optimize trajectories."""
        
        @staticmethod
        def analyze_efficiency(trajectory: Trajectory) -> Dict[str, float]:
            """Analyze trajectory efficiency."""
            total_steps = len(trajectory.steps)
            tool_steps = len(trajectory.get_tool_calls())
            thought_steps = len([s for s in trajectory.steps if s.step_type == StepType.THOUGHT])
            
            return {
                "tool_ratio": tool_steps / total_steps if total_steps > 0 else 0,
                "thought_ratio": thought_steps / total_steps if total_steps > 0 else 0,
                "steps_per_tool": total_steps / tool_steps if tool_steps > 0 else float('inf'),
                "efficiency_score": tool_steps / max(total_steps, 1)  # Higher is better
            }
        
        @staticmethod
        def identify_patterns(trajectories: List[Trajectory]) -> Dict[str, Any]:
            """Identify patterns across trajectories."""
            patterns = {
                "common_step_sequences": [],
                "average_steps": sum(len(t.steps) for t in trajectories) / len(trajectories),
                "success_rate": sum(1 for t in trajectories if t.is_completed) / len(trajectories),
                "most_used_tools": {}
            }
            
            # Tool usage analysis
            tool_usage = {}
            for traj in trajectories:
                for step in traj.get_tool_calls():
                    tool = step.tool_name
                    tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            patterns["most_used_tools"] = dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True))
            
            return patterns
    
    # Generate trajectories for analysis
    config = AgenticRolloutConfig(agent_type="react", max_steps=6)
    llm_func = RealLLMFunction()
    rollout = AgenticRollout(config=config, llm_generate_func=llm_func)
    rollout.tools = {
        "calculator": CalculatorTool(),
        "web_search": WebSearchTool(),
        "code_executor": CodeExecutorTool()
    }
    rollout.agent.set_tools(rollout.tools)
    
    # Different types of prompts
    test_prompts = [
        {"content": "Calculate 15! and explain the result", "type": "math"},
        {"content": "Search for Python tutorials and summarize", "type": "search"},
        {"content": "Write code to sort a list in Python", "type": "coding"},
        {"content": "Find the weather in Paris and convert temperature to Fahrenheit", "type": "mixed"},
        {"content": "Calculate compound interest for $1000 at 5% for 10 years", "type": "math"}
    ]
    
    trajectories = []
    for i, prompt in enumerate(test_prompts):
        trajectory = await rollout.agent.run_trajectory(
            prompt=prompt,
            llm_generate_func=llm_func,
            request_id=f"analysis_{i}"
        )
        trajectories.append(trajectory)
    
    # Analyze trajectories
    analyzer = TrajectoryAnalyzer()
    
    print(f"Trajectory analysis results:")
    print(f"  Analyzed {len(trajectories)} trajectories")
    
    # Individual analysis
    for i, traj in enumerate(trajectories):
        efficiency = analyzer.analyze_efficiency(traj)
        print(f"  Trajectory {i+1}: efficiency={efficiency['efficiency_score']:.2f}, "
              f"tool_ratio={efficiency['tool_ratio']:.2f}")
    
    # Pattern analysis
    patterns = analyzer.identify_patterns(trajectories)
    print(f"  Overall patterns:")
    print(f"    Average steps: {patterns['average_steps']:.1f}")
    print(f"    Success rate: {patterns['success_rate']:.1f}")
    print(f"    Most used tools: {patterns['most_used_tools']}")
    
    # Save analysis results
    analysis_results = {
        "trajectories": [t.to_dict() for t in trajectories],
        "patterns": patterns,
        "individual_analysis": [analyzer.analyze_efficiency(t) for t in trajectories]
    }
    
    # Save to file for further analysis
    with open("/tmp/trajectory_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"  Analysis saved to /tmp/trajectory_analysis.json")
    
    return trajectories, patterns


async def main():
    """Run all advanced integration examples."""
    print("AgenticRollout Library - Advanced Integration Examples")
    print("=" * 60)
    
    examples = [
        ("Real LLM Integration", example_1_real_llm_integration),
        ("Multi-Tool Workflow", example_2_multi_tool_workflow),
        ("Agent Composition", example_3_agent_composition),
        ("Production Monitoring", example_4_production_monitoring),
        ("Trajectory Analysis", example_5_custom_trajectory_analysis)
    ]
    
    results = {}
    for name, example_func in examples:
        print(f"\nRunning: {name}")
        try:
            result = await example_func()
            results[name] = {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # Final summary
    print("\n" + "="*60)
    print("ADVANCED INTEGRATION SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"Completed {successful}/{total} advanced examples")
    for name, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"  {name}: {status}")
        if not result["success"]:
            print(f"    Error: {result['error']}")
    
    print("\nAdvanced AgenticRollout capabilities demonstrated:")
    print("  • Real LLM API integration")
    print("  • Complex multi-tool workflows")
    print("  • Agent composition and chaining")
    print("  • Production monitoring and error handling")
    print("  • Trajectory analysis and optimization")
    print("  • Performance metrics and debugging")
    print("  • Scalable architecture patterns")


if __name__ == "__main__":
    asyncio.run(main())