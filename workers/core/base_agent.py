"""
Base agent class for agentic rollouts.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging

from .trajectory import Trajectory, TrajectoryStep, StepType


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in AgenticRolloutLib.
    
    An agent is responsible for generating a complete trajectory by:
    1. Taking initial observations/prompts
    2. Reasoning about the problem
    3. Deciding on actions/tool calls
    4. Processing action results
    5. Generating final responses
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        max_tokens_per_step: int = 512,
        temperature: float = 0.7,
        tools_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the agent.
        
        Args:
            max_steps: Maximum number of steps in a trajectory
            max_tokens_per_step: Maximum tokens to generate per step
            temperature: Sampling temperature for generation
            tools_config: Configuration for tools
            **kwargs: Additional agent-specific configuration
        """
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.tools_config = tools_config or {}
        self.config = kwargs
        
        # Tools and interactions will be initialized by the rollout
        self.tools = {}
        self.interactions = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with max_steps={max_steps}")
    
    def set_tools(self, tools: Dict[str, Any]) -> None:
        """Set the tools available to this agent."""
        self.tools = tools
        logger.info(f"Agent {self.__class__.__name__} configured with {len(tools)} tools")
    
    def set_interactions(self, interactions: Dict[str, Any]) -> None:
        """Set the interactions available to this agent."""
        self.interactions = interactions
        logger.info(f"Agent {self.__class__.__name__} configured with {len(interactions)} interactions")
    
    @abstractmethod
    async def run_trajectory(
        self,
        prompt: Dict[str, Any],
        llm_generate_func,
        request_id: str,
        **kwargs
    ) -> Trajectory:
        """
        Run a complete trajectory for the given prompt.
        
        Args:
            prompt: The initial prompt/observation
            llm_generate_func: Function to call LLM for generation
            request_id: Unique identifier for this request
            **kwargs: Additional arguments
            
        Returns:
            Complete trajectory with all steps
        """
        pass
    
    def should_continue(self, trajectory: Trajectory) -> bool:
        """
        Determine if the trajectory should continue.
        
        Args:
            trajectory: Current trajectory
            
        Returns:
            True if should continue, False otherwise
        """
        # Basic termination conditions
        if len(trajectory.steps) >= self.max_steps:
            logger.info(f"Trajectory {trajectory.request_id} reached max steps ({self.max_steps})")
            return False
        
        if trajectory.is_completed:
            logger.info(f"Trajectory {trajectory.request_id} marked as completed")
            return False
        
        # Check if last step is a final answer
        if trajectory.steps and trajectory.steps[-1].step_type == StepType.FINAL_ANSWER:
            logger.info(f"Trajectory {trajectory.request_id} has final answer")
            return False
        
        return True
    
    def create_system_prompt(self) -> str:
        """
        Create system prompt for the agent.
        This can be overridden by subclasses to customize behavior.
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that can use tools to solve problems.
Think step by step and use the available tools when necessary."""
    
    def parse_llm_output(self, output: str, step_type: StepType = StepType.THOUGHT) -> TrajectoryStep:
        """
        Parse LLM output into a trajectory step.
        This can be overridden by subclasses for custom parsing logic.
        
        Args:
            output: Raw LLM output
            step_type: Type of step to create
            
        Returns:
            Parsed trajectory step
        """
        return TrajectoryStep(
            step_type=step_type,
            content=output.strip(),
            metadata={"raw_output": output}
        )
    
    async def execute_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        trajectory: Trajectory
    ) -> TrajectoryStep:
        """
        Execute a tool call and return the result step.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            trajectory: Current trajectory
            
        Returns:
            Step containing tool execution result
        """
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}")
            
            tool = self.tools[tool_name]
            
            # Execute the tool
            if hasattr(tool, 'execute_tool'):
                # This is a BaseAgenticTool - use the new interface
                import uuid
                instance_id = str(uuid.uuid4())
                tool_result = await tool.execute_tool(instance_id, tool_args)
                result = tool_result.result if tool_result.success else f"Tool error: {tool_result.error}"
            else:
                # Legacy tool interface
                result = await tool.execute(**tool_args)
            
            # Create result step
            result_step = TrajectoryStep(
                step_type=StepType.ACTION_RESULT,
                content=str(result),
                metadata={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "execution_successful": True
                },
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=result
            )
            
            logger.info(f"Tool {tool_name} executed successfully for trajectory {trajectory.request_id}")
            return result_step
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            error_step = TrajectoryStep(
                step_type=StepType.ACTION_RESULT,
                content=f"Error executing tool {tool_name}: {str(e)}",
                metadata={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "execution_successful": False,
                    "error": str(e)
                },
                tool_name=tool_name,
                tool_args=tool_args
            )
            return error_step
    
    def format_messages_for_llm(self, trajectory: Trajectory, additional_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format trajectory into messages for LLM input.
        
        Args:
            trajectory: Current trajectory
            additional_message: Additional message to append
            
        Returns:
            List of messages in chat format
        """
        messages = []
        
        # Add system prompt
        system_prompt = self.create_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            # Store system prompt in trajectory metadata for later use
            if not hasattr(trajectory, 'system_prompt'):
                trajectory.system_prompt = system_prompt
        
        # Add trajectory steps as messages
        messages.extend(trajectory.get_messages())
        
        # Add additional message if provided
        if additional_message:
            messages.append({"role": "user", "content": additional_message})
        
        return messages
    
    def calculate_step_reward(self, step: TrajectoryStep, trajectory: Trajectory) -> float:
        """
        Calculate reward for a specific step.
        This can be overridden by subclasses for custom reward logic.
        
        Args:
            step: The step to calculate reward for
            trajectory: Current trajectory
            
        Returns:
            Reward score for the step
        """
        # Default: no reward unless specified
        return step.reward_score or 0.0
    
    def finalize_trajectory(self, trajectory: Trajectory) -> None:
        """
        Finalize the trajectory by calculating final rewards and metadata.
        
        Args:
            trajectory: Trajectory to finalize
        """
        trajectory.is_completed = True
        trajectory.final_reward = trajectory.get_total_reward()
        trajectory.total_tokens = sum(
            len(step.content.split()) for step in trajectory.steps
        )
        
        logger.info(
            f"Finalized trajectory {trajectory.request_id}: "
            f"{len(trajectory.steps)} steps, "
            f"reward={trajectory.final_reward:.3f}, "
            f"tokens≈{trajectory.total_tokens}"
        )