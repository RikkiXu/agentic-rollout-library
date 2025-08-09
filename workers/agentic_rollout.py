"""
Main AgenticRollout class that integrates with VERL's rollout system.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .core.base_agent import BaseAgent
from .core.registry import get_agent_class
from .core.trajectory import Trajectory

# Optional VERL imports for integration
try:
    from verl.workers.rollout.base import BaseRollout
    from verl.protocol import DataProto
    VERL_AVAILABLE = True
except ImportError:
    # Fallback for independent usage
    BaseRollout = object
    DataProto = None
    VERL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AgenticRolloutConfig:
    """Configuration for AgenticRollout."""
    
    # Agent configuration
    agent_type: str = "react"
    max_steps: int = 10
    max_tokens_per_step: int = 16000
    temperature: float = 0.7
    
    # Tool and interaction configuration
    tools_config: Optional[Dict[str, Any]] = None
    interactions_config: Optional[Dict[str, Any]] = None
    
    # Performance settings
    batch_size: int = 1
    concurrent_requests: int = 4
    
    # Output settings
    include_trajectory_in_output: bool = True
    save_trajectories: bool = False
    trajectory_save_path: Optional[str] = None


class AgenticRollout(BaseRollout):
    """
    Agentic rollout implementation that can use various agents to generate
    multi-step trajectories involving LLM generation and tool calls.
    
    This class integrates with VERL's rollout system by implementing the
    BaseRollout interface while providing advanced agentic capabilities.
    """
    
    def __init__(
        self,
        config: AgenticRolloutConfig,
        llm_generate_func=None,
        tokenizer=None,
        **kwargs
    ):
        """
        Initialize AgenticRollout.
        
        Args:
            config: Configuration for the agentic rollout
            llm_generate_func: Function to call for LLM generation
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.config = config
        self.llm_generate_func = llm_generate_func
        self.tokenizer = tokenizer
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize tools and interactions
        self.tools = {}
        self.interactions = {}
        
        if config.tools_config:
            self._initialize_tools()
        
        if config.interactions_config:
            self._initialize_interactions()
        
        logger.info(f"Initialized AgenticRollout with agent '{config.agent_type}'")
    
    def _create_agent(self) -> BaseAgent:
        """Create the agent instance based on configuration."""
        agent_class = get_agent_class(self.config.agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
        
        agent_config = {
            "max_steps": self.config.max_steps,
            "max_tokens_per_step": self.config.max_tokens_per_step,
            "temperature": self.config.temperature,
            "tools_config": self.config.tools_config,
        }
        
        return agent_class(**agent_config)
    
    def _initialize_tools(self):
        """Initialize tools from configuration."""
        # Import here to avoid circular imports
        from verl.utils.tools_utils import initialize_tools_from_config
        
        try:
            self.tools = initialize_tools_from_config(self.config.tools_config)
            self.agent.set_tools(self.tools)
            logger.info(f"Initialized {len(self.tools)} tools")
        except Exception as e:
            logger.warning(f"Failed to initialize tools: {e}")
            self.tools = {}
    
    def _initialize_interactions(self):
        """Initialize interactions from configuration."""
        # Import here to avoid circular imports
        from verl.utils.interaction_utils import initialize_interactions_from_config
        
        try:
            self.interactions = initialize_interactions_from_config(self.config.interactions_config)
            self.agent.set_interactions(self.interactions)
            logger.info(f"Initialized {len(self.interactions)} interactions")
        except Exception as e:
            logger.warning(f"Failed to initialize interactions: {e}")
            self.interactions = {}
    
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences using agentic rollouts.
        
        Args:
            prompts: Input prompts in DataProto format
            
        Returns:
            Generated sequences in DataProto format
        """
        batch_size = len(prompts.batch.get("input_ids", []))
        logger.info(f"Starting agentic rollout for batch of {batch_size} prompts")
        
        # Extract prompts from DataProto
        prompt_list = self._extract_prompts_from_data_proto(prompts)
        
        # Generate trajectories
        trajectories = await self._generate_trajectories_batch(prompt_list)
        
        # Convert trajectories back to DataProto format
        output_data = self._trajectories_to_data_proto(trajectories, prompts)
        
        logger.info(f"Completed agentic rollout for {len(trajectories)} trajectories")
        return output_data
    
    def _extract_prompts_from_data_proto(self, prompts: DataProto) -> List[Dict[str, Any]]:
        """Extract individual prompts from DataProto."""
        prompt_list = []
        
        # Get batch size
        if "input_ids" in prompts.batch:
            batch_size = len(prompts.batch["input_ids"])
        else:
            batch_size = len(prompts.non_tensor_batch.get("raw_prompt", []))
        
        for i in range(batch_size):
            prompt_data = {}
            
            # Extract tensor data
            for key, tensor_list in prompts.batch.items():
                if isinstance(tensor_list, list) and len(tensor_list) > i:
                    prompt_data[key] = tensor_list[i]
            
            # Extract non-tensor data
            for key, value_list in prompts.non_tensor_batch.items():
                if isinstance(value_list, list) and len(value_list) > i:
                    prompt_data[key] = value_list[i]
                elif not isinstance(value_list, list):
                    prompt_data[key] = value_list
            
            # Extract meta info
            prompt_data.update(prompts.meta_info)
            
            prompt_list.append(prompt_data)
        
        return prompt_list
    
    async def _generate_trajectories_batch(self, prompt_list: List[Dict[str, Any]]) -> List[Trajectory]:
        """Generate trajectories for a batch of prompts."""
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def generate_single_trajectory(prompt_data: Dict[str, Any], index: int) -> Trajectory:
            async with semaphore:
                request_id = f"agentic_rollout_{index}"
                return await self.agent.run_trajectory(
                    prompt=prompt_data,
                    llm_generate_func=self._llm_generate_wrapper,
                    request_id=request_id
                )
        
        # Generate all trajectories concurrently
        tasks = [
            generate_single_trajectory(prompt_data, i)
            for i, prompt_data in enumerate(prompt_list)
        ]
        
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(trajectories):
            if isinstance(result, Exception):
                logger.error(f"Error generating trajectory {i}: {result}")
                # Create a minimal trajectory with error
                trajectories[i] = Trajectory(
                    request_id=f"agentic_rollout_{i}_error",
                    metadata={"error": str(result)}
                )
        
        return trajectories
    
    async def _llm_generate_wrapper(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Wrapper for LLM generation that adapts to VERL's generation interface.
        """
        if self.llm_generate_func is None:
            raise ValueError("LLM generation function not provided")
        
        # Set defaults
        if max_tokens is None:
            max_tokens = self.config.max_tokens_per_step
        if temperature is None:
            temperature = self.config.temperature
        
        # Call the LLM generation function
        # This will depend on the specific implementation in VERL
        try:
            if asyncio.iscoroutinefunction(self.llm_generate_func):
                response = await self.llm_generate_func(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                response = self.llm_generate_func(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            
            # Extract text from response (format may vary)
            if isinstance(response, str):
                return response
            elif isinstance(response, dict) and "content" in response:
                return response["content"]
            elif hasattr(response, "text"):
                return response.text
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _trajectories_to_data_proto(
        self,
        trajectories: List[Trajectory],
        original_prompts: DataProto
    ) -> DataProto:
        """Convert trajectories back to DataProto format."""
        batch_data = {}
        non_tensor_data = {}
        
        # Extract responses and other data from trajectories
        responses = []
        response_masks = []
        final_rewards = []
        trajectory_data = []
        
        for trajectory in trajectories:
            # Get final response
            final_response = trajectory.get_final_response()
            responses.append(final_response)
            
            # Create response mask (1 for generated tokens, 0 for tool responses)
            # For now, we'll mark everything as generated
            response_length = len(final_response.split())  # Rough token count
            response_masks.append([1] * response_length)
            
            # Get final reward
            final_rewards.append(trajectory.final_reward or 0.0)
            
            # Include trajectory data if requested
            if self.config.include_trajectory_in_output:
                trajectory_data.append(trajectory.to_dict())
        
        # Prepare batch data
        batch_data["responses"] = responses
        batch_data["response_mask"] = response_masks
        
        # Copy relevant data from original prompts
        for key, value in original_prompts.batch.items():
            if key not in batch_data:
                batch_data[key] = value
        
        # Prepare non-tensor data
        non_tensor_data["final_rewards"] = final_rewards
        
        if self.config.include_trajectory_in_output:
            non_tensor_data["trajectories"] = trajectory_data
        
        # Copy relevant non-tensor data from original prompts
        for key, value in original_prompts.non_tensor_batch.items():
            if key not in non_tensor_data:
                non_tensor_data[key] = value
        
        # Create output DataProto
        output_data = DataProto(
            batch=batch_data,
            non_tensor_batch=non_tensor_data,
            meta_info=original_prompts.meta_info.copy()
        )
        
        return output_data
    
    def save_trajectories(self, trajectories: List[Trajectory]):
        """Save trajectories to disk if configured."""
        if not self.config.save_trajectories or not self.config.trajectory_save_path:
            return
        
        import json
        import os
        from datetime import datetime
        
        try:
            # Create save directory if it doesn't exist
            os.makedirs(self.config.trajectory_save_path, exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_{timestamp}.json"
            filepath = os.path.join(self.config.trajectory_save_path, filename)
            
            # Convert trajectories to serializable format
            data = [trajectory.to_dict() for trajectory in trajectories]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(trajectories)} trajectories to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save trajectories: {e}")


# Helper function for easy instantiation
def create_agentic_rollout(config_dict: Dict[str, Any], **kwargs) -> AgenticRollout:
    """
    Create an AgenticRollout instance from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        **kwargs: Additional arguments for AgenticRollout
        
    Returns:
        AgenticRollout instance
    """
    config = AgenticRolloutConfig(**config_dict)
    return AgenticRollout(config=config, **kwargs)