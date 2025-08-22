"""
ReAct (Reasoning + Acting) agent implementation.
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..core.base_agent import BaseAgent
from ..core.registry import register_agent
from ..core.trajectory import Trajectory, TrajectoryStep, StepType

logger = logging.getLogger(__name__)


@register_agent("react")
class ReactAgent(BaseAgent):
    """
    ReAct agent that alternates between reasoning (Thought) and acting (Action).
    
    The agent follows the ReAct pattern:
    1. Thought: Reason about the current situation
    2. Action: Take an action (tool call or final answer)
    3. Observation: Receive action result
    4. Repeat until problem is solved
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ReAct specific configuration
        self.require_thought_before_action = kwargs.get("require_thought_before_action", True)
        self.max_consecutive_thoughts = kwargs.get("max_consecutive_thoughts", 3)
        self.enable_self_reflection = kwargs.get("enable_self_reflection", True)
    
    def create_system_prompt(self) -> str:
        """Create ReAct-specific system prompt."""
        tools_description = ""
        if self.tools:
            tools_list = []
            for tool_name, tool in self.tools.items():
                # Get tool description from tool schema or class
                description = getattr(tool, 'description', f"Tool: {tool_name}")
                tools_list.append(f"- {tool_name}: {description}")
            tools_description = f"\n\nAvailable tools:\n" + "\n".join(tools_list)
        
        return f"""You are a helpful AI assistant that uses the ReAct (Reasoning + Acting) framework to solve problems.

Follow this pattern:
1. Thought: Think about what you need to do next
2. Action: Take an action using available tools or provide a final answer
3. Observation: Receive the result of your action
4. Repeat until you can provide a final answer

When taking actions:
- Use "Action: tool_name(arg1=value1, arg2=value2)" for tool calls
- Use "Action: Final Answer: [your answer]" when you're ready to provide the final response

Always think step by step and explain your reasoning.{tools_description}"""
    
    async def run_trajectory(
        self,
        prompt: Dict[str, Any],
        llm_generate_func,
        request_id: str,
        **kwargs
    ) -> Trajectory:
        """Run a complete ReAct trajectory."""
        trajectory = Trajectory(request_id=request_id)
        
        # Add initial observation
        initial_content = self._extract_prompt_content(prompt)
        initial_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=initial_content,
            metadata={"prompt": prompt}
        )
        trajectory.add_step(initial_step)
        
        consecutive_thoughts = 0
        
        while self.should_continue(trajectory):
            try:
                # Generate next step
                messages = self.format_messages_for_llm(trajectory)
                
                # Determine what kind of step we expect next
                expected_step_type = self._determine_next_step_type(trajectory, consecutive_thoughts)
                
                # Add guidance for the expected step type
                if expected_step_type == StepType.THOUGHT:
                    messages.append({"role": "user", "content": "Think: What should you do next?"})
                elif expected_step_type == StepType.ACTION:
                    messages.append({"role": "user", "content": "Action: Take your next action."})
                
                # Generate response
                response = await llm_generate_func(
                    messages,
                    max_tokens=self.max_tokens_per_step,
                    temperature=self.temperature
                )
                
                # Parse the response
                step = self._parse_react_output(response, expected_step_type)
                trajectory.add_step(step)
                
                # Handle the step based on its type
                if step.step_type == StepType.THOUGHT:
                    consecutive_thoughts += 1
                else:
                    consecutive_thoughts = 0
                    
                    if step.step_type == StepType.ACTION:
                        # Execute action
                        await self._handle_action(step, trajectory, llm_generate_func)
                        
                    elif step.step_type == StepType.FINAL_ANSWER:
                        # Mark trajectory as completed
                        trajectory.is_completed = True
                        break
                
            except Exception as e:
                logger.error(f"Error in trajectory {request_id}: {e}")
                error_step = TrajectoryStep(
                    step_type=StepType.THOUGHT,
                    content=f"I encountered an error: {str(e)}. Let me try a different approach.",
                    metadata={"error": str(e)}
                )
                trajectory.add_step(error_step)
        
        self.finalize_trajectory(trajectory)
        return trajectory
    
    def _extract_prompt_content(self, prompt: Dict[str, Any]) -> str:
        """Extract content from prompt data structure."""
        if isinstance(prompt, str):
            return prompt
        
        # Handle different prompt formats
        if "content" in prompt:
            return prompt["content"]
        elif "prompt" in prompt:
            return prompt["prompt"]
        elif "messages" in prompt and prompt["messages"]:
            # Extract from last user message
            for msg in reversed(prompt["messages"]):
                if msg.get("role") == "user":
                    return msg.get("content", "")
        
        return str(prompt)
    
    def _determine_next_step_type(self, trajectory: Trajectory, consecutive_thoughts: int) -> StepType:
        """Determine what type of step should come next."""
        if not trajectory.steps:
            return StepType.THOUGHT
        
        last_step = trajectory.steps[-1]
        
        # After observation, we usually want thought
        if last_step.step_type == StepType.OBSERVATION:
            return StepType.THOUGHT
        
        # After action result, we want thought
        if last_step.step_type == StepType.ACTION_RESULT:
            return StepType.THOUGHT
        
        # After thought, we want action (unless too many consecutive thoughts)
        if last_step.step_type == StepType.THOUGHT:
            if consecutive_thoughts >= self.max_consecutive_thoughts:
                return StepType.ACTION
            # Allow either thought or action
            return StepType.THOUGHT
        
        # After action, we expect observation (handled by action execution)
        return StepType.THOUGHT
    
    def _parse_react_output(self, output: str, expected_type: StepType = None) -> TrajectoryStep:
        """Parse LLM output into ReAct format."""
        output = output.strip()
        
        # Check for explicit markers
        if output.startswith("Thought:"):
            content = output[8:].strip()
            return TrajectoryStep(
                step_type=StepType.THOUGHT,
                content=content,
                metadata={"raw_output": output}
            )
        
        elif output.startswith("Action:"):
            action_content = output[7:].strip()
            
            # Check if it's a final answer
            if action_content.lower().startswith("final answer:"):
                final_content = action_content[13:].strip()
                return TrajectoryStep(
                    step_type=StepType.FINAL_ANSWER,
                    content=final_content,
                    metadata={"raw_output": output}
                )
            
            # Parse tool call
            tool_name, tool_args = self._parse_tool_call(action_content)
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=action_content,
                metadata={"raw_output": output},
                tool_name=tool_name,
                tool_args=tool_args
            )
        
        # If no explicit marker, infer from expected type or content
        if "final answer" in output.lower() or "answer:" in output.lower():
            return TrajectoryStep(
                step_type=StepType.FINAL_ANSWER,
                content=output,
                metadata={"raw_output": output, "inferred_type": True}
            )
        
        # Check if it looks like a tool call
        if self._looks_like_tool_call(output):
            tool_name, tool_args = self._parse_tool_call(output)
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=output,
                metadata={"raw_output": output, "inferred_type": True},
                tool_name=tool_name,
                tool_args=tool_args
            )
        
        # Default to thought
        step_type = expected_type if expected_type else StepType.THOUGHT
        return TrajectoryStep(
            step_type=step_type,
            content=output,
            metadata={"raw_output": output, "inferred_type": True}
        )
    
    def _looks_like_tool_call(self, text: str) -> bool:
        """Check if text looks like a tool call."""
        # Look for patterns like: tool_name(arg=value)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'
        return bool(re.search(pattern, text))
    
    def _parse_tool_call(self, action_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse tool call from action text."""
        try:
            # Pattern: tool_name(arg1=value1, arg2=value2)
            match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', action_text.strip())
            if not match:
                return None, None
            
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            if not args_str:
                return tool_name, {}
            
            # Parse arguments
            args = {}
            
            # Try to parse as key=value pairs
            for arg_pair in args_str.split(','):
                arg_pair = arg_pair.strip()
                if '=' in arg_pair:
                    key, value = arg_pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # Try to convert to appropriate type
                    try:
                        # Try JSON parsing for complex types
                        args[key] = json.loads(value)
                    except:
                        # Keep as string
                        args[key] = value
                else:
                    # Positional argument
                    args[f"arg_{len(args)}"] = arg_pair
            
            return tool_name, args
            
        except Exception as e:
            logger.warning(f"Failed to parse tool call '{action_text}': {e}")
            return None, {"raw_text": action_text}
    
    async def _handle_action(self, action_step: TrajectoryStep, trajectory: Trajectory, llm_generate_func):
        """Handle action step execution."""
        if action_step.tool_name and action_step.tool_name in self.tools:
            # Execute tool
            result_step = await self.execute_tool_call(
                action_step.tool_name,
                action_step.tool_args or {},
                trajectory
            )
            trajectory.add_step(result_step)
            
        else:
            # No valid tool call found, add observation about this
            observation_step = TrajectoryStep(
                step_type=StepType.ACTION_RESULT,
                content=f"I cannot execute the action '{action_step.content}'. Available tools: {list(self.tools.keys())}",
                metadata={"action_failed": True}
            )
            trajectory.add_step(observation_step)