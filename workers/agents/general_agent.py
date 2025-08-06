#!/usr/bin/env python3
"""
General ReAct agent implementation with configurable tools and termination.
This agent can be configured to use different tools, system prompts, and termination conditions.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from ..core.base_agent import BaseAgent
from ..core.registry import register_agent
from ..core.trajectory import Trajectory, TrajectoryStep, StepType
from ..core.profiler import RolloutProfiler, EventType, get_profiler

logger = logging.getLogger(__name__)

try:
    from json_repair import repair_json
    REPAIR_JSON_AVAILABLE = True
except ImportError:
    REPAIR_JSON_AVAILABLE = False
    logger.warning("json-repair not available. Install with: pip install json-repair")


@register_agent("general")
class GeneralAgent(BaseAgent):
    """
    General purpose ReAct agent with configurable tools and behavior.
    
    This agent supports:
    - Configurable system prompts
    - Configurable tool lists  
    - Configurable maximum rounds/steps
    - Configurable termination tool names
    - Trajectory recording and dumping
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        max_rounds: int = 10,
        termination_tool_names: Optional[List[str]] = None,
        debug: bool = False,
        action_parser: Optional[callable] = None,
        steps_remaining_template: Optional[str] = None,
        show_steps_remaining: bool = True,
        profiler: Optional[RolloutProfiler] = None,
        **kwargs
    ):
        """
        Initialize the General Agent.
        
        Args:
            system_prompt: Custom system prompt to use
            tool_names: List of tool names to register for this agent
            max_rounds: Maximum number of reasoning rounds
            termination_tool_names: List of tool names that signal termination
            debug: Enable debug mode to log all LLM inputs and outputs
            action_parser: Custom function to parse actions from LLM output
            steps_remaining_template: Template for steps remaining message. 
                                     Use {steps} placeholder for step count.
                                     Default: "\n\nSteps Remaining: {steps}"
            show_steps_remaining: Whether to show steps remaining in user messages
            profiler: Optional profiler instance for performance tracking
            **kwargs: Additional configuration passed to BaseAgent
        """
        super().__init__(max_steps=max_rounds * 2, **kwargs)  # *2 for thought+action pairs
        
        self.custom_system_prompt = system_prompt
        self.tool_names = tool_names or []
        self.max_rounds = max_rounds
        self.termination_tool_names = termination_tool_names or ["finish"]
        self.debug = debug
        self.custom_action_parser = action_parser
        
        # Steps remaining configuration
        self.show_steps_remaining = show_steps_remaining
        self.steps_remaining_template = steps_remaining_template or "\n\nSteps Remaining: {steps}"
        
        # ReAct specific configuration
        self.require_thought_before_action = kwargs.get("require_thought_before_action", True)
        self.max_consecutive_thoughts = kwargs.get("max_consecutive_thoughts", 3)
        
        # Performance profiler
        self.profiler = profiler
        
        logger.info(f"Initialized GeneralAgent with max_rounds={max_rounds}")
    
    def create_system_prompt(self) -> str:
        """
        Create comprehensive system prompt.
        
        If custom_system_prompt is provided, it will be returned as-is.
        Users can access tool descriptions via self.tools['tool_name'].get_description()
        in their custom prompts.
        
        Otherwise, creates a default ReAct prompt with tool descriptions embedded.
        """
        # If custom system prompt is provided, return it directly
        if self.custom_system_prompt:
            prompt = self.custom_system_prompt
            
            # Log the final prompt if debug is enabled
            if self.debug:
                logger.info("="*80)
                logger.info("🔍 [DEBUG] Final System Prompt:")
                logger.info("-"*80)
                logger.info(prompt)
                logger.info("="*80)
            
            return prompt
        
        # Build tools documentation for default prompt
        tools_documentation = self._build_tools_documentation()
        
        # Default comprehensive system prompt with JSON Action format
        default_prompt = f"""# ReAct Agent with JSON-Structured Actions

You are an intelligent AI assistant that uses the **ReAct** (Reasoning + Acting) framework to solve problems systematically.

## Output Format Requirements

Your response MUST contain both **Thought** and **Action** sections in this exact format:

```
Thought: [Your reasoning about what to do next, analysis of the situation, planning your approach]

Action: 
{{
  "name": "function_name",
  "parameters": {{
    "param1": "value1",
    "param2": 42,
    "param3": true
  }}
}}
```

## Critical Format Rules

1. **Always include both Thought and Action** in a single response
2. **Thought section**: Free-form reasoning in natural language
3. **Action section**: Valid JSON object with "tool_name" and "parameters" fields
4. **JSON must be valid** - use proper quotes, brackets, and data types
5. **Use exact tool names** from the schemas below
6. **Follow parameter types** as specified in the schemas

## Process Flow

1. **Thought**: Analyze the situation and plan your next action
2. **Action**: Execute a tool call using valid JSON format
3. **Observation**: Review the tool result (provided automatically)
4. **Repeat**: Continue until task completion

## Guidelines

- **Think systematically** - break complex problems into steps
- **Use precise tool names** and parameter names from schemas
- **Handle errors gracefully** - adapt if a tool call fails
- **Stay focused** on the user's goal
- **Complete with finish tool** when task is done

{tools_documentation}

## Example Flow

```
Thought: I need to get system information to help the user. Let me start by checking the current directory.

Action:
{{
  "name": "bash_executor", 
  "parameters": {{
    "command": "pwd && ls -la"
  }}
}}
```

After receiving the observation, continue with the next thought and action until completion.

**Remember**: Always output valid JSON in the Action section. The JSON will be parsed programmatically."""
        
        # Log the final prompt if debug is enabled
        if self.debug:
            logger.info("="*80)
            logger.info("🔍 [DEBUG] Final System Prompt (Default):")
            logger.info("-"*80)
            logger.info(default_prompt)
            logger.info("="*80)
            
        return default_prompt
    
    def _add_steps_remaining(self, messages: List[Dict[str, str]], current_round: int) -> List[Dict[str, str]]:
        """
        Add steps remaining information to the last user message.
        
        Args:
            messages: List of messages
            current_round: Current round number (0-indexed)
            
        Returns:
            Modified messages list with steps remaining added
        """
        if not messages:
            return messages
        
        # Calculate remaining steps
        remaining_steps = self.max_rounds - current_round
        
        # Find the last user message and add steps remaining
        messages_copy = messages.copy()
        for i in range(len(messages_copy) - 1, -1, -1):
            if messages_copy[i].get("role") == "user":
                # Add steps remaining to the last user message
                original_content = messages_copy[i]["content"]
                steps_info = self.steps_remaining_template.format(steps=remaining_steps)
                messages_copy[i] = {
                    "role": "user",
                    "content": original_content + steps_info
                }
                break
        
        return messages_copy
    
    def _build_tools_documentation(self) -> str:
        """Build tools documentation using tool descriptions."""
        if not self.tools:
            return ""
        
        tools_list = []
        
        for tool_name, tool in self.tools.items():
            try:
                # Use tool's get_description method if available
                if hasattr(tool, 'get_description'):
                    tool_desc = tool.get_description()
                    tools_list.append(f"""## Tool: {tool_name}

{tool_desc}""")
                else:
                    # Fallback to description attribute or default
                    description = getattr(tool, 'description', f"Tool: {tool_name}")
                    tools_list.append(f"""## Tool: {tool_name}

**Description:** {description}""")
                    
            except Exception as e:
                logger.warning(f"Error building documentation for tool {tool_name}: {e}")
                # Minimal fallback
                tools_list.append(f"""## Tool: {tool_name}

**Status:** Description unavailable""")
        
        if tools_list:
            return "\n# Available Tools\n\n" + "\n\n".join(tools_list)
        return ""
    
    async def run_trajectory(
        self,
        prompt: Union[str, Dict[str, Any]],
        llm_generate_func,
        request_id: str,
        **kwargs
    ) -> Trajectory:
        """Run a complete ReAct trajectory."""
        # Get profiler instance (use provided or global)
        profiler = self.profiler or get_profiler()
        
        # Start trajectory profiling
        trajectory_event_id = None
        if profiler.enabled:
            trajectory_event_id = profiler.start_event(
                f"trajectory_{request_id}",
                EventType.TRAJECTORY_STEP,
                {"request_id": request_id}
            )
        
        trajectory = Trajectory(request_id=request_id)
        
        # Initialize trajectory metadata
        # Try to get model name from llm_generate_func if it's an LLMAPIClient method
        model_name = "unknown"
        if hasattr(llm_generate_func, "__self__") and hasattr(llm_generate_func.__self__, "model"):
            model_name = llm_generate_func.__self__.model
        elif "model_name" in kwargs:
            model_name = kwargs["model_name"]
        
        trajectory.metadata = {
            "start_time": datetime.now().isoformat(),
            "model": model_name,
            "max_rounds": self.max_rounds,
            "assistant_timings": [],  # List of timing for each assistant response
            "tool_calls": [],  # List of tool names called
            "message_lengths": [],  # List of message lengths
            "total_tool_calls": 0,
            "error_count": 0
        }
        
        # Add initial observation
        initial_content = self._extract_prompt_content(prompt)
        initial_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=initial_content,  # No "Task:" prefix
            metadata={"prompt": prompt}
        )
        trajectory.add_step(initial_step)
        trajectory.metadata["message_lengths"].append({
            "role": "user",
            "length": len(initial_content),
            "step_type": "observation"
        })
        
        consecutive_thoughts = 0
        round_count = 0
        
        while self.should_continue(trajectory) and round_count < self.max_rounds:
            try:
                # Generate next step
                messages = self.format_messages_for_llm(trajectory)
                
                # Add steps remaining to the last user message if enabled
                if self.show_steps_remaining:
                    messages = self._add_steps_remaining(messages, round_count)
                
                # Determine what kind of step we expect next
                expected_step_type = self._determine_next_step_type(trajectory, consecutive_thoughts)
                
                # Debug: Log LLM input
                if self.debug:
                    self._log_llm_input(messages, round_count + 1)
                
                # Time the LLM generation
                start_time = time.time()
                
                # Generate response with profiling
                async with profiler.profile_async(
                    f"llm_call_round_{round_count + 1}",
                    EventType.LLM_CALL,
                    {
                        "round": round_count + 1,
                        "model": trajectory.metadata.get("model", "unknown"),
                        "expected_step_type": expected_step_type.value if expected_step_type else None
                    }
                ):
                    response = await llm_generate_func(
                        messages,
                        max_tokens=self.max_tokens_per_step,
                        temperature=self.temperature
                    )
                
                # Record timing
                generation_time = time.time() - start_time
                trajectory.metadata["assistant_timings"].append({
                    "round": round_count + 1,
                    "generation_time_seconds": round(generation_time, 3),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Debug: Log LLM output
                if self.debug:
                    self._log_llm_output(response, round_count + 1)
                
                # Parse the response (may contain multiple steps)
                with profiler.profile(
                    f"action_parsing_round_{round_count + 1}",
                    EventType.ACTION_PARSING,
                    {"round": round_count + 1}
                ):
                    parsed_steps = self._parse_react_response(response, expected_step_type)
                
                # Process each parsed step
                for step in parsed_steps:
                    trajectory.add_step(step)
                    
                    # Record message length for assistant messages
                    if step.step_type in [StepType.THOUGHT, StepType.ACTION, StepType.FINAL_ANSWER]:
                        trajectory.metadata["message_lengths"].append({
                            "role": "assistant",
                            "length": len(step.content),
                            "step_type": step.step_type.value
                        })
                    
                    # Handle the step based on its type
                    if step.step_type == StepType.THOUGHT:
                        consecutive_thoughts += 1
                    else:
                        consecutive_thoughts = 0
                        
                        if step.step_type == StepType.ACTION:
                            # Record tool call
                            if step.tool_name:
                                trajectory.metadata["tool_calls"].append(step.tool_name)
                                trajectory.metadata["total_tool_calls"] += 1
                            
                            # Execute action
                            result_step = await self._handle_action(step, trajectory)
                            if result_step:
                                trajectory.add_step(result_step)
                                
                                # Record observation length
                                trajectory.metadata["message_lengths"].append({
                                    "role": "user",
                                    "length": len(result_step.content),
                                    "step_type": "action_result"
                                })
                                
                                # Check if this was a termination tool
                                if (step.tool_name and 
                                    step.tool_name in self.termination_tool_names):
                                    trajectory.is_completed = True
                                    break
                            
                            round_count += 1
                            
                        elif step.step_type == StepType.FINAL_ANSWER:
                            # Mark trajectory as completed
                            trajectory.is_completed = True
                            break
                    
                    # Break if trajectory is completed
                    if trajectory.is_completed:
                        break
                
            except Exception as e:
                logger.error(f"Error in trajectory {request_id}: {e}")
                trajectory.metadata["error_count"] += 1
                error_step = TrajectoryStep(
                    step_type=StepType.THOUGHT,
                    content=f"I encountered an error: {str(e)}. Let me try a different approach.",
                    metadata={"error": str(e)}
                )
                trajectory.add_step(error_step)
        
        # Finalize metadata
        trajectory.metadata["end_time"] = datetime.now().isoformat()
        trajectory.metadata["total_rounds"] = round_count
        trajectory.metadata["total_steps"] = len(trajectory.steps)
        
        # Calculate total execution time
        start_time = datetime.fromisoformat(trajectory.metadata["start_time"])
        end_time = datetime.fromisoformat(trajectory.metadata["end_time"])
        trajectory.metadata["total_execution_time_seconds"] = round((end_time - start_time).total_seconds(), 3)
        
        # Calculate average message lengths
        if trajectory.metadata["message_lengths"]:
            user_lengths = [m["length"] for m in trajectory.metadata["message_lengths"] if m["role"] == "user"]
            assistant_lengths = [m["length"] for m in trajectory.metadata["message_lengths"] if m["role"] == "assistant"]
            
            trajectory.metadata["average_user_message_length"] = round(sum(user_lengths) / len(user_lengths), 2) if user_lengths else 0
            trajectory.metadata["average_assistant_message_length"] = round(sum(assistant_lengths) / len(assistant_lengths), 2) if assistant_lengths else 0
            trajectory.metadata["total_user_chars"] = sum(user_lengths)
            trajectory.metadata["total_assistant_chars"] = sum(assistant_lengths)
        
        # Calculate average assistant response time
        if trajectory.metadata["assistant_timings"]:
            avg_time = sum(t["generation_time_seconds"] for t in trajectory.metadata["assistant_timings"]) / len(trajectory.metadata["assistant_timings"])
            trajectory.metadata["average_assistant_response_time_seconds"] = round(avg_time, 3)
        
        self.finalize_trajectory(trajectory)
        
        # End trajectory profiling
        if trajectory_event_id:
            try:
                profiler.end_event(trajectory_event_id)
            except Exception as e:
                logger.warning(f"Error ending profiler event: {e}")
        
        # Add profiler data to trajectory metadata if enabled
        if profiler.enabled and profiler.events:
            try:
                # Add timeout protection for profiler summary generation
                trajectory.metadata["profiler_summary"] = profiler.get_summary()
            except Exception as e:
                logger.warning(f"Error getting profiler summary: {e}")
                trajectory.metadata["profiler_summary"] = {"error": str(e)}
        
        return trajectory
    
    def _extract_prompt_content(self, prompt: Union[str, Dict[str, Any]]) -> str:
        """Extract content from prompt data structure."""
        if isinstance(prompt, str):
            return prompt
        
        # Handle different prompt formats
        if isinstance(prompt, dict):
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
    
    def _parse_react_response(self, output: str, expected_type: StepType = None) -> List[TrajectoryStep]:
        """
        Parse LLM response into a list of trajectory steps.
        
        Returns:
            List of TrajectoryStep objects (usually 1-2 steps for Thought+Action)
        """
        output = output.strip()
        
        # If custom action parser is provided, use it
        if self.custom_action_parser:
            try:
                parsed_result = self.custom_action_parser(output)
                if parsed_result:
                    # Convert to TrajectoryStep format if needed
                    if isinstance(parsed_result, list):
                        return parsed_result
                    elif isinstance(parsed_result, TrajectoryStep):
                        return [parsed_result]
                    elif isinstance(parsed_result, dict):
                        # Assume it's an action
                        metadata = {
                            "raw_output": output, 
                            "custom_parsed": True
                        }
                        
                        # If the parser extracted thought content, add it to metadata
                        if parsed_result.get("thought_content"):
                            metadata["thought_content"] = parsed_result["thought_content"]
                            metadata["has_thought"] = True
                            metadata["combined_output"] = True
                        
                        return [TrajectoryStep(
                            step_type=StepType.ACTION,
                            content=output,  # Keep original LLM output as content
                            metadata=metadata,
                            tool_name=parsed_result.get("tool_name"),
                            tool_args=parsed_result.get("tool_args", {})
                        )]
            except Exception as e:
                logger.warning(f"Custom action parser failed: {e}, falling back to default parser")
        
        # Try to parse combined Thought + Action response first
        combined_steps = self._parse_combined_output(output)
        if combined_steps:
            return combined_steps
        
        # Fallback to single step parsing
        single_step = self._parse_react_output(output, expected_type)
        return [single_step]
    
    def _parse_react_output(self, output: str, expected_type: StepType = None) -> TrajectoryStep:
        """
        Parse LLM output containing both Thought and Action sections.
        Expected format:
        
        Thought: [reasoning text]
        
        Action:
        {
          "name": "function_name",
          "parameters": { ... }
        }
        """
        output = output.strip()
        
        # Try to parse combined Thought + Action response
        thought_action_steps = self._parse_combined_output(output)
        if thought_action_steps:
            # Return the first step (Thought), Action will be processed in next iteration
            return thought_action_steps[0]
        
        # Fallback: try individual section parsing
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
            
            # Try JSON parsing first (preferred format)
            json_action = self._parse_json_action(action_content, output)
            if json_action:
                return json_action
            
            # Fallback to legacy tool call parsing
            tool_name, tool_args = self._parse_tool_call(action_content)
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=output,  # Keep full original output including "Action:" prefix
                metadata={"raw_output": output, "action_content": action_content},
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
        if self._looks_like_tool_call(output) or self._looks_like_json(output):
            # Try JSON parsing first
            json_action = self._parse_json_action(output, output)
            if json_action:
                return json_action
            
            # Fallback to legacy tool call parsing
            tool_name, tool_args = self._parse_tool_call(output)
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=output,  # Keep original output as content
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
    
    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON (contains JSON object)."""
        # Look for JSON patterns with tool_name and parameters
        return ('{' in text and '}' in text and 
                ('name' in text or '"name"' in text or 'tool_name' in text or '"tool_name"' in text))
    
    def _parse_tool_call(self, action_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse tool call from action text."""
        try:
            # Pattern: tool_name(arg1=value1, arg2=value2) or tool_name({json})
            match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', action_text.strip(), re.DOTALL)
            if not match:
                return None, None
            
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            if not args_str:
                return tool_name, {}
            
            # Check if args_str looks like JSON (starts with { and ends with })
            if args_str.startswith('{') and args_str.endswith('}'):
                try:
                    # Parse the JSON object
                    json_obj = json.loads(args_str)
                    if isinstance(json_obj, dict):
                        # If the JSON has 'parameters' field, use that
                        # Otherwise use the whole object as parameters
                        return tool_name, json_obj.get('parameters', json_obj)
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON arguments: {args_str}")
            
            # Parse arguments as key=value pairs
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
                    if args_str.strip():
                        args[f"arg_{len(args)}"] = arg_pair
            
            return tool_name, args
            
        except Exception as e:
            logger.warning(f"Failed to parse tool call '{action_text}': {e}")
            return None, {"raw_text": action_text}
    
    def _parse_combined_output(self, output: str) -> Optional[List[TrajectoryStep]]:
        """
        Parse combined Thought + Action output.
        
        Returns:
            List containing [thought_step, action_step] if successful, None if parsing fails
        """
        try:
            # More flexible parsing to handle different formats
            thought_content = ""
            action_content = ""
            
            # Try to extract Thought and Action sections
            thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', output, re.DOTALL | re.IGNORECASE)
            action_match = re.search(r'Action:\s*(.*)', output, re.DOTALL | re.IGNORECASE)
            
            if thought_match:
                thought_content = thought_match.group(1).strip()
            
            if action_match:
                action_content = action_match.group(1).strip()
                
                # If action content is on the same line as "Action:", it might be inline JSON
                if not action_content.startswith('{'):
                    # Try to extract JSON from the action line
                    json_extracted = self._extract_json_from_text(action_content)
                    if json_extracted:
                        action_content = json_extracted
            
            if not thought_content:
                return None
            
            # If we have both thought and action, create a single ACTION step
            if action_content.strip():
                action_step = self._parse_json_action(action_content.strip(), output)
                if action_step:
                    # Store thought content in metadata, but keep original output as content
                    action_step.metadata["thought_content"] = thought_content.strip()
                    action_step.metadata["has_thought"] = True
                    action_step.metadata["combined_output"] = True
                    # Keep the full original output as content
                    action_step.content = output
                    return [action_step]
            
            # If only thought (no action), create thought step
            thought_step = TrajectoryStep(
                step_type=StepType.THOUGHT,
                content=output,  # Keep full output
                metadata={"raw_output": output, "thought_content": thought_content.strip()}
            )
            return [thought_step]
            
        except Exception as e:
            logger.debug(f"Failed to parse combined output: {e}")
            return None
    
    def _parse_json_action(self, action_content: str, full_output: str) -> Optional[TrajectoryStep]:
        """
        Parse JSON action content with repair_json fallback.
        
        Args:
            action_content: The JSON string to parse
            full_output: The full LLM output for metadata
            
        Returns:
            TrajectoryStep for action, or None if parsing fails
        """
        try:
            # First, try standard JSON parsing
            try:
                action_data = json.loads(action_content)
            except json.JSONDecodeError:
                # Try to clean the JSON (remove comments, fix formatting)
                cleaned_json = self._clean_json_content(action_content)
                
                try:
                    action_data = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    # Try repair_json if available
                    if REPAIR_JSON_AVAILABLE:
                        logger.debug("Standard JSON parsing failed, trying repair_json")
                        try:
                            action_data = json.loads(repair_json(cleaned_json))
                        except Exception as e:
                            logger.debug(f"repair_json also failed: {e}")
                            return None
                    else:
                        logger.warning("JSON parsing failed and repair_json not available")
                        return None
            
            # Validate action structure
            if not isinstance(action_data, dict):
                logger.warning(f"Action must be a JSON object, got: {type(action_data)}")
                return None
            
            tool_name = action_data.get("name")
            parameters = action_data.get("parameters", {})
            
            if not tool_name:
                logger.warning("Action JSON missing 'name' field")
                return None
            
            if not isinstance(parameters, dict):
                logger.warning(f"Action parameters must be a dict, got: {type(parameters)}")
                parameters = {}
            
            # Check for final answer pattern
            if tool_name.lower() == "finish" or "final" in tool_name.lower():
                return TrajectoryStep(
                    step_type=StepType.FINAL_ANSWER,
                    content=str(parameters.get("answer", parameters)),
                    metadata={"raw_output": full_output, "json_action": action_data}
                )
            
            # Create action step - keep full output as content
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=full_output,  # Keep full original LLM output
                metadata={"raw_output": full_output, "json_action": action_data, "parsed_action": action_content},
                tool_name=tool_name,
                tool_args=parameters
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse JSON action: {e}")
            logger.debug(f"Action content: {action_content[:200]}...")
            return None
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that might contain additional content.
        
        Looks for JSON objects between { and } brackets.
        """
        try:
            # Find the first { and last } to extract JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx + 1]
                return json_text
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract JSON from text: {e}")
            return None
    
    def _clean_json_content(self, json_str: str) -> str:
        """
        Clean JSON content by removing comments and fixing common issues.
        
        Args:
            json_str: JSON string that might contain comments or formatting issues
            
        Returns:
            Cleaned JSON string
        """
        try:
            # Remove JavaScript-style comments (// comments)
            lines = json_str.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Remove // comments but keep quoted strings intact
                in_string = False
                escaped = False
                comment_start = -1
                
                for i, char in enumerate(line):
                    if escaped:
                        escaped = False
                        continue
                    
                    if char == '\\':
                        escaped = True
                        continue
                    
                    if char == '"' and not escaped:
                        in_string = not in_string
                        continue
                    
                    if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                        comment_start = i
                        break
                
                if comment_start != -1:
                    line = line[:comment_start].rstrip()
                
                cleaned_lines.append(line)
            
            cleaned_json = '\n'.join(cleaned_lines)
            
            # Remove /* */ style comments (simple approach)
            import re
            cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
            
            return cleaned_json.strip()
            
        except Exception as e:
            logger.debug(f"Failed to clean JSON content: {e}")
            return json_str
    
    def _log_llm_input(self, messages: List[Dict[str, str]], round_num: int):
        """记录LLM输入的详细调试信息"""
        logger.debug("=" * 80)
        logger.debug(f"🔍 [DEBUG] GeneralAgent LLM Input - Round {round_num}")
        logger.debug("=" * 80)
        logger.debug(f"📊 Messages count: {len(messages)}")
        logger.debug(f"🔧 Max tokens: {self.max_tokens_per_step}")
        logger.debug(f"🌡️  Temperature: {self.temperature}")
        
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            logger.debug(f"\n📝 Message {i+1}:")
            logger.debug(f"   Role: {role}")
            logger.debug(f"   Content length: {len(content)} characters")
            
            if len(content) > 500:
                logger.debug(f"   Content (first 500 chars):")
                logger.debug(f"   {content[:500]}...")
                logger.debug(f"   Content (full):")
                logger.debug(f"   {content}")
            else:
                logger.debug(f"   Content:")
                logger.debug(f"   {content}")
        
        logger.debug("🚀 Sending to LLM...")
        logger.debug("=" * 80)
    
    def _log_llm_output(self, response: str, round_num: int):
        """记录LLM输出的详细调试信息"""
        logger.debug("=" * 80)
        logger.debug(f"📤 [DEBUG] GeneralAgent LLM Output - Round {round_num}")
        logger.debug("=" * 80)
        logger.debug(f"📝 Response length: {len(response)} characters")
        
        if len(response) > 800:
            logger.debug(f"📄 Response (first 800 chars):")
            logger.debug(f"{response[:800]}...")
            logger.debug(f"📄 Response (full):")
            logger.debug(f"{response}")
        else:
            logger.debug(f"📄 Response:")
            logger.debug(f"{response}")
        
        # 分析response内容
        has_thought = "Thought:" in response or "thought" in response.lower()
        has_action = "Action:" in response or "action" in response.lower()
        has_json = "{" in response and "}" in response
        
        logger.debug(f"\n🔍 Response Analysis:")
        logger.debug(f"   Contains 'Thought': {has_thought}")
        logger.debug(f"   Contains 'Action': {has_action}")
        logger.debug(f"   Contains JSON: {has_json}")
        
        logger.debug("✅ LLM response received")
        logger.debug("=" * 80)
    
    async def _handle_action(self, action_step: TrajectoryStep, trajectory: Trajectory) -> Optional[TrajectoryStep]:
        """Handle action step execution using the existing tool system."""
        # Get profiler instance
        profiler = self.profiler or get_profiler()
        
        if not action_step.tool_name:
            # No valid tool call found
            return TrajectoryStep(
                step_type=StepType.ACTION_RESULT,
                content=f"Invalid action format: '{action_step.content}'. Please use: tool_name(param=value)",
                metadata={"action_failed": True}
            )
        
        if action_step.tool_name not in self.tools:
            # Tool not available
            available_tools = list(self.tools.keys())
            return TrajectoryStep(
                step_type=StepType.ACTION_RESULT,
                content=f"Tool '{action_step.tool_name}' not available. Available tools: {available_tools}",
                metadata={"action_failed": True}
            )
        
        # Debug: Print tool execution details
        if self.debug:
            print("\n" + "="*80)
            print(f"🔧 [DEBUG] Tool Execution")
            print(f"   Tool Name: {action_step.tool_name}")
            print(f"   Tool Args: {json.dumps(action_step.tool_args or {}, indent=2)}")
            print("="*80)
        
        # Use the existing execute_tool_call method from BaseAgent with profiling
        tool_event_type = EventType.TOOL_EXECUTION
        
        # Map specific tool types to more specific event types
        if "bash" in action_step.tool_name.lower():
            tool_event_type = EventType.BASH_COMMAND
        elif "file" in action_step.tool_name.lower() and "read" in action_step.tool_name.lower():
            tool_event_type = EventType.FILE_READ
        elif "file" in action_step.tool_name.lower() and ("write" in action_step.tool_name.lower() or "edit" in action_step.tool_name.lower()):
            tool_event_type = EventType.FILE_WRITE
        elif "search" in action_step.tool_name.lower():
            tool_event_type = EventType.SEARCH_OPERATION
        
        async with profiler.profile_async(
            f"tool_{action_step.tool_name}",
            tool_event_type,
            {
                "tool_name": action_step.tool_name,
                "tool_args": action_step.tool_args
            }
        ):
            result_step = await self.execute_tool_call(
                action_step.tool_name,
                action_step.tool_args or {},
                trajectory
            )
        
        # Debug: Print tool result
        if self.debug:
            print("\n" + "="*80)
            print(f"📤 [DEBUG] Tool Result")
            print(f"   Tool: {action_step.tool_name}")
            if result_step:
                print(f"   Success: {not result_step.metadata.get('action_failed', False)}")
                print(f"   Result: {result_step.content}")
            else:
                print(f"   Result: None (no result returned)")
            print("="*80)
        
        return result_step
    


def dump_trajectory(trajectory: Trajectory, filepath: str, format: str = "json") -> None:
    """
    Dump trajectory data to file.
    
    Args:
        trajectory: Trajectory to dump
        filepath: Output file path
        format: Output format ("json", "jsonl", or "txt")
    """
    try:
        if format.lower() == "jsonl":
            # Save as JSONL format - one message per line
            with open(filepath, 'w', encoding='utf-8') as f:
                # First, add metadata as a special message
                if trajectory.metadata:
                    meta_msg = {
                        "role": "meta",
                        "content": json.dumps(trajectory.metadata, ensure_ascii=False),
                        "meta": trajectory.metadata  # Also include as structured data
                    }
                    f.write(json.dumps(meta_msg, ensure_ascii=False) + '\n')
                
                # Then, add system prompt if available
                if hasattr(trajectory, 'system_prompt') and trajectory.system_prompt:
                    system_msg = {
                        "role": "system",
                        "content": trajectory.system_prompt
                    }
                    f.write(json.dumps(system_msg, ensure_ascii=False) + '\n')
                
                # Then add all trajectory messages
                messages = trajectory.get_messages()
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        
        elif format.lower() == "json":
            # Convert trajectory to JSON-serializable format
            data = {
                "request_id": trajectory.request_id,
                "is_completed": trajectory.is_completed,
                "final_reward": trajectory.final_reward,
                "total_tokens": trajectory.total_tokens,
                "system_prompt": getattr(trajectory, 'system_prompt', None),
                "metadata": trajectory.metadata,
                "steps": []
            }
            
            for step in trajectory.steps:
                step_data = {
                    "step_type": step.step_type.value,
                    "content": step.content,
                    "metadata": step.metadata,
                    "reward_score": step.reward_score
                }
                
                if step.tool_name:
                    step_data["tool_name"] = step.tool_name
                if step.tool_args:
                    step_data["tool_args"] = step.tool_args
                if step.tool_result:
                    step_data["tool_result"] = step.tool_result
                
                data["steps"].append(step_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "txt":
            # Create human-readable text format
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Trajectory: {trajectory.request_id}\n")
                f.write(f"Completed: {trajectory.is_completed}\n")
                f.write(f"Final Reward: {trajectory.final_reward}\n")
                f.write(f"Total Steps: {len(trajectory.steps)}\n")
                f.write("="*50 + "\n\n")
                
                for i, step in enumerate(trajectory.steps, 1):
                    f.write(f"Step {i}: {step.step_type.value.upper()}\n")
                    f.write(f"Content: {step.content}\n")
                    
                    if step.tool_name:
                        f.write(f"Tool: {step.tool_name}\n")
                        if step.tool_args:
                            f.write(f"Args: {step.tool_args}\n")
                        if step.tool_result:
                            f.write(f"Result: {step.tool_result}\n")
                    
                    if step.reward_score:
                        f.write(f"Reward: {step.reward_score}\n")
                    
                    f.write("-" * 30 + "\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'jsonl', or 'txt'")
        
        logger.info(f"Trajectory dumped to {filepath} in {format} format")
        
    except Exception as e:
        logger.error(f"Failed to dump trajectory: {e}")
        raise


def save_trajectory_as_messages(trajectory: Trajectory, filepath: str, include_system_prompt: bool = True, include_meta: bool = True) -> None:
    """
    Save trajectory as a JSONL file with standard message format.
    
    This creates a clean conversation history that can be used for:
    - Training data
    - Debugging conversations
    - Replaying trajectories
    
    Args:
        trajectory: Trajectory to save
        filepath: Output file path (should end with .jsonl)
        include_system_prompt: Whether to include system prompt as first message
        include_meta: Whether to include metadata as first message
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Optionally add metadata as first message
            if include_meta and trajectory.metadata:
                meta_msg = {
                    "role": "meta",
                    "content": json.dumps(trajectory.metadata, ensure_ascii=False),
                    "meta": trajectory.metadata
                }
                f.write(json.dumps(meta_msg, ensure_ascii=False) + '\n')
            
            # Optionally add system prompt
            if include_system_prompt and hasattr(trajectory, 'system_prompt') and trajectory.system_prompt:
                system_msg = {
                    "role": "system",
                    "content": trajectory.system_prompt
                }
                f.write(json.dumps(system_msg, ensure_ascii=False) + '\n')
            
            # Write each message
            messages = trajectory.get_messages()
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        
        logger.info(f"Trajectory saved as messages to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save trajectory as messages: {e}")
        raise


__all__ = ['GeneralAgent', 'dump_trajectory', 'save_trajectory_as_messages']