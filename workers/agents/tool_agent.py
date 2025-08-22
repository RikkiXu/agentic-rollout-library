"""
Simple tool-calling agent implementation.
"""
import logging
from typing import Dict, List, Any, Optional

from ..core.base_agent import BaseAgent
from ..core.registry import register_agent
from ..core.trajectory import Trajectory, TrajectoryStep, StepType

logger = logging.getLogger(__name__)


@register_agent("tool")
class ToolAgent(BaseAgent):
    """
    Simple tool-calling agent that can:
    1. Understand the problem
    2. Call appropriate tools
    3. Provide final answer based on tool results
    
    This is a simpler alternative to ReAct for cases where explicit
    reasoning steps are not required.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Tool agent specific configuration
        self.allow_multiple_tool_calls = kwargs.get("allow_multiple_tool_calls", True)
        self.require_tool_usage = kwargs.get("require_tool_usage", False)
        self.max_tool_calls_per_step = kwargs.get("max_tool_calls_per_step", 3)
    
    def create_system_prompt(self) -> str:
        """Create tool agent system prompt."""
        tools_description = ""
        if self.tools:
            tools_list = []
            for tool_name, tool in self.tools.items():
                description = getattr(tool, 'description', f"Tool: {tool_name}")
                tools_list.append(f"- {tool_name}: {description}")
            tools_description = f"\n\nAvailable tools:\n" + "\n".join(tools_list)
        
        return f"""You are a helpful AI assistant that can use tools to solve problems.

When you need to use a tool, format your response as:
TOOL_CALL: tool_name(arg1=value1, arg2=value2)

You can make multiple tool calls in a single response if needed.
After getting tool results, provide a clear final answer.{tools_description}"""
    
    async def run_trajectory(
        self,
        prompt: Dict[str, Any],
        llm_generate_func,
        request_id: str,
        **kwargs
    ) -> Trajectory:
        """Run a complete tool agent trajectory."""
        trajectory = Trajectory(request_id=request_id)
        
        # Add initial observation
        initial_content = self._extract_prompt_content(prompt)
        initial_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=initial_content,
            metadata={"prompt": prompt}
        )
        trajectory.add_step(initial_step)
        
        while self.should_continue(trajectory):
            try:
                # Generate response
                messages = self.format_messages_for_llm(trajectory)
                
                response = await llm_generate_func(
                    messages,
                    max_tokens=self.max_tokens_per_step,
                    temperature=self.temperature
                )
                
                # Parse response for tool calls and content
                tool_calls, remaining_content = self._parse_tool_calls(response)
                
                if tool_calls:
                    # Execute tool calls
                    for tool_name, tool_args in tool_calls:
                        # Add action step
                        action_step = TrajectoryStep(
                            step_type=StepType.ACTION,
                            content=f"Calling {tool_name} with {tool_args}",
                            metadata={"raw_output": response},
                            tool_name=tool_name,
                            tool_args=tool_args
                        )
                        trajectory.add_step(action_step)
                        
                        # Execute tool
                        result_step = await self.execute_tool_call(tool_name, tool_args, trajectory)
                        trajectory.add_step(result_step)
                
                # If there's remaining content after tool calls, treat as final answer
                if remaining_content.strip():
                    final_step = TrajectoryStep(
                        step_type=StepType.FINAL_ANSWER,
                        content=remaining_content.strip(),
                        metadata={"raw_output": response}
                    )
                    trajectory.add_step(final_step)
                    trajectory.is_completed = True
                    break
                
                # If only tool calls and no remaining content, continue for final answer
                if tool_calls and not remaining_content.strip():
                    # Add a prompt for final answer
                    prompt_step = TrajectoryStep(
                        step_type=StepType.OBSERVATION,
                        content="Based on the tool results above, please provide a final answer.",
                        metadata={"system_prompt": True}
                    )
                    trajectory.add_step(prompt_step)
                    continue
                
                # If no tool calls and no clear final answer, treat as thought/intermediate response
                if not tool_calls:
                    if self._looks_like_final_answer(response):
                        final_step = TrajectoryStep(
                            step_type=StepType.FINAL_ANSWER,
                            content=response.strip(),
                            metadata={"raw_output": response}
                        )
                        trajectory.add_step(final_step)
                        trajectory.is_completed = True
                        break
                    else:
                        thought_step = TrajectoryStep(
                            step_type=StepType.THOUGHT,
                            content=response.strip(),
                            metadata={"raw_output": response}
                        )
                        trajectory.add_step(thought_step)
                
            except Exception as e:
                logger.error(f"Error in trajectory {request_id}: {e}")
                error_step = TrajectoryStep(
                    step_type=StepType.THOUGHT,
                    content=f"I encountered an error: {str(e)}. Let me try to provide an answer anyway.",
                    metadata={"error": str(e)}
                )
                trajectory.add_step(error_step)
        
        self.finalize_trajectory(trajectory)
        return trajectory
    
    def _extract_prompt_content(self, prompt: Dict[str, Any]) -> str:
        """Extract content from prompt data structure."""
        if isinstance(prompt, str):
            return prompt
        
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
    
    def _parse_tool_calls(self, text: str) -> tuple[List[tuple[str, Dict[str, Any]]], str]:
        """
        Parse tool calls from text.
        
        Returns:
            (tool_calls, remaining_content)
        """
        import re
        
        tool_calls = []
        remaining_text = text
        
        # Pattern for TOOL_CALL: tool_name(args)
        pattern = r'TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
        
        for match in re.finditer(pattern, text):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Parse arguments
            args = self._parse_args_string(args_str)
            tool_calls.append((tool_name, args))
            
            # Remove this tool call from remaining text
            remaining_text = remaining_text.replace(match.group(0), "")
        
        return tool_calls, remaining_text.strip()
    
    def _parse_args_string(self, args_str: str) -> Dict[str, Any]:
        """Parse arguments string into dictionary."""
        import json
        
        if not args_str:
            return {}
        
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
                    args[key] = json.loads(value)
                except:
                    args[key] = value
            else:
                # Positional argument
                args[f"arg_{len(args)}"] = arg_pair
        
        return args
    
    def _looks_like_final_answer(self, text: str) -> bool:
        """Check if text looks like a final answer."""
        text_lower = text.lower()
        
        # Check for explicit final answer indicators
        final_indicators = [
            "final answer:",
            "answer:",
            "the answer is",
            "in conclusion",
            "therefore",
            "so the answer",
        ]
        
        for indicator in final_indicators:
            if indicator in text_lower:
                return True
        
        # Check if it's a short, definitive response
        if len(text.split()) < 50 and not text.startswith("I need to"):
            return True
        
        return False