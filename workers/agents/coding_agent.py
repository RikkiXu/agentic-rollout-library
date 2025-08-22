"""
Coding agent implementation for Software Engineering tasks.

This agent is specifically designed for SWE tasks including:
- Code understanding and exploration
- File manipulation (view, edit, create)
- Code execution and testing
- Bash command execution
- Search across codebase
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field

from ..core.base_agent import BaseAgent
from ..core.registry import register_agent
from ..core.trajectory import Trajectory, TrajectoryStep, StepType

logger = logging.getLogger(__name__)


class SWEPhase(Enum):
    """Different phases in SWE problem solving."""
    UNDERSTANDING = "understanding"      # Understanding the problem
    EXPLORATION = "exploration"          # Exploring codebase
    PLANNING = "planning"               # Planning the solution
    IMPLEMENTATION = "implementation"    # Implementing changes
    TESTING = "testing"                 # Testing and validation
    REFINEMENT = "refinement"           # Refining and fixing issues


@dataclass
class CodebaseState:
    """Track the state of the codebase during agent execution."""
    
    # Files that have been viewed/explored
    viewed_files: Set[str] = field(default_factory=set)
    
    # Files that have been modified
    modified_files: Set[str] = field(default_factory=set)
    
    # Test results
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Search results cache
    search_cache: Dict[str, List[str]] = field(default_factory=dict)
    
    # Current working directory
    current_dir: str = "."
    
    # Error history
    errors_encountered: List[str] = field(default_factory=list)
    
    # Implementation plan
    implementation_plan: List[str] = field(default_factory=list)


@register_agent("coding")
class CodingAgent(BaseAgent):
    """
    Specialized agent for Software Engineering tasks.
    
    This agent follows a structured approach:
    1. Problem Understanding - Analyze the requirements
    2. Codebase Exploration - Understand existing code structure
    3. Planning - Create implementation plan
    4. Implementation - Make code changes
    5. Testing - Validate changes work correctly
    6. Refinement - Fix issues and optimize
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # SWE-specific configuration
        self.max_files_to_explore = kwargs.get("max_files_to_explore", 20)
        self.max_search_results = kwargs.get("max_search_results", 10)
        self.enable_bash_execution = kwargs.get("enable_bash_execution", True)
        self.test_after_changes = kwargs.get("test_after_changes", True)
        self.max_implementation_attempts = kwargs.get("max_implementation_attempts", 3)
        
        # Required tools for SWE tasks
        self.required_tools = {
            "view_file", "edit_file", "create_file", "search_code", 
            "execute_bash", "run_tests", "search_files"
        }
        
        # Initialize codebase state
        self.codebase_state = CodebaseState()
        self.current_phase = SWEPhase.UNDERSTANDING
        self.implementation_attempts = 0
    
    def create_system_prompt(self) -> str:
        """Create SWE-specific system prompt."""
        tools_description = self._get_tools_description()
        
        return f"""You are an expert software engineer AI that solves coding problems systematically.

Your approach should follow these phases:
1. **Understanding**: Analyze the problem requirements carefully
2. **Exploration**: Explore the codebase to understand structure and existing code
3. **Planning**: Create a detailed implementation plan
4. **Implementation**: Make the necessary code changes
5. **Testing**: Validate that changes work correctly
6. **Refinement**: Fix any issues found during testing

Guidelines:
- Always explore relevant files before making changes
- Search for similar patterns or existing implementations
- Make minimal, focused changes
- Test your changes thoroughly
- Provide clear explanations for your actions

{tools_description}

Format your actions as: Action: tool_name(arg1=value1, arg2=value2)
Think step by step and explain your reasoning before each action."""
    
    def _get_tools_description(self) -> str:
        """Generate description of available tools."""
        if not self.tools:
            return "\nNo tools available."
        
        tool_descriptions = []
        tool_mapping = {
            "view_file": "View contents of a file",
            "edit_file": "Edit or modify a file", 
            "create_file": "Create a new file",
            "search_code": "Search for code patterns in the codebase",
            "search_files": "Search for files by name pattern",
            "execute_bash": "Execute bash commands",
            "run_tests": "Run tests to validate changes"
        }
        
        for tool_name in self.tools:
            description = tool_mapping.get(tool_name, f"Tool: {tool_name}")
            tool_descriptions.append(f"- {tool_name}: {description}")
        
        return f"\nAvailable tools:\n" + "\n".join(tool_descriptions)
    
    async def run_trajectory(
        self,
        prompt: Dict[str, Any],
        llm_generate_func,
        request_id: str,
        **kwargs
    ) -> Trajectory:
        """Run a complete coding trajectory."""
        trajectory = Trajectory(request_id=request_id)
        
        # Reset state for new trajectory
        self.codebase_state = CodebaseState()
        self.current_phase = SWEPhase.UNDERSTANDING
        self.implementation_attempts = 0
        
        # Add initial problem description
        problem_description = self._extract_prompt_content(prompt)
        initial_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=problem_description,
            metadata={
                "prompt": prompt,
                "phase": self.current_phase.value
            }
        )
        trajectory.add_step(initial_step)
        
        while self.should_continue(trajectory):
            try:
                # Determine next action based on current phase
                next_action = await self._determine_next_action(trajectory, llm_generate_func)
                
                if next_action is None:
                    break
                
                # Execute the action
                await self._execute_swe_action(next_action, trajectory, llm_generate_func)
                
                # Update phase if needed
                self._update_phase(trajectory)
                
            except Exception as e:
                logger.error(f"Error in coding trajectory {request_id}: {e}")
                self.codebase_state.errors_encountered.append(str(e))
                
                error_step = TrajectoryStep(
                    step_type=StepType.THOUGHT,
                    content=f"I encountered an error: {str(e)}. Let me try a different approach.",
                    metadata={
                        "error": str(e),
                        "phase": self.current_phase.value
                    }
                )
                trajectory.add_step(error_step)
        
        self.finalize_trajectory(trajectory)
        return trajectory
    
    async def _determine_next_action(
        self,
        trajectory: Trajectory,
        llm_generate_func
    ) -> Optional[TrajectoryStep]:
        """Determine the next action based on current phase and state."""
        
        # Create context-aware prompt based on current phase
        phase_guidance = self._get_phase_guidance()
        state_summary = self._get_state_summary()
        
        messages = self.format_messages_for_llm(trajectory)
        messages.append({
            "role": "user", 
            "content": f"{phase_guidance}\n\n{state_summary}\n\nWhat should you do next?"
        })
        
        # Generate response
        response = await llm_generate_func(
            messages,
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature
        )
        
        # Parse the response
        action_step = self._parse_swe_output(response)
        trajectory.add_step(action_step)
        
        return action_step
    
    def _get_phase_guidance(self) -> str:
        """Get guidance for current phase."""
        guidance_map = {
            SWEPhase.UNDERSTANDING: 
                "Focus on understanding the problem requirements. Ask clarifying questions if needed.",
            
            SWEPhase.EXPLORATION: 
                "Explore the codebase to understand structure and find relevant files. Use search tools effectively.",
            
            SWEPhase.PLANNING: 
                "Create a detailed implementation plan based on your exploration. Break down the task into steps.",
            
            SWEPhase.IMPLEMENTATION: 
                "Implement the planned changes. Make focused, minimal modifications.",
            
            SWEPhase.TESTING: 
                "Test your implementation to ensure it works correctly. Run relevant tests.",
            
            SWEPhase.REFINEMENT: 
                "Fix any issues found during testing and optimize the solution."
        }
        
        return f"Current Phase: {self.current_phase.value.upper()}\n{guidance_map[self.current_phase]}"
    
    def _get_state_summary(self) -> str:
        """Get summary of current codebase state."""
        summary_parts = []
        
        if self.codebase_state.viewed_files:
            summary_parts.append(f"Viewed files: {', '.join(list(self.codebase_state.viewed_files)[:5])}")
        
        if self.codebase_state.modified_files:
            summary_parts.append(f"Modified files: {', '.join(self.codebase_state.modified_files)}")
        
        if self.codebase_state.test_results:
            passed = sum(1 for r in self.codebase_state.test_results.values() if r.get("passed", False))
            total = len(self.codebase_state.test_results)
            summary_parts.append(f"Test results: {passed}/{total} passed")
        
        if self.codebase_state.errors_encountered:
            summary_parts.append(f"Recent errors: {len(self.codebase_state.errors_encountered)}")
        
        return "Current State:\n" + "\n".join(f"- {part}" for part in summary_parts) if summary_parts else "No state information yet."
    
    def _parse_swe_output(self, output: str) -> TrajectoryStep:
        """Parse LLM output for SWE-specific patterns."""
        output = output.strip()
        
        # Check for thought/reasoning
        if output.startswith("Thought:") or "let me think" in output.lower():
            return TrajectoryStep(
                step_type=StepType.THOUGHT,
                content=output,
                metadata={
                    "phase": self.current_phase.value,
                    "raw_output": output
                }
            )
        
        # Check for action
        if output.startswith("Action:"):
            action_content = output[7:].strip()
            tool_name, tool_args = self._parse_tool_call(action_content)
            
            return TrajectoryStep(
                step_type=StepType.ACTION,
                content=action_content,
                metadata={
                    "phase": self.current_phase.value,
                    "raw_output": output
                },
                tool_name=tool_name,
                tool_args=tool_args
            )
        
        # Check for final answer
        if any(keyword in output.lower() for keyword in ["final answer", "solution complete", "implementation done"]):
            return TrajectoryStep(
                step_type=StepType.FINAL_ANSWER,
                content=output,
                metadata={
                    "phase": self.current_phase.value,
                    "raw_output": output
                }
            )
        
        # Default to thought
        return TrajectoryStep(
            step_type=StepType.THOUGHT,
            content=output,
            metadata={
                "phase": self.current_phase.value,
                "raw_output": output,
                "inferred_type": True
            }
        )
    
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
            
            # Handle different argument patterns
            if '=' in args_str:
                # key=value format
                for arg_pair in self._split_args(args_str):
                    if '=' in arg_pair:
                        key, value = arg_pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        args[key] = value
            else:
                # Positional arguments
                args["content"] = args_str
            
            return tool_name, args
            
        except Exception as e:
            logger.warning(f"Failed to parse tool call '{action_text}': {e}")
            return None, {"raw_text": action_text}
    
    def _split_args(self, args_str: str) -> List[str]:
        """Split arguments string, respecting quotes."""
        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None
        
        for char in args_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == ',' and not in_quotes:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    async def _execute_swe_action(
        self,
        action_step: TrajectoryStep,
        trajectory: Trajectory,
        llm_generate_func
    ):
        """Execute SWE-specific action with state tracking."""
        if action_step.step_type != StepType.ACTION or not action_step.tool_name:
            return
        
        tool_name = action_step.tool_name
        tool_args = action_step.tool_args or {}
        
        # Execute the tool
        result_step = await self.execute_tool_call(tool_name, tool_args, trajectory)
        
        # Update codebase state based on tool execution
        self._update_codebase_state(tool_name, tool_args, result_step)
        
        # Add result to trajectory
        trajectory.add_step(result_step)
    
    def _update_codebase_state(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result_step: TrajectoryStep
    ):
        """Update codebase state based on tool execution."""
        
        if tool_name == "view_file":
            file_path = tool_args.get("file_path", tool_args.get("path"))
            if file_path:
                self.codebase_state.viewed_files.add(file_path)
        
        elif tool_name in ["edit_file", "create_file"]:
            file_path = tool_args.get("file_path", tool_args.get("path"))
            if file_path:
                self.codebase_state.modified_files.add(file_path)
        
        elif tool_name == "search_code":
            query = tool_args.get("query", tool_args.get("pattern"))
            if query and result_step.tool_result:
                self.codebase_state.search_cache[query] = result_step.tool_result
        
        elif tool_name == "run_tests":
            test_name = tool_args.get("test_name", "all_tests")
            self.codebase_state.test_results[test_name] = result_step.tool_result
        
        elif tool_name == "execute_bash":
            command = tool_args.get("command")
            if command and "cd " in command:
                # Track directory changes
                match = re.search(r'cd\s+([^\s;]+)', command)
                if match:
                    self.codebase_state.current_dir = match.group(1)
    
    def _update_phase(self, trajectory: Trajectory):
        """Update current phase based on trajectory progress."""
        if len(trajectory.steps) < 2:
            return
        
        # Phase transition logic based on actions taken
        recent_actions = [
            step for step in trajectory.steps[-10:] 
            if step.step_type == StepType.ACTION and step.tool_name
        ]
        
        action_types = [step.tool_name for step in recent_actions]
        
        # Transition logic
        if self.current_phase == SWEPhase.UNDERSTANDING:
            if any(tool in action_types for tool in ["search_files", "view_file", "search_code"]):
                self.current_phase = SWEPhase.EXPLORATION
        
        elif self.current_phase == SWEPhase.EXPLORATION:
            if len(self.codebase_state.viewed_files) >= 3:
                self.current_phase = SWEPhase.PLANNING
        
        elif self.current_phase == SWEPhase.PLANNING:
            if any(tool in action_types for tool in ["edit_file", "create_file"]):
                self.current_phase = SWEPhase.IMPLEMENTATION
                self.implementation_attempts += 1
        
        elif self.current_phase == SWEPhase.IMPLEMENTATION:
            if "run_tests" in action_types:
                self.current_phase = SWEPhase.TESTING
        
        elif self.current_phase == SWEPhase.TESTING:
            # Check if tests are passing
            if self.codebase_state.test_results:
                all_passed = all(
                    result.get("passed", False) 
                    for result in self.codebase_state.test_results.values()
                )
                if all_passed:
                    self.current_phase = SWEPhase.REFINEMENT
                elif self.implementation_attempts < self.max_implementation_attempts:
                    self.current_phase = SWEPhase.IMPLEMENTATION
                    self.implementation_attempts += 1
    
    def should_continue(self, trajectory: Trajectory) -> bool:
        """SWE-specific continuation logic."""
        # Call parent method first
        if not super().should_continue(trajectory):
            return False
        
        # Additional SWE-specific termination conditions
        
        # Stop if we've made too many implementation attempts
        if self.implementation_attempts >= self.max_implementation_attempts:
            logger.info(f"Reached max implementation attempts ({self.max_implementation_attempts})")
            return False
        
        # Stop if we're in refinement phase and tests are passing
        if (self.current_phase == SWEPhase.REFINEMENT and 
            self.codebase_state.test_results and
            all(result.get("passed", False) for result in self.codebase_state.test_results.values())):
            logger.info("All tests passing in refinement phase")
            return False
        
        # Stop if too many errors
        if len(self.codebase_state.errors_encountered) > 5:
            logger.info("Too many errors encountered")
            return False
        
        return True
    
    def _extract_prompt_content(self, prompt: Dict[str, Any]) -> str:
        """Extract problem description from prompt."""
        if isinstance(prompt, str):
            return prompt
        
        if "problem" in prompt:
            return prompt["problem"]
        elif "description" in prompt:
            return prompt["description"]
        elif "content" in prompt:
            return prompt["content"]
        elif "messages" in prompt and prompt["messages"]:
            for msg in reversed(prompt["messages"]):
                if msg.get("role") == "user":
                    return msg.get("content", "")
        
        return str(prompt)
    
    def finalize_trajectory(self, trajectory: Trajectory) -> None:
        """Finalize coding trajectory with SWE-specific metrics."""
        super().finalize_trajectory(trajectory)
        
        # Add SWE-specific metadata
        trajectory.metadata.update({
            "final_phase": self.current_phase.value,
            "files_explored": len(self.codebase_state.viewed_files),
            "files_modified": len(self.codebase_state.modified_files),
            "implementation_attempts": self.implementation_attempts,
            "tests_run": len(self.codebase_state.test_results),
            "errors_count": len(self.codebase_state.errors_encountered),
            "codebase_state": {
                "viewed_files": list(self.codebase_state.viewed_files),
                "modified_files": list(self.codebase_state.modified_files),
                "test_results": self.codebase_state.test_results
            }
        })
        
        logger.info(
            f"Finalized coding trajectory {trajectory.request_id}: "
            f"phase={self.current_phase.value}, "
            f"files_explored={len(self.codebase_state.viewed_files)}, "
            f"files_modified={len(self.codebase_state.modified_files)}, "
            f"tests_run={len(self.codebase_state.test_results)}"
        )