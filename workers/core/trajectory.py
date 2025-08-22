"""
Trajectory data structures for agentic rollouts.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class StepType(Enum):
    """Types of steps in a trajectory."""
    OBSERVATION = "observation"  # Initial observation/user input
    THOUGHT = "thought"          # Agent reasoning/thinking
    ACTION = "action"            # Tool call or action
    ACTION_RESULT = "action_result"  # Tool execution result
    FINAL_ANSWER = "final_answer"    # Final response


@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory."""
    
    step_type: StepType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tool-specific fields
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    
    # Reward/scoring fields
    reward_score: Optional[float] = None
    is_correct: Optional[bool] = None
    
    # Generation metadata
    log_probs: Optional[List[float]] = None
    attention_mask: Optional[List[int]] = None
    
    def to_message(self) -> Dict[str, str]:
        """Convert step to message format for LLM input."""
        if self.step_type == StepType.OBSERVATION:
            # Don't add "Task:" or "Observation:" prefix, use content as is
            return {"role": "user", "content": self.content}
        elif self.step_type == StepType.THOUGHT:
            # Use content as is, no "Thought:" prefix
            return {"role": "assistant", "content": self.content}
        elif self.step_type == StepType.ACTION:
            # Use the original content which contains the full LLM output
            # This preserves the exact format that the LLM generated
            return {"role": "assistant", "content": self.content}
        elif self.step_type == StepType.ACTION_RESULT:
            # Don't add "Observation:" prefix, use content as is
            return {"role": "user", "content": self.content}
        elif self.step_type == StepType.FINAL_ANSWER:
            return {"role": "assistant", "content": self.content}
        else:
            return {"role": "assistant", "content": self.content}


@dataclass
class Trajectory:
    """A complete trajectory consisting of multiple steps."""
    
    request_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Final trajectory properties
    is_completed: bool = False
    final_reward: Optional[float] = None
    total_tokens: int = 0
    
    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Convert trajectory to message format for LLM input."""
        messages = []
        for step in self.steps:
            message = step.to_message()
            if message:
                messages.append(message)
        return messages
    
    def get_final_response(self) -> str:
        """Get the final response content."""
        for step in reversed(self.steps):
            if step.step_type == StepType.FINAL_ANSWER:
                return step.content
        # Fallback to last assistant message
        for step in reversed(self.steps):
            if step.to_message().get("role") == "assistant":
                return step.content
        return ""
    
    def get_tool_calls(self) -> List[TrajectoryStep]:
        """Get all tool call steps."""
        return [step for step in self.steps if step.step_type == StepType.ACTION]
    
    def get_total_reward(self) -> float:
        """Calculate total reward from all steps."""
        total = 0.0
        for step in self.steps:
            if step.reward_score is not None:
                total += step.reward_score
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "steps": [
                {
                    "step_type": step.step_type.value,
                    "content": step.content,
                    "metadata": step.metadata,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "tool_result": step.tool_result,
                    "reward_score": step.reward_score,
                    "is_correct": step.is_correct,
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
            "is_completed": self.is_completed,
            "final_reward": self.final_reward,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create trajectory from dictionary."""
        trajectory = cls(request_id=data["request_id"])
        trajectory.metadata = data.get("metadata", {})
        trajectory.is_completed = data.get("is_completed", False)
        trajectory.final_reward = data.get("final_reward")
        trajectory.total_tokens = data.get("total_tokens", 0)
        
        for step_data in data.get("steps", []):
            step = TrajectoryStep(
                step_type=StepType(step_data["step_type"]),
                content=step_data["content"],
                metadata=step_data.get("metadata", {}),
                tool_name=step_data.get("tool_name"),
                tool_args=step_data.get("tool_args"),
                tool_result=step_data.get("tool_result"),
                reward_score=step_data.get("reward_score"),
                is_correct=step_data.get("is_correct"),
            )
            trajectory.add_step(step)
        
        return trajectory