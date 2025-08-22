"""
AgenticRolloutLib - A flexible framework for agentic rollouts in VERL.

This module provides an extensible architecture for implementing various types of agents
that can combine LLM generation with tool calling to produce multi-step trajectories.
"""

from .agentic_rollout import AgenticRollout, AgenticRolloutConfig, create_agentic_rollout
from .core.base_agent import BaseAgent
from .core.trajectory import Trajectory, TrajectoryStep, StepType
from .core.registry import register_agent, get_agent_class

# Tool system
from .core.base_tool import BaseAgenticTool, SimpleAgenticTool, AgenticBaseTool
from .core.tool_registry import ToolRegistry, get_global_tool_registry, register_tool, register_tool_instance
from .core.tool_schemas import ToolResult, ToolCallInfo, create_openai_tool_schema

# Agents
from .agents.react_agent import ReactAgent
from .agents.tool_agent import ToolAgent

# Core tools
from .tools import CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool

# Utils
from .utils import LLMAPIClient

# K8s tools (conditionally import if kodo is available)
try:
    from .tools import K8sBashExecutorTool, K8sFileEditorTool, K8sSearchTool, K8S_TOOLS_AVAILABLE
except ImportError:
    K8sBashExecutorTool = None
    K8sFileEditorTool = None
    K8sSearchTool = None
    K8S_TOOLS_AVAILABLE = False

__all__ = [
    # Core rollout system
    "AgenticRollout",
    "AgenticRolloutConfig",
    "create_agentic_rollout",
    
    # Agent system
    "BaseAgent", 
    "Trajectory",
    "TrajectoryStep",
    "StepType",
    "register_agent",
    "get_agent_class",
    
    # Tool system
    "BaseAgenticTool",
    "SimpleAgenticTool", 
    "AgenticBaseTool",
    "ToolRegistry",
    "get_global_tool_registry",
    "register_tool",
    "register_tool_instance",
    "ToolResult",
    "ToolCallInfo",
    "create_openai_tool_schema",
    
    # Agents
    "ReactAgent",
    "ToolAgent",
    
    # Core tools
    "CalculatorTool",
    "BashExecutorTool",
    "FileEditorTool", 
    "SearchTool",
    
    # Utils
    "LLMAPIClient",
]

# Add K8s tools to exports if available
if K8S_TOOLS_AVAILABLE:
    __all__.extend([
        "K8sBashExecutorTool",
        "K8sFileEditorTool",
        "K8sSearchTool",
    ])