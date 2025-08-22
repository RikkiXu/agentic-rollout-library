#!/usr/bin/env python3
"""
Base tool classes for the agentic rollout library.
Provides compatibility with VERL tools when available, standalone operation otherwise.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from .tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class BaseAgenticTool(ABC):
    """
    Base class for tools in the agentic rollout library.
    
    This class provides a unified interface that can work with or without VERL.
    When VERL is available, it inherits from VERL's BaseTool.
    When VERL is not available, it provides standalone functionality.
    """
    
    def __init__(self, config: Optional[Dict] = None, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        Initialize the base tool.
        
        Args:
            config: Tool configuration dictionary
            tool_schema: OpenAI function tool schema
        """
        self.config = config or {}
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, f"Tool schema is not set for {self.__class__.__name__}!"
        self.name = self.tool_schema.function.name
        
        # Enable debug logging if configured
        if self.config.get("debug", False):
            logger.info(f"Tool {self.name} schema:")
            logger.info(json.dumps(
                self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), 
                indent=2
            ))
    
    @abstractmethod
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema for this tool."""
        pass
    
    def get_description(self) -> str:
        """
        Get tool description for system prompts.
        
        By default, returns the stringified JSON schema.
        Override this method to provide custom descriptions.
        
        Returns:
            Tool description string
        """
        schema = self.get_openai_tool_schema()
        # Convert Pydantic model to dict if needed
        if hasattr(schema, 'model_dump'):
            schema_dict = schema.model_dump()
        elif hasattr(schema, 'dict'):
            schema_dict = schema.dict()
        else:
            schema_dict = schema
        
        # Return formatted JSON schema as default description
        return json.dumps(schema_dict, indent=2, ensure_ascii=False)
    
    async def create_instance(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Create a tool instance.
        
        Args:
            instance_id: Optional instance ID, will generate UUID if not provided
            **kwargs: Additional arguments for instance creation
            
        Returns:
            The instance ID of the created tool instance
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize any instance-specific state
        await self._initialize_instance(instance_id, **kwargs)
        return instance_id
    
    async def _initialize_instance(self, instance_id: str, **kwargs) -> None:
        """
        Initialize instance-specific state. Override in subclasses if needed.
        
        Args:
            instance_id: The instance ID
            **kwargs: Additional initialization arguments
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            instance_id: The tool instance ID
            parameters: Tool execution parameters
            **kwargs: Additional execution arguments
            
        Returns:
            ToolResult containing the execution result
        """
        pass
    
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """
        Execute the tool and return VERL-compatible format.
        
        This method provides compatibility with VERL's expected return format.
        
        Args:
            instance_id: The tool instance ID
            parameters: Tool execution parameters
            **kwargs: Additional execution arguments
            
        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        try:
            result = await self.execute_tool(instance_id, parameters, **kwargs)
            
            # Format response for VERL compatibility
            if result.success:
                response = str(result.result) if result.result is not None else "Tool executed successfully"
            else:
                response = f"Tool execution failed: {result.error}"
            
            return response, result.reward_score, result.metrics
            
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            error_msg = f"Tool execution failed: {str(e)}"
            return error_msg, 0.0, {"error": str(e), "error_type": type(e).__name__}
    
    async def calculate_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate reward for the tool execution.
        
        Args:
            instance_id: The tool instance ID
            **kwargs: Additional arguments for reward calculation
            
        Returns:
            Reward score (float)
        """
        return 0.0
    
    async def release_instance(self, instance_id: str, **kwargs) -> None:
        """
        Release a tool instance and clean up resources.
        
        Args:
            instance_id: The tool instance ID
            **kwargs: Additional cleanup arguments
        """
        await self._cleanup_instance(instance_id, **kwargs)
    
    async def _cleanup_instance(self, instance_id: str, **kwargs) -> None:
        """
        Cleanup instance-specific resources. Override in subclasses if needed.
        
        Args:
            instance_id: The instance ID
            **kwargs: Additional cleanup arguments
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters against the tool schema.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            required_params = self.tool_schema.function.parameters.required
            for param in required_params:
                if param not in parameters:
                    logger.warning(f"Missing required parameter '{param}' for tool {self.name}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Parameter validation failed for tool {self.name}: {e}")
            return False
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information including schema and config."""
        return {
            "name": self.name,
            "schema": self.tool_schema.model_dump(exclude_unset=True, exclude_none=True),
            "config": self.config
        }


class SimpleAgenticTool(BaseAgenticTool):
    """
    A simple implementation of BaseAgenticTool for basic use cases.
    
    This class provides a simpler interface for tools that don't need
    complex instance management.
    """
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 required: Optional[list] = None, config: Optional[Dict] = None):
        """
        Initialize a simple tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
            required: Required parameter names
            config: Tool configuration
        """
        self._name = name
        self._description = description
        self._parameters = parameters
        self._required = required or []
        
        # Create schema
        tool_schema = create_openai_tool_schema(name, description, parameters, required)
        super().__init__(config, tool_schema)
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return create_openai_tool_schema(
            self._name, 
            self._description, 
            self._parameters, 
            self._required
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Default implementation that calls the simple_execute method.
        
        Override either this method or simple_execute in subclasses.
        """
        try:
            result = await self.simple_execute(parameters, **kwargs)
            return ToolResult(success=True, result=result)
        except Exception as e:
            logger.error(f"Simple tool {self.name} execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def simple_execute(self, parameters: Dict[str, Any], **kwargs) -> Any:
        """
        Simple execution method for tools that don't need instance management.
        
        Override this method in subclasses to implement tool functionality.
        
        Args:
            parameters: Tool parameters
            **kwargs: Additional arguments
            
        Returns:
            Tool execution result
        """
        raise NotImplementedError("Subclasses must implement simple_execute method")


# Try to import VERL BaseTool and create compatible class
try:
    from verl.tools.base_tool import BaseTool as VERLBaseTool
    from verl.utils.rollout_trace import rollout_trace_op
    
    class VERLCompatibleTool(VERLBaseTool, BaseAgenticTool):
        """
        VERL-compatible tool class that inherits from both VERL BaseTool
        and BaseAgenticTool to provide full compatibility.
        """
        
        def __init__(self, config: Optional[Dict] = None, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
            # Initialize BaseAgenticTool first
            BaseAgenticTool.__init__(self, config, tool_schema)
            # Initialize VERL BaseTool with compatible parameters
            VERLBaseTool.__init__(self, self.config, self.tool_schema)
        
        async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
            """VERL create method compatibility."""
            return await self.create_instance(instance_id, **kwargs)
        
        @rollout_trace_op
        async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
            """VERL execute method with tracing."""
            return await super().execute(instance_id, parameters, **kwargs)
        
        async def calc_reward(self, instance_id: str, **kwargs) -> float:
            """VERL calc_reward method compatibility."""
            return await self.calculate_reward(instance_id, **kwargs)
        
        async def release(self, instance_id: str, **kwargs) -> None:
            """VERL release method compatibility."""
            await self.release_instance(instance_id, **kwargs)
    
    # Use VERL-compatible class as the base
    AgenticBaseTool = VERLCompatibleTool
    logger.info("VERL tools detected - using VERL-compatible tool base class")
    
except ImportError:
    # VERL not available, use standalone implementation
    AgenticBaseTool = BaseAgenticTool
    logger.info("VERL not available - using standalone tool base class")


__all__ = [
    'BaseAgenticTool',
    'SimpleAgenticTool', 
    'AgenticBaseTool',
    'ToolResult'
]