#!/usr/bin/env python3
"""
Tool registry for the agentic rollout library.
Manages tool registration, discovery, and instantiation.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from collections import defaultdict
import asyncio

from .base_tool import BaseAgenticTool, AgenticBaseTool
from .tool_schemas import OpenAIFunctionToolSchema, ToolCallInfo, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools in the agentic rollout library.
    
    Provides centralized tool registration, discovery, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Type[BaseAgenticTool]] = {}
        self._tool_instances: Dict[str, Dict[str, BaseAgenticTool]] = defaultdict(dict)
        self._tool_configs: Dict[str, Dict] = {}
        self._tool_schemas: Dict[str, OpenAIFunctionToolSchema] = {}
    
    def register_tool(self, tool_class: Type[BaseAgenticTool], 
                     config: Optional[Dict] = None, 
                     name: Optional[str] = None) -> None:
        """
        Register a tool class in the registry.
        
        Args:
            tool_class: The tool class to register
            config: Optional configuration for the tool
            name: Optional name override (uses class default if not provided)
        """
        try:
            # Create temporary instance to get schema and name
            temp_instance = tool_class(config or {})
            tool_name = name or temp_instance.name
            
            # Store tool information
            self._tools[tool_name] = tool_class
            self._tool_configs[tool_name] = config or {}
            self._tool_schemas[tool_name] = temp_instance.tool_schema
            
            logger.info(f"Registered tool: {tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_class.__name__}: {e}")
            raise
    
    def register_tool_instance(self, tool_instance: BaseAgenticTool, 
                              name: Optional[str] = None) -> None:
        """
        Register a pre-created tool instance.
        
        Args:
            tool_instance: The tool instance to register
            name: Optional name override
        """
        tool_name = name or tool_instance.name
        self._tools[tool_name] = type(tool_instance)
        self._tool_configs[tool_name] = tool_instance.config
        self._tool_schemas[tool_name] = tool_instance.tool_schema
        
        logger.info(f"Registered tool instance: {tool_name}")
    
    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Optional[OpenAIFunctionToolSchema]:
        """Get the schema for a specific tool."""
        return self._tool_schemas.get(tool_name)
    
    def get_tool_schemas(self) -> Dict[str, OpenAIFunctionToolSchema]:
        """Get schemas for all registered tools."""
        return self._tool_schemas.copy()
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    async def create_tool_instance(self, tool_name: str, instance_id: Optional[str] = None, 
                                  **kwargs) -> str:
        """
        Create an instance of a registered tool.
        
        Args:
            tool_name: Name of the tool to instantiate
            instance_id: Optional instance ID
            **kwargs: Additional arguments for tool creation
            
        Returns:
            The instance ID of the created tool
            
        Raises:
            ValueError: If tool is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        try:
            # Create tool instance
            tool_class = self._tools[tool_name]
            config = self._tool_configs[tool_name]
            tool_instance = tool_class(config)
            
            # Create instance and get ID
            actual_instance_id = await tool_instance.create_instance(instance_id, **kwargs)
            
            # Store in registry
            self._tool_instances[tool_name][actual_instance_id] = tool_instance
            
            logger.debug(f"Created instance {actual_instance_id} of tool {tool_name}")
            return actual_instance_id
            
        except Exception as e:
            logger.error(f"Failed to create instance of tool {tool_name}: {e}")
            raise
    
    async def execute_tool(self, tool_name: str, instance_id: str, 
                          parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute a tool instance.
        
        Args:
            tool_name: Name of the tool
            instance_id: Instance ID of the tool
            parameters: Execution parameters
            **kwargs: Additional execution arguments
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            ValueError: If tool or instance is not found
        """
        if tool_name not in self._tool_instances:
            raise ValueError(f"No instances found for tool '{tool_name}'")
        
        if instance_id not in self._tool_instances[tool_name]:
            raise ValueError(f"Instance '{instance_id}' not found for tool '{tool_name}'")
        
        try:
            tool_instance = self._tool_instances[tool_name][instance_id]
            
            # Validate parameters
            if not tool_instance.validate_parameters(parameters):
                return ToolResult(
                    success=False, 
                    error=f"Invalid parameters for tool {tool_name}"
                )
            
            # Execute tool
            result = await tool_instance.execute_tool(instance_id, parameters, **kwargs)
            
            logger.debug(f"Executed tool {tool_name} instance {instance_id}")
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}:{instance_id}: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def execute_tool_call(self, tool_call: ToolCallInfo, **kwargs) -> ToolResult:
        """
        Execute a tool call using ToolCallInfo.
        
        Args:
            tool_call: ToolCallInfo containing call details
            **kwargs: Additional execution arguments
            
        Returns:
            ToolResult containing the execution result
        """
        return await self.execute_tool(
            tool_call.tool_name,
            tool_call.instance_id,
            tool_call.parameters,
            **kwargs
        )
    
    async def calculate_tool_reward(self, tool_name: str, instance_id: str, **kwargs) -> float:
        """
        Calculate reward for a tool instance.
        
        Args:
            tool_name: Name of the tool
            instance_id: Instance ID of the tool
            **kwargs: Additional arguments for reward calculation
            
        Returns:
            Reward score
        """
        if (tool_name not in self._tool_instances or 
            instance_id not in self._tool_instances[tool_name]):
            logger.warning(f"Tool instance {tool_name}:{instance_id} not found for reward calculation")
            return 0.0
        
        try:
            tool_instance = self._tool_instances[tool_name][instance_id]
            return await tool_instance.calculate_reward(instance_id, **kwargs)
        except Exception as e:
            logger.error(f"Reward calculation failed for {tool_name}:{instance_id}: {e}")
            return 0.0
    
    async def release_tool_instance(self, tool_name: str, instance_id: str, **kwargs) -> None:
        """
        Release a tool instance and clean up resources.
        
        Args:
            tool_name: Name of the tool
            instance_id: Instance ID of the tool
            **kwargs: Additional cleanup arguments
        """
        if (tool_name not in self._tool_instances or 
            instance_id not in self._tool_instances[tool_name]):
            logger.warning(f"Tool instance {tool_name}:{instance_id} not found for release")
            return
        
        try:
            tool_instance = self._tool_instances[tool_name][instance_id]
            await tool_instance.release_instance(instance_id, **kwargs)
            
            # Remove from registry
            del self._tool_instances[tool_name][instance_id]
            
            logger.debug(f"Released tool instance {tool_name}:{instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to release tool instance {tool_name}:{instance_id}: {e}")
    
    async def release_all_instances(self, tool_name: Optional[str] = None) -> None:
        """
        Release all instances of a tool, or all tool instances if no tool specified.
        
        Args:
            tool_name: Optional tool name to release instances for
        """
        if tool_name:
            if tool_name in self._tool_instances:
                instance_ids = list(self._tool_instances[tool_name].keys())
                for instance_id in instance_ids:
                    await self.release_tool_instance(tool_name, instance_id)
        else:
            # Release all instances of all tools
            for tool_name in list(self._tool_instances.keys()):
                await self.release_all_instances(tool_name)
    
    def get_tool_info(self, tool_name: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get information about registered tools.
        
        Args:
            tool_name: Optional tool name, returns info for all tools if not provided
            
        Returns:
            Tool information dictionary or list of dictionaries
        """
        if tool_name:
            if tool_name not in self._tools:
                return {}
            
            return {
                "name": tool_name,
                "class": self._tools[tool_name].__name__,
                "config": self._tool_configs[tool_name],
                "schema": self._tool_schemas[tool_name].model_dump(exclude_unset=True, exclude_none=True),
                "active_instances": len(self._tool_instances.get(tool_name, {}))
            }
        else:
            return [self.get_tool_info(name) for name in self._tools.keys()]
    
    def clear_registry(self) -> None:
        """Clear all registered tools and instances."""
        asyncio.create_task(self.release_all_instances())
        self._tools.clear()
        self._tool_instances.clear()
        self._tool_configs.clear()
        self._tool_schemas.clear()
        logger.info("Tool registry cleared")


# Global tool registry instance
_global_registry = None


def get_global_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool_class: Type[BaseAgenticTool], config: Optional[Dict] = None, 
                 name: Optional[str] = None) -> None:
    """Register a tool in the global registry."""
    get_global_tool_registry().register_tool(tool_class, config, name)


def register_tool_instance(tool_instance: BaseAgenticTool, name: Optional[str] = None) -> None:
    """Register a tool instance in the global registry."""
    get_global_tool_registry().register_tool_instance(tool_instance, name)


__all__ = [
    'ToolRegistry',
    'get_global_tool_registry',
    'register_tool',
    'register_tool_instance'
]