#!/usr/bin/env python3
"""
Tool factory for creating tool instances based on class names and configurations.
"""

import logging
from typing import Any, Dict, Optional, Type, Union
import importlib

from .base_tool import BaseAgenticTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Factory for creating tool instances based on class names and configurations.
    
    This factory supports:
    - Creating tools by class name (e.g., "Calculator" -> CalculatorTool)
    - Passing configuration arguments to tool constructors
    - Automatic module loading and class discovery
    - Tool registration and caching
    """
    
    def __init__(self):
        """Initialize the tool factory."""
        self._tool_classes: Dict[str, Type[BaseAgenticTool]] = {}
        self._module_paths: Dict[str, str] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tool classes."""
        builtin_tools = {
            "Calculator": "workers.tools.calculator_tool.CalculatorTool",
            "BashExecutor": "workers.tools.bash_executor_tool.BashExecutorTool",
            "FileEditor": "workers.tools.file_editor_tool.FileEditorTool",
            "Search": "workers.tools.search_tool.SearchTool",
            "Finish": "workers.tools.finish_tool.FinishTool"
        }
        
        # Try to register K8s tools if available
        try:
            builtin_tools.update({
                "K8sBashExecutor": "workers.tools.k8s_bash_executor_tool.K8sBashExecutorTool",
                "K8sFileEditor": "workers.tools.k8s_file_editor_tool.K8sFileEditorTool",
                "K8sSearch": "workers.tools.k8s_search_tool.K8sSearchTool"
            })
        except ImportError:
            logger.debug("K8s tools not available")
        
        # Try to register R2E tools if available
        try:
            builtin_tools.update({
                "R2EBashExecutor": "workers.tools.r2e_tools.r2e_bash_executor_tool.R2EBashExecutorTool",
                "R2ESearch": "workers.tools.r2e_tools.r2e_search_tool.R2ESearchTool",
                "R2EFileEditor": "workers.tools.r2e_tools.r2e_file_editor_tool.R2EFileEditorTool",
                "R2EStrReplaceEditor": "workers.tools.r2e_tools.r2e_str_replace_editor_tool.R2EStrReplaceEditorTool",
                "R2ESubmit": "workers.tools.r2e_tools.r2e_submit_tool.R2ESubmitTool"
            })
        except ImportError:
            logger.debug("R2E tools not available")
        
        for tool_name, module_path in builtin_tools.items():
            self._module_paths[tool_name] = module_path
    
    def register_tool_class(self, name: str, tool_class: Type[BaseAgenticTool]) -> None:
        """
        Register a tool class directly.
        
        Args:
            name: Tool name (e.g., "Calculator")
            tool_class: Tool class
        """
        self._tool_classes[name] = tool_class
        logger.info(f"Registered tool class: {name}")
    
    def register_tool_module(self, name: str, module_path: str) -> None:
        """
        Register a tool by module path.
        
        Args:
            name: Tool name (e.g., "Calculator")
            module_path: Module path (e.g., "workers.tools.calculator_tool.CalculatorTool")
        """
        self._module_paths[name] = module_path
        logger.info(f"Registered tool module: {name} -> {module_path}")
    
    def _load_tool_class(self, name: str) -> Type[BaseAgenticTool]:
        """
        Load a tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class
            
        Raises:
            ValueError: If tool is not found
        """
        # Check if already loaded
        if name in self._tool_classes:
            return self._tool_classes[name]
        
        # Try to load from module path
        if name in self._module_paths:
            module_path = self._module_paths[name]
            try:
                # Split module and class name
                module_name, class_name = module_path.rsplit('.', 1)
                
                # Import module
                module = importlib.import_module(module_name)
                
                # Get class
                tool_class = getattr(module, class_name)
                
                # Verify it's a valid tool class
                if not issubclass(tool_class, BaseAgenticTool):
                    raise ValueError(f"Class {class_name} is not a subclass of BaseAgenticTool")
                
                # Cache the class
                self._tool_classes[name] = tool_class
                logger.debug(f"Loaded tool class: {name}")
                
                return tool_class
                
            except Exception as e:
                logger.error(f"Failed to load tool class {name} from {module_path}: {e}")
                raise ValueError(f"Failed to load tool class {name}: {e}")
        
        raise ValueError(f"Tool '{name}' not found. Available tools: {list(self._module_paths.keys())}")
    
    def create_tool(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseAgenticTool:
        """
        Create a tool instance by name.
        
        Args:
            name: Tool name (e.g., "Calculator", "BashExecutor")
            config: Configuration dictionary to pass to tool constructor
            **kwargs: Additional keyword arguments for tool constructor
            
        Returns:
            Tool instance
            
        Raises:
            ValueError: If tool is not found or creation fails
        """
        try:
            # Load tool class
            tool_class = self._load_tool_class(name)
            
            # Merge config and kwargs
            final_config = config or {}
            final_config.update(kwargs)
            
            # Create tool instance
            tool_instance = tool_class(final_config)
            
            logger.info(f"Created tool instance: {name} with config: {final_config}")
            return tool_instance
            
        except Exception as e:
            logger.error(f"Failed to create tool {name}: {e}")
            raise ValueError(f"Failed to create tool {name}: {e}")
    
    def create_tools(
        self, 
        tool_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseAgenticTool]:
        """
        Create multiple tool instances from configurations.
        
        Args:
            tool_configs: Dictionary of tool name -> configuration
                         Format: {"Calculator": {"precision": 10}, "Search": {"max_results": 100}}
            
        Returns:
            Dictionary of tool name -> tool instance
        """
        tools = {}
        
        for tool_name, config in tool_configs.items():
            try:
                tools[tool_name.lower()] = self.create_tool(tool_name, config)
            except Exception as e:
                logger.error(f"Failed to create tool {tool_name}: {e}")
                # Continue creating other tools
        
        return tools
    
    def list_available_tools(self) -> Dict[str, str]:
        """
        List all available tool names and their module paths.
        
        Returns:
            Dictionary of tool name -> module path
        """
        available = {}
        available.update(self._module_paths)
        
        # Add already loaded classes
        for name, tool_class in self._tool_classes.items():
            if name not in available:
                available[name] = f"{tool_class.__module__}.{tool_class.__name__}"
        
        return available
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information dictionary
        """
        try:
            tool_class = self._load_tool_class(name)
            
            # Create a temporary instance to get schema info
            temp_instance = tool_class({})
            
            return {
                "name": name,
                "class": f"{tool_class.__module__}.{tool_class.__name__}",
                "schema": temp_instance.tool_schema.model_dump(exclude_unset=True, exclude_none=True) if hasattr(temp_instance, 'tool_schema') else None,
                "description": temp_instance.tool_schema.function.description if hasattr(temp_instance, 'tool_schema') else "No description available"
            }
            
        except Exception as e:
            return {
                "name": name,
                "error": str(e),
                "available": False
            }


# Global tool factory instance
_global_tool_factory = None


def get_global_tool_factory() -> ToolFactory:
    """Get the global tool factory instance."""
    global _global_tool_factory
    if _global_tool_factory is None:
        _global_tool_factory = ToolFactory()
    return _global_tool_factory


def create_tool(
    name: str, 
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseAgenticTool:
    """
    Create a tool instance using the global factory.
    
    Args:
        name: Tool name
        config: Configuration dictionary
        **kwargs: Additional configuration
        
    Returns:
        Tool instance
    """
    return get_global_tool_factory().create_tool(name, config, **kwargs)


def create_tools(tool_configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseAgenticTool]:
    """
    Create multiple tools using the global factory.
    
    Args:
        tool_configs: Dictionary of tool configurations
        
    Returns:
        Dictionary of tool instances
    """
    return get_global_tool_factory().create_tools(tool_configs)


__all__ = [
    'ToolFactory',
    'get_global_tool_factory',
    'create_tool',
    'create_tools'
]