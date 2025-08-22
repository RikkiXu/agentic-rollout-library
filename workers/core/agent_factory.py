#!/usr/bin/env python3
"""
Agent factory for creating agent instances based on class names and configurations.
"""

import logging
from typing import Any, Dict, Optional, Type
import importlib

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating agent instances based on class names and configurations.
    
    This factory supports:
    - Creating agents by class name (e.g., "General" -> GeneralAgent)
    - Passing configuration arguments to agent constructors
    - Automatic module loading and class discovery
    - Agent registration and caching
    """
    
    def __init__(self):
        """Initialize the agent factory."""
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._module_paths: Dict[str, str] = {}
        
        # Register built-in agents
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """Register built-in agent classes."""
        builtin_agents = {
            "General": "workers.agents.general_agent.GeneralAgent",
            "React": "workers.agents.react_agent.ReactAgent",
            "Tool": "workers.agents.tool_agent.ToolAgent",
            "Coding": "workers.agents.coding_agent.CodingAgent"
        }
        
        for agent_name, module_path in builtin_agents.items():
            self._module_paths[agent_name] = module_path
    
    def register_agent_class(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class directly.
        
        Args:
            name: Agent name (e.g., "General")
            agent_class: Agent class
        """
        self._agent_classes[name] = agent_class
        logger.info(f"Registered agent class: {name}")
    
    def register_agent_module(self, name: str, module_path: str) -> None:
        """
        Register an agent by module path.
        
        Args:
            name: Agent name (e.g., "General")
            module_path: Module path (e.g., "workers.agents.general_agent.GeneralAgent")
        """
        self._module_paths[name] = module_path
        logger.info(f"Registered agent module: {name} -> {module_path}")
    
    def _load_agent_class(self, name: str) -> Type[BaseAgent]:
        """
        Load an agent class by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent class
            
        Raises:
            ValueError: If agent is not found
        """
        # Check if already loaded
        if name in self._agent_classes:
            return self._agent_classes[name]
        
        # Try to load from module path
        if name in self._module_paths:
            module_path = self._module_paths[name]
            try:
                # Split module and class name
                module_name, class_name = module_path.rsplit('.', 1)
                
                # Import module
                module = importlib.import_module(module_name)
                
                # Get class
                agent_class = getattr(module, class_name)
                
                # Verify it's a valid agent class
                if not issubclass(agent_class, BaseAgent):
                    raise ValueError(f"Class {class_name} is not a subclass of BaseAgent")
                
                # Cache the class
                self._agent_classes[name] = agent_class
                logger.debug(f"Loaded agent class: {name}")
                
                return agent_class
                
            except Exception as e:
                logger.error(f"Failed to load agent class {name} from {module_path}: {e}")
                raise ValueError(f"Failed to load agent class {name}: {e}")
        
        raise ValueError(f"Agent '{name}' not found. Available agents: {list(self._module_paths.keys())}")
    
    def create_agent(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent instance by name.
        
        Args:
            name: Agent name (e.g., "General", "React")
            config: Configuration dictionary to pass to agent constructor
            **kwargs: Additional keyword arguments for agent constructor
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent is not found or creation fails
        """
        try:
            # Load agent class
            agent_class = self._load_agent_class(name)
            
            # Merge config and kwargs
            final_config = config or {}
            final_config.update(kwargs)
            
            # Create agent instance
            agent_instance = agent_class(**final_config)
            
            logger.info(f"Created agent instance: {name} with config: {final_config}")
            return agent_instance
            
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {e}")
            raise ValueError(f"Failed to create agent {name}: {e}")
    
    def create_agents(
        self, 
        agent_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseAgent]:
        """
        Create multiple agent instances from configurations.
        
        Args:
            agent_configs: Dictionary of agent name -> configuration
                          Format: {"General": {"max_rounds": 10}, "React": {"temperature": 0.7}}
            
        Returns:
            Dictionary of agent name -> agent instance
        """
        agents = {}
        
        for agent_name, config in agent_configs.items():
            try:
                agents[agent_name.lower()] = self.create_agent(agent_name, config)
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {e}")
                # Continue creating other agents
        
        return agents
    
    def list_available_agents(self) -> Dict[str, str]:
        """
        List all available agent names and their module paths.
        
        Returns:
            Dictionary of agent name -> module path
        """
        available = {}
        available.update(self._module_paths)
        
        # Add already loaded classes
        for name, agent_class in self._agent_classes.items():
            if name not in available:
                available[name] = f"{agent_class.__module__}.{agent_class.__name__}"
        
        return available
    
    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about an agent.
        
        Args:
            name: Agent name
            
        Returns:
            Agent information dictionary
        """
        try:
            agent_class = self._load_agent_class(name)
            
            return {
                "name": name,
                "class": f"{agent_class.__module__}.{agent_class.__name__}",
                "doc": agent_class.__doc__ or "No description available"
            }
            
        except Exception as e:
            return {
                "name": name,
                "error": str(e),
                "available": False
            }


# Global agent factory instance
_global_agent_factory = None


def get_global_agent_factory() -> AgentFactory:
    """Get the global agent factory instance."""
    global _global_agent_factory
    if _global_agent_factory is None:
        _global_agent_factory = AgentFactory()
    return _global_agent_factory


def create_agent(
    name: str, 
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseAgent:
    """
    Create an agent instance using the global factory.
    
    Args:
        name: Agent name
        config: Configuration dictionary
        **kwargs: Additional configuration
        
    Returns:
        Agent instance
    """
    return get_global_agent_factory().create_agent(name, config, **kwargs)


def create_agents(agent_configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseAgent]:
    """
    Create multiple agents using the global factory.
    
    Args:
        agent_configs: Dictionary of agent configurations
        
    Returns:
        Dictionary of agent instances
    """
    return get_global_agent_factory().create_agents(agent_configs)


__all__ = [
    'AgentFactory',
    'get_global_agent_factory',
    'create_agent',
    'create_agents'
]