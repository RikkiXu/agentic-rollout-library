"""
Registry system for agent types.
"""
from typing import Dict, Type, Optional
from abc import ABC


# Global registry for agent classes
_AGENT_REGISTRY: Dict[str, Type] = {}


def register_agent(name: str):
    """
    Decorator to register an agent class.
    
    Args:
        name: The name to register the agent under
        
    Usage:
        @register_agent("react")
        class ReactAgent(BaseAgent):
            pass
    """
    def wrapper(cls):
        if name in _AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered")
        _AGENT_REGISTRY[name] = cls
        return cls
    return wrapper


def get_agent_class(name: str) -> Optional[Type]:
    """
    Get an agent class by name.
    
    Args:
        name: The name of the agent to retrieve
        
    Returns:
        The agent class or None if not found
    """
    return _AGENT_REGISTRY.get(name)


def list_agents() -> Dict[str, Type]:
    """
    List all registered agents.
    
    Returns:
        Dictionary mapping agent names to classes
    """
    return _AGENT_REGISTRY.copy()


def clear_registry():
    """Clear the agent registry (mainly for testing)."""
    global _AGENT_REGISTRY
    _AGENT_REGISTRY = {}


class AgentRegistry:
    """Registry class for managing agent types."""
    
    def __init__(self):
        """Initialize agent registry."""
        self._registry = _AGENT_REGISTRY
    
    def register(self, name: str, agent_class: Type) -> None:
        """Register an agent class."""
        if name in self._registry:
            raise ValueError(f"Agent '{name}' is already registered")
        self._registry[name] = agent_class
    
    def get(self, name: str) -> Optional[Type]:
        """Get an agent class by name."""
        return self._registry.get(name)
    
    def list(self) -> Dict[str, Type]:
        """List all registered agents."""
        return self._registry.copy()
    
    def clear(self) -> None:
        """Clear the registry."""
        self._registry.clear()


# Global registry instance
_global_agent_registry = AgentRegistry()