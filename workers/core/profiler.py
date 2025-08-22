#!/usr/bin/env python3
"""
Performance Profiler for Agentic Rollout Library.
Tracks and visualizes timing information for different operations during agent execution.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from contextlib import contextmanager, asynccontextmanager
import threading


class EventType(Enum):
    """Types of events to track in the profiler."""
    # Core operations
    LLM_CALL = "llm_call"
    TOOL_EXECUTION = "tool_execution"
    ACTION_PARSING = "action_parsing"
    
    # Tool-specific operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    BASH_COMMAND = "bash_command"
    SEARCH_OPERATION = "search_operation"
    
    # K8s operations
    K8S_COMMAND = "k8s_command"
    POD_OPERATION = "pod_operation"
    POD_CREATION = "pod_creation"
    POD_DELETION = "pod_deletion"
    
    # Agent operations
    TRAJECTORY_STEP = "trajectory_step"
    THOUGHT_GENERATION = "thought_generation"
    
    # System operations
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    
    # Instance processing
    INSTANCE_PROCESSING = "instance_processing"
    
    # Custom events
    CUSTOM = "custom"


# Color mapping for different event types
EVENT_COLORS = {
    EventType.LLM_CALL: "#FF6B6B",          # Red
    EventType.TOOL_EXECUTION: "#4ECDC4",    # Teal
    EventType.ACTION_PARSING: "#45B7D1",    # Light Blue
    EventType.FILE_READ: "#96CEB4",        # Green
    EventType.FILE_WRITE: "#DDA0DD",       # Plum
    EventType.BASH_COMMAND: "#F4A460",     # Sandy Brown
    EventType.SEARCH_OPERATION: "#98D8C8", # Mint
    EventType.K8S_COMMAND: "#F7DC6F",      # Yellow
    EventType.POD_OPERATION: "#F8C471",    # Orange
    EventType.POD_CREATION: "#FF8C00",     # Dark Orange
    EventType.POD_DELETION: "#DC143C",     # Crimson
    EventType.TRAJECTORY_STEP: "#BB8FCE",  # Purple
    EventType.THOUGHT_GENERATION: "#85C1E2", # Sky Blue
    EventType.NETWORK_IO: "#F1948A",       # Salmon
    EventType.DISK_IO: "#82E0AA",          # Light Green
    EventType.INSTANCE_PROCESSING: "#4169E1", # Royal Blue
    EventType.CUSTOM: "#D7BDE2",           # Lavender
}


@dataclass
class ProfileEvent:
    """Represents a single profiled event."""
    name: str
    event_type: EventType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"{time.time()}_{threading.get_ident()}")
    
    def complete(self):
        """Mark the event as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "event_type": self.event_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "event_id": self.event_id
        }


class RolloutProfiler:
    """Performance profiler for tracking rollout execution."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.events: List[ProfileEvent] = []
        self.active_events: Dict[str, ProfileEvent] = {}
        self.start_time = time.time()
        self._lock = threading.RLock()  # Use RLock to allow re-entrant locking
    
    @contextmanager
    def profile(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a synchronous operation.
        
        Args:
            name: Name of the operation
            event_type: Type of event
            metadata: Additional metadata to store
        """
        if not self.enabled:
            yield
            return
        
        event = self._start_event(name, event_type, metadata)
        try:
            yield event
        finally:
            self._end_event(event)
    
    @asynccontextmanager
    async def profile_async(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling an asynchronous operation.
        
        Args:
            name: Name of the operation
            event_type: Type of event
            metadata: Additional metadata to store
        """
        if not self.enabled:
            yield
            return
        
        event = self._start_event(name, event_type, metadata)
        try:
            yield event
        finally:
            self._end_event(event)
    
    def start_event(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an event manually.
        
        Args:
            name: Name of the operation
            event_type: Type of event
            metadata: Additional metadata to store
            
        Returns:
            Event ID for ending the event later
        """
        if not self.enabled:
            return ""
        
        event = self._start_event(name, event_type, metadata)
        return event.event_id
    
    def end_event(self, event_id: str):
        """
        End a manually tracked event.
        
        Args:
            event_id: ID of the event to end
        """
        if not self.enabled or not event_id:
            return
        
        with self._lock:
            if event_id in self.active_events:
                event = self.active_events.pop(event_id)
                event.complete()
                self.events.append(event)
    
    def _start_event(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None) -> ProfileEvent:
        """Internal method to start an event."""
        event = ProfileEvent(
            name=name,
            event_type=event_type,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.active_events[event.event_id] = event
        
        return event
    
    def _end_event(self, event: ProfileEvent):
        """Internal method to end an event."""
        event.complete()
        with self._lock:
            self.events.append(event)
            # Remove from active events if still there
            self.active_events.pop(event.event_id, None)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of profiled events.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.events:
            return {"total_duration": 0, "event_count": 0, "events_by_type": {}}
        
        total_duration = time.time() - self.start_time
        events_by_type = {}
        
        for event in self.events:
            event_type = event.event_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "max_duration": 0,
                    "min_duration": float('inf')
                }
            
            stats = events_by_type[event_type]
            stats["count"] += 1
            if event.duration:
                stats["total_duration"] += event.duration
                stats["max_duration"] = max(stats["max_duration"], event.duration)
                stats["min_duration"] = min(stats["min_duration"], event.duration)
        
        # Calculate averages
        for stats in events_by_type.values():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                if stats["min_duration"] == float('inf'):
                    stats["min_duration"] = 0
        
        return {
            "total_duration": total_duration,
            "event_count": len(self.events),
            "events_by_type": events_by_type,
            "start_time": self.start_time,
            "end_time": time.time()
        }
    
    def export_events(self, filepath: str):
        """
        Export events to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "summary": self.get_summary(),
            "events": [event.to_dict() for event in self.events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self):
        """Reset the profiler."""
        with self._lock:
            self.events.clear()
            self.active_events.clear()
            self.start_time = time.time()
    
    def get_timeline_data(self) -> List[Dict[str, Any]]:
        """
        Get timeline data for visualization.
        
        Returns:
            List of event data suitable for timeline visualization
        """
        timeline_data = []
        
        for event in self.events:
            if event.duration is None:
                continue
            
            timeline_data.append({
                "name": event.name,
                "type": event.event_type.value,
                "start": event.start_time - self.start_time,
                "duration": event.duration,
                "color": EVENT_COLORS.get(event.event_type, "#CCCCCC"),
                "metadata": event.metadata
            })
        
        return sorted(timeline_data, key=lambda x: x["start"])


# Global profiler instance
_global_profiler: Optional[RolloutProfiler] = None


def get_profiler() -> RolloutProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = RolloutProfiler()
    return _global_profiler


def set_profiler(profiler: RolloutProfiler):
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def profile(name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for profiling functions."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                async with get_profiler().profile_async(name, event_type, metadata):
                    return await func(*args, **kwargs)
        else:
            def wrapper(*args, **kwargs):
                with get_profiler().profile(name, event_type, metadata):
                    return func(*args, **kwargs)
        return wrapper
    return decorator