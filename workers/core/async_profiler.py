#!/usr/bin/env python3
"""
Async-safe Performance Profiler for Agentic Rollout Library.
This version uses asyncio locks instead of threading locks.
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
import uuid

from .profiler import EventType, ProfileEvent, EVENT_COLORS


class AsyncRolloutProfiler:
    """Async-safe performance profiler for tracking rollout execution."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the async profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.events: List[ProfileEvent] = []
        self.active_events: Dict[str, ProfileEvent] = {}
        self.start_time = time.time()
        self._lock = asyncio.Lock()  # Use asyncio lock instead of threading lock
    
    @asynccontextmanager
    async def profile_async(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
        """
        Async context manager for profiling an operation.
        
        Args:
            name: Name of the operation
            event_type: Type of event
            metadata: Additional metadata to store
        """
        if not self.enabled:
            yield
            return
        
        event = await self._start_event_async(name, event_type, metadata)
        try:
            yield event
        finally:
            await self._end_event_async(event)
    
    async def start_event(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None) -> str:
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
        
        event = await self._start_event_async(name, event_type, metadata)
        return event.event_id
    
    async def end_event(self, event_id: str):
        """
        End a manually tracked event.
        
        Args:
            event_id: ID of the event to end
        """
        if not self.enabled or not event_id:
            return
        
        async with self._lock:
            if event_id in self.active_events:
                event = self.active_events.pop(event_id)
                await self._end_event_async(event)
    
    async def _start_event_async(self, name: str, event_type: EventType, metadata: Optional[Dict[str, Any]] = None) -> ProfileEvent:
        """Internal async method to start an event."""
        event = ProfileEvent(
            name=name,
            event_type=event_type,
            start_time=time.time(),
            metadata=metadata or {},
            event_id=f"{time.time()}_{uuid.uuid4().hex[:8]}"
        )
        
        async with self._lock:
            self.active_events[event.event_id] = event
        
        return event
    
    async def _end_event_async(self, event: ProfileEvent):
        """Internal async method to end an event."""
        event.complete()
        async with self._lock:
            self.events.append(event)
            # Remove from active events if still there
            self.active_events.pop(event.event_id, None)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of profiled events.
        This method is synchronous as it only reads data.
        
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
        This method is synchronous as it only writes data.
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "summary": self.get_summary(),
            "events": [event.to_dict() for event in self.events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def reset(self):
        """Reset the profiler."""
        async with self._lock:
            self.events.clear()
            self.active_events.clear()
            self.start_time = time.time()
    
    def get_timeline_data(self) -> List[Dict[str, Any]]:
        """
        Get timeline data for visualization.
        This method is synchronous as it only reads data.
        
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


# Global async profiler instance
_global_async_profiler: Optional[AsyncRolloutProfiler] = None


def get_async_profiler() -> AsyncRolloutProfiler:
    """Get the global async profiler instance."""
    global _global_async_profiler
    if _global_async_profiler is None:
        _global_async_profiler = AsyncRolloutProfiler()
    return _global_async_profiler


def set_async_profiler(profiler: AsyncRolloutProfiler):
    """Set the global async profiler instance."""
    global _global_async_profiler
    _global_async_profiler = profiler
