#!/usr/bin/env python3
"""
Thread-safe profiler wrapper that isolates profiling from external library calls.
This helps prevent deadlocks when profiling code that uses its own threading/locking.
"""

import time
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager
import queue
import logging

from .profiler import RolloutProfiler, EventType, ProfileEvent

logger = logging.getLogger(__name__)


class SafeProfiler:
    """
    A wrapper around RolloutProfiler that isolates profiling operations
    to prevent deadlocks with external libraries.
    """
    
    def __init__(self, enabled: bool = True):
        """Initialize the safe profiler."""
        self.enabled = enabled
        self._profiler = RolloutProfiler(enabled=enabled)
        self._pending_ends = queue.Queue()
        self._active_events = {}
        self._lock = threading.RLock()
    
    def start_event(self, name: str, event_type, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an event.
        
        This method is safe to call even when external libraries might have their own locks.
        """
        if not self.enabled:
            return ""
        
        try:
            # Handle both EventType enum and string
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except (ValueError, KeyError):
                    event_type = EventType.CUSTOM
            
            # Create event directly without going through the profiler's lock
            event = ProfileEvent(
                name=name,
                event_type=event_type,
                start_time=time.time(),
                metadata=metadata or {}
            )
            
            with self._lock:
                self._active_events[event.event_id] = event
            
            return event.event_id
        except Exception as e:
            logger.warning(f"Failed to start profiling event {name}: {e}")
            return ""
    
    def end_event(self, event_id: str):
        """
        End a tracked event.
        
        This method queues the end operation to avoid lock contention.
        """
        if not self.enabled or not event_id:
            return
        
        try:
            with self._lock:
                if event_id in self._active_events:
                    event = self._active_events.pop(event_id)
                    event.complete()
                    # Add completed event directly to profiler's events list
                    self._profiler.events.append(event)
        except Exception as e:
            logger.warning(f"Failed to end profiling event {event_id}: {e}")
    
    def end_event_async(self, event_id: str):
        """
        Queue an event to be ended asynchronously.
        
        This is useful when you need to end an event but don't want to
        risk any blocking operations.
        """
        if not self.enabled or not event_id:
            return
        
        self._pending_ends.put(event_id)
    
    def flush_pending_ends(self):
        """Process any pending event ends."""
        while not self._pending_ends.empty():
            try:
                event_id = self._pending_ends.get_nowait()
                self.end_event(event_id)
            except queue.Empty:
                break
            except Exception as e:
                logger.warning(f"Failed to flush pending end: {e}")
    
    @contextmanager
    def profile(self, name: str, event_type, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling an operation.
        
        This is safe to use even around external library calls.
        """
        if not self.enabled:
            yield
            return
        
        event_id = self.start_event(name, event_type, metadata)
        try:
            yield event_id
        finally:
            self.end_event(event_id)
    
    @asynccontextmanager
    async def profile_async(self, name: str, event_type, metadata: Optional[Dict[str, Any]] = None):
        """
        Async context manager for profiling an asynchronous operation.
        
        This is safe to use even around external library calls.
        """
        if not self.enabled:
            yield
            return
        
        event_id = self.start_event(name, event_type, metadata)
        try:
            yield event_id
        finally:
            self.end_event(event_id)
    
    @property
    def events(self):
        """Get the list of completed events."""
        return self._profiler.events
    
    def get_summary(self):
        """Get profiling summary."""
        self.flush_pending_ends()  # Ensure all pending events are processed
        return self._profiler.get_summary()
    
    def export_events(self, filepath: str):
        """Export events to a file."""
        self.flush_pending_ends()  # Ensure all pending events are processed
        return self._profiler.export_events(filepath)


def create_safe_profiler(enabled: bool = True) -> SafeProfiler:
    """
    Factory function to create a safe profiler instance.
    
    Args:
        enabled: Whether profiling should be enabled
        
    Returns:
        A SafeProfiler instance
    """
    return SafeProfiler(enabled=enabled)