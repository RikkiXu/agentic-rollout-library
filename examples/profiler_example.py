#!/usr/bin/env python3
"""
Example demonstrating the performance profiler with GeneralAgent.

This example shows how to:
1. Enable profiling for agent rollouts
2. Track performance of different operations
3. Generate timeline visualizations
4. Analyze performance bottlenecks
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.agents.general_agent import GeneralAgent, dump_trajectory
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.core.profiler import RolloutProfiler, EventType
from workers.core.profiler_visualizer import ProfilerVisualizer


async def main():
    """Run a profiled agent example."""
    
    # Create a profiler instance
    profiler = RolloutProfiler(enabled=True)
    
    # Configure LLM client (using environment variables)
    api_key = os.getenv("LLM_API_KEY", "your-api-key")
    base_url = os.getenv("LLM_BASE_URL", "your-base-url")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
    
    llm_client = create_llm_client(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        debug=False
    )
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor", {}),
        "file_editor": create_tool("FileEditor", {}),
        "search": create_tool("Search", {}),
    }
    
    # Create agent with profiler
    agent = GeneralAgent(
        max_rounds=10,
        debug=False,
        profiler=profiler,  # Pass the profiler instance
        termination_tool_names=["finish"],
        system_prompt="""
You are an AI assistant helping with coding tasks. You have access to:
- bash_executor: Execute bash commands
- file_editor: Create, read, and edit files
- search: Search for text in files

Analyze problems step by step and use the appropriate tools to solve them.
Call finish() when you're done.
"""
    )
    
    # Set tools
    agent.set_tools(tools)
    
    # Example task
    prompt = """
Create a simple Python script that:
1. Creates a file called 'hello.py' with a function that prints "Hello, World!"
2. Tests that the file runs correctly
3. Search for the word "Hello" in the file to confirm it was created
"""
    
    print("🚀 Starting profiled agent execution...")
    print(f"Task: {prompt}\n")
    
    # Run the agent with profiling
    trajectory = await agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=llm_client.generate,
        request_id="profiler_demo_001"
    )
    
    print(f"\n✅ Agent execution completed!")
    print(f"Total steps: {len(trajectory.steps)}")
    print(f"Completed: {trajectory.is_completed}")
    
    # Save trajectory
    trajectory_file = "profiler_demo_trajectory.jsonl"
    dump_trajectory(trajectory, trajectory_file, format="jsonl")
    print(f"\n📄 Trajectory saved to: {trajectory_file}")
    
    # Export profiler data
    profiler_data_file = "profiler_demo_data.json"
    profiler.export_events(profiler_data_file)
    print(f"📊 Profiler data saved to: {profiler_data_file}")
    
    # Generate timeline visualization
    timeline_file = "profiler_demo_timeline.html"
    visualizer = ProfilerVisualizer({
        "summary": profiler.get_summary(),
        "events": [event.to_dict() for event in profiler.events]
    })
    visualizer.generate_html_timeline(timeline_file, title="Agent Execution Timeline")
    print(f"📈 Timeline visualization saved to: {timeline_file}")
    
    # Print performance summary
    print("\n📊 Performance Summary:")
    summary = profiler.get_summary()
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Total Events: {summary['event_count']}")
    
    print("\n📋 Events by Type:")
    for event_type, stats in summary['events_by_type'].items():
        print(f"\n  {event_type}:")
        print(f"    Count: {stats['count']}")
        print(f"    Total Duration: {stats['total_duration']:.3f}s")
        print(f"    Average Duration: {stats['avg_duration']:.3f}s")
        if stats['count'] > 1:
            print(f"    Min Duration: {stats['min_duration']:.3f}s")
            print(f"    Max Duration: {stats['max_duration']:.3f}s")
    
    # Analyze performance bottlenecks
    print("\n🔍 Performance Analysis:")
    
    # Find slowest operations
    all_events = [(e, e.duration) for e in profiler.events if e.duration]
    slowest_events = sorted(all_events, key=lambda x: x[1], reverse=True)[:5]
    
    print("\n  Top 5 Slowest Operations:")
    for event, duration in slowest_events:
        print(f"    - {event.name} ({event.event_type.value}): {duration:.3f}s")
    
    # Calculate time spent in different categories
    category_times = {}
    for event in profiler.events:
        if event.duration:
            category = event.event_type.value
            if category not in category_times:
                category_times[category] = 0
            category_times[category] += event.duration
    
    print("\n  Time Distribution:")
    total_tracked_time = sum(category_times.values())
    for category, time_spent in sorted(category_times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_spent / total_tracked_time) * 100 if total_tracked_time > 0 else 0
        print(f"    - {category}: {time_spent:.3f}s ({percentage:.1f}%)")
    
    print(f"\n🎉 Profiling complete! Open {timeline_file} in a browser to see the interactive timeline.")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
