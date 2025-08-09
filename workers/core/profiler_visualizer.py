#!/usr/bin/env python3
"""
Profiler Visualizer for generating timeline visualizations.
Creates interactive HTML timeline charts from profiler data.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import io

# Try to import matplotlib for advanced visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ProfilerVisualizer:
    """Generate timeline visualizations from profiler data."""
    
    def __init__(self, profiler_data: Dict[str, Any]):
        """
        Initialize visualizer with profiler data.
        
        Args:
            profiler_data: Dictionary containing 'summary' and 'events' from profiler export
        """
        self.summary = profiler_data.get('summary', {})
        self.events = profiler_data.get('events', [])
        
    def generate_html_timeline(self, output_path: str, title: str = "Rollout Timeline"):
        """
        Generate an interactive HTML timeline visualization.
        
        Args:
            output_path: Path to save the HTML file
            title: Title for the visualization
        """
        # Convert events to timeline format
        timeline_data = self._prepare_timeline_data()
        
        # Calculate layout parameters
        if not timeline_data:
            html_content = self._generate_empty_timeline(title)
        else:
            html_content = self._generate_timeline_html(timeline_data, title)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _prepare_timeline_data(self) -> List[Dict[str, Any]]:
        """Prepare events for timeline visualization."""
        timeline_data = []
        
        if not self.events:
            return timeline_data
        
        # Find the earliest start time
        start_time = min(event['start_time'] for event in self.events if event.get('start_time'))
        
        # Process events
        for event in self.events:
            if not event.get('duration'):
                continue
                
            timeline_data.append({
                'name': event['name'],
                'type': event['event_type'],
                'start': event['start_time'] - start_time,
                'duration': event['duration'],
                'metadata': event.get('metadata', {}),
                'event_id': event.get('event_id', '')
            })
        
        # Sort by start time
        timeline_data.sort(key=lambda x: x['start'])
        
        return timeline_data
    
    def _generate_empty_timeline(self, title: str) -> str:
        """Generate HTML for empty timeline."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .empty-message {{
            text-align: center;
            color: #666;
            padding: 40px;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="empty-message">
            No events recorded in this profiling session.
        </div>
    </div>
</body>
</html>"""
    
    def _generate_timeline_html(self, timeline_data: List[Dict[str, Any]], title: str) -> str:
        """Generate the full HTML timeline visualization."""
        # Event type colors
        event_colors = {
            'llm_call': '#FF6B6B',
            'tool_execution': '#4ECDC4',
            'action_parsing': '#45B7D1',
            'file_read': '#96CEB4',
            'file_write': '#DDA0DD',
            'bash_command': '#F4A460',
            'search_operation': '#98D8C8',
            'k8s_command': '#F7DC6F',
            'pod_operation': '#F8C471',
            'trajectory_step': '#BB8FCE',
            'thought_generation': '#85C1E2',
            'network_io': '#F1948A',
            'disk_io': '#82E0AA',
            'custom': '#D7BDE2'
        }
        
        # Calculate timeline dimensions
        total_duration = self.summary.get('total_duration', 0)
        if total_duration == 0 and timeline_data:
            total_duration = max(event['start'] + event['duration'] for event in timeline_data)
        
        # Generate JSON data for JavaScript
        timeline_json = json.dumps(timeline_data)
        event_colors_json = json.dumps(event_colors)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 30px;
            color: #666;
        }}
        .summary-value {{
            font-weight: bold;
            color: #333;
        }}
        #timeline {{
            position: relative;
            width: 100%;
            overflow-x: auto;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .timeline-canvas {{
            position: relative;
            min-height: 400px;
            width: 100%;
            min-width: 1200px;
        }}
        .event-bar {{
            position: absolute;
            height: 30px;
            border-radius: 3px;
            cursor: pointer;
            transition: opacity 0.2s;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            padding: 5px;
            font-size: 12px;
            color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        .event-bar:hover {{
            opacity: 0.8;
            z-index: 10;
        }}
        .time-axis {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 40px;
            border-top: 1px solid #ddd;
            background-color: #f5f5f5;
        }}
        .time-marker {{
            position: absolute;
            bottom: 0;
            height: 40px;
            border-left: 1px solid #ccc;
            font-size: 11px;
            color: #666;
            padding-left: 3px;
            padding-top: 10px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 13px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        .tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
            pointer-events: none;
            display: none;
            max-width: 300px;
        }}
        .tooltip-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .tooltip-item {{
            margin: 2px 0;
        }}
        .controls {{
            margin-bottom: 15px;
        }}
        .control-button {{
            padding: 5px 15px;
            margin-right: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 13px;
        }}
        .control-button:hover {{
            background-color: #45a049;
        }}
        .control-button.active {{
            background-color: #2196F3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="summary">
            <span class="summary-item">
                Total Duration: <span class="summary-value">{total_duration:.2f}s</span>
            </span>
            <span class="summary-item">
                Total Events: <span class="summary-value">{len(timeline_data)}</span>
            </span>
            <span class="summary-item">
                Start Time: <span class="summary-value">{datetime.fromtimestamp(self.summary.get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}</span>
            </span>
        </div>
        
        <div class="controls">
            <button class="control-button" onclick="zoomIn()">Zoom In</button>
            <button class="control-button" onclick="zoomOut()">Zoom Out</button>
            <button class="control-button" onclick="resetZoom()">Reset</button>
            <button class="control-button" onclick="toggleStacking()">Toggle Stacking</button>
        </div>
        
        <div class="legend" id="legend"></div>
        
        <div id="timeline">
            <div class="timeline-canvas" id="timeline-canvas"></div>
        </div>
        
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        const timelineData = {timeline_json};
        const eventColors = {event_colors_json};
        const totalDuration = {total_duration};
        
        let zoomLevel = 1;
        let stackingEnabled = true;
        
        function formatDuration(seconds) {{
            if (seconds < 0.001) return (seconds * 1000000).toFixed(0) + 'μs';
            if (seconds < 1) return (seconds * 1000).toFixed(2) + 'ms';
            if (seconds < 60) return seconds.toFixed(2) + 's';
            const minutes = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(2);
            return minutes + 'm ' + secs + 's';
        }}
        
        function createLegend() {{
            const legendEl = document.getElementById('legend');
            const eventTypes = new Set(timelineData.map(e => e.type));
            
            eventTypes.forEach(type => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                
                const color = document.createElement('div');
                color.className = 'legend-color';
                color.style.backgroundColor = eventColors[type] || '#999';
                
                const label = document.createElement('span');
                label.textContent = type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                
                item.appendChild(color);
                item.appendChild(label);
                legendEl.appendChild(item);
            }});
        }}
        
        function renderTimeline() {{
            const canvas = document.getElementById('timeline-canvas');
            canvas.innerHTML = '';
            
            const canvasWidth = canvas.offsetWidth * zoomLevel;
            canvas.style.width = canvasWidth + 'px';
            
            // Calculate positions
            const pixelsPerSecond = canvasWidth / totalDuration;
            const barHeight = 30;
            const barMargin = 5;
            
            // Group events by overlapping time for stacking
            const lanes = [];
            
            timelineData.forEach(event => {{
                const startX = event.start * pixelsPerSecond;
                const width = Math.max(event.duration * pixelsPerSecond, 2); // Minimum width of 2px
                
                // Find a lane for this event
                let laneIndex = 0;
                if (stackingEnabled) {{
                    for (let i = 0; i < lanes.length; i++) {{
                        const lane = lanes[i];
                        const lastEvent = lane[lane.length - 1];
                        if (lastEvent.endX + 2 < startX) {{
                            laneIndex = i;
                            break;
                        }}
                    }}
                    if (laneIndex === lanes.length || lanes.length === 0) {{
                        lanes.push([]);
                        laneIndex = lanes.length - 1;
                    }}
                }}
                
                // Create event bar
                const bar = document.createElement('div');
                bar.className = 'event-bar';
                bar.style.left = startX + 'px';
                bar.style.top = (laneIndex * (barHeight + barMargin) + 50) + 'px';
                bar.style.width = width + 'px';
                bar.style.backgroundColor = eventColors[event.type] || '#999';
                bar.textContent = event.name;
                
                // Store event data
                bar.dataset.event = JSON.stringify(event);
                event.endX = startX + width;
                
                if (stackingEnabled && lanes[laneIndex]) {{
                    lanes[laneIndex].push(event);
                }}
                
                // Add event listeners
                bar.addEventListener('mouseenter', showTooltip);
                bar.addEventListener('mouseleave', hideTooltip);
                
                canvas.appendChild(bar);
            }});
            
            // Update canvas height based on lanes
            const canvasHeight = stackingEnabled ? 
                (lanes.length * (barHeight + barMargin) + 100) : 
                (timelineData.length * (barHeight + barMargin) + 100);
            canvas.style.height = canvasHeight + 'px';
            
            // Add time axis
            renderTimeAxis(pixelsPerSecond);
        }}
        
        function renderTimeAxis(pixelsPerSecond) {{
            const canvas = document.getElementById('timeline-canvas');
            
            // Create time axis container
            const axis = document.createElement('div');
            axis.className = 'time-axis';
            
            // Calculate time markers
            const markerInterval = totalDuration > 60 ? 10 : totalDuration > 10 ? 5 : 1;
            
            for (let time = 0; time <= totalDuration; time += markerInterval) {{
                const marker = document.createElement('div');
                marker.className = 'time-marker';
                marker.style.left = (time * pixelsPerSecond) + 'px';
                marker.textContent = formatDuration(time);
                axis.appendChild(marker);
            }}
            
            canvas.appendChild(axis);
        }}
        
        function showTooltip(e) {{
            const event = JSON.parse(e.target.dataset.event);
            const tooltip = document.getElementById('tooltip');
            
            let content = `<div class="tooltip-title">${{event.name}}</div>`;
            content += `<div class="tooltip-item">Type: ${{event.type}}</div>`;
            content += `<div class="tooltip-item">Start: ${{formatDuration(event.start)}}</div>`;
            content += `<div class="tooltip-item">Duration: ${{formatDuration(event.duration)}}</div>`;
            
            if (event.metadata && Object.keys(event.metadata).length > 0) {{
                content += `<div class="tooltip-item" style="margin-top: 5px;">Metadata:</div>`;
                for (const [key, value] of Object.entries(event.metadata)) {{
                    let displayValue = value;
                    // Handle different value types
                    if (typeof value === 'object' && value !== null) {{
                        // For objects and arrays, show formatted JSON
                        try {{
                            displayValue = JSON.stringify(value, null, 2);
                            // If it's too long, truncate it
                            if (displayValue.length > 200) {{
                                displayValue = displayValue.substring(0, 200) + '...';
                            }}
                            // Escape HTML and preserve formatting
                            displayValue = '<pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">' + 
                                         displayValue.replace(/</g, '&lt;').replace(/>/g, '&gt;') + 
                                         '</pre>';
                        }} catch (e) {{
                            displayValue = String(value);
                        }}
                    }} else if (typeof value === 'string') {{
                        // Escape HTML for strings
                        displayValue = value.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        // If it's too long, truncate it
                        if (displayValue.length > 100) {{
                            displayValue = displayValue.substring(0, 100) + '...';
                        }}
                    }}
                    content += `<div class="tooltip-item" style="margin-left: 10px;">${{key}}: ${{displayValue}}</div>`;
                }}
            }}
            
            tooltip.innerHTML = content;
            tooltip.style.display = 'block';
            
            // Position tooltip
            const rect = e.target.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();
            
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
            let top = rect.top - tooltipRect.height - 10;
            
            // Keep tooltip within viewport
            if (left < 10) left = 10;
            if (left + tooltipRect.width > window.innerWidth - 10) {{
                left = window.innerWidth - tooltipRect.width - 10;
            }}
            if (top < 10) {{
                top = rect.bottom + 10;
            }}
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }}
        
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}
        
        function zoomIn() {{
            zoomLevel = Math.min(zoomLevel * 1.5, 10);
            renderTimeline();
        }}
        
        function zoomOut() {{
            zoomLevel = Math.max(zoomLevel / 1.5, 0.5);
            renderTimeline();
        }}
        
        function resetZoom() {{
            zoomLevel = 1;
            renderTimeline();
        }}
        
        function toggleStacking() {{
            stackingEnabled = !stackingEnabled;
            renderTimeline();
        }}
        
        // Initialize
        createLegend();
        renderTimeline();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            renderTimeline();
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def generate_matplotlib_timeline(self, output_path: str, title: str = "Rollout Timeline"):
        """
        Generate a static timeline visualization using matplotlib.
        
        Args:
            output_path: Path to save the image file (PNG)
            title: Title for the visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for generating static timeline images")
        
        timeline_data = self._prepare_timeline_data()
        if not timeline_data:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No events recorded', ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            fig, ax = self._create_matplotlib_timeline(timeline_data, title)
        
        # Save figure
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _create_matplotlib_timeline(self, timeline_data: List[Dict[str, Any]], title: str):
        """Create matplotlib timeline figure."""
        # Event type colors
        event_colors = {
            'llm_call': '#FF6B6B',
            'tool_execution': '#4ECDC4',
            'action_parsing': '#45B7D1',
            'file_read': '#96CEB4',
            'file_write': '#DDA0DD',
            'bash_command': '#F4A460',
            'search_operation': '#98D8C8',
            'k8s_command': '#F7DC6F',
            'pod_operation': '#F8C471',
            'trajectory_step': '#BB8FCE',
            'thought_generation': '#85C1E2',
            'network_io': '#F1948A',
            'disk_io': '#82E0AA',
            'custom': '#D7BDE2'
        }
        
        # Calculate figure size and layout
        n_events = len(timeline_data)
        fig_height = max(6, n_events * 0.3)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        # Set title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Plot events
        y_positions = {}
        current_y = 0
        lane_ends = {}  # Track end time for each lane
        
        for event in timeline_data:
            # Find available lane
            lane_found = False
            for lane_y, end_time in lane_ends.items():
                if event['start'] > end_time + 0.01:  # Small gap between events
                    y_positions[event['event_id']] = lane_y
                    lane_ends[lane_y] = event['start'] + event['duration']
                    lane_found = True
                    break
            
            if not lane_found:
                y_positions[event['event_id']] = current_y
                lane_ends[current_y] = event['start'] + event['duration']
                current_y += 1
            
            # Draw rectangle for event
            y = y_positions[event['event_id']]
            rect = Rectangle(
                (event['start'], y - 0.4),
                event['duration'],
                0.8,
                facecolor=event_colors.get(event['type'], '#999'),
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Add event name if there's enough space
            if event['duration'] > 0.5:  # Only show text for longer events
                ax.text(
                    event['start'] + event['duration'] / 2,
                    y,
                    event['name'][:20] + '...' if len(event['name']) > 20 else event['name'],
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    weight='bold'
                )
        
        # Set axis limits and labels
        total_duration = self.summary.get('total_duration', 
                                        max(e['start'] + e['duration'] for e in timeline_data))
        ax.set_xlim(0, total_duration * 1.05)
        ax.set_ylim(-1, max(y_positions.values()) + 1)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Execution Lane', fontsize=12)
        
        # Create legend
        event_types = list(set(e['type'] for e in timeline_data))
        legend_elements = [
            mpatches.Patch(
                color=event_colors.get(etype, '#999'),
                label=etype.replace('_', ' ').title()
            )
            for etype in sorted(event_types)
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Grid
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Summary text
        summary_text = (
            f"Total Duration: {total_duration:.2f}s | "
            f"Total Events: {len(timeline_data)}"
        )
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, ax


def create_timeline_from_file(json_path: str, output_path: str, format: str = 'html'):
    """
    Create a timeline visualization from a profiler JSON export file.
    
    Args:
        json_path: Path to the profiler JSON export
        output_path: Path to save the visualization
        format: Output format ('html' or 'png')
    """
    with open(json_path, 'r') as f:
        profiler_data = json.load(f)
    
    visualizer = ProfilerVisualizer(profiler_data)
    
    if format == 'html':
        visualizer.generate_html_timeline(output_path)
    elif format == 'png':
        visualizer.generate_matplotlib_timeline(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html' or 'png'")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python profiler_visualizer.py <input.json> <output.html|output.png>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if output_file.endswith('.png'):
        format = 'png'
    else:
        format = 'html'
    
    create_timeline_from_file(input_file, output_file, format)
    print(f"Timeline visualization saved to: {output_file}")