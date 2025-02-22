# API Monitor Visualization Script
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

class APIMetricsVisualizer:
    def __init__(self, log_file):
        self.log_file = log_file
        # Set style for better visualization
        #plt.style.use('seaborn')
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def read_log_data(self):
        timestamps = []
        chars_per_second = []
        total_chars = []
        active_sessions = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    timestamps.append(datetime.fromisoformat(data['timestamp']))
                    chars_per_second.append(data['chars_per_second'])
                    total_chars.append(data['total_chars'])
                    active_sessions.append(data['active_sessions'])
        
        return timestamps, chars_per_second, total_chars, active_sessions

    def create_visualization(self, output_file='api_metrics.png'):
        # Read data
        timestamps, chars_per_second, total_chars, active_sessions = self.read_log_data()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[1, 1, 1])
        fig.suptitle('API Performance Metrics', fontsize=16, y=0.95)
        
        # Plot 1: Characters per Second
        axes[0].plot(timestamps, chars_per_second, linewidth=2, marker='o', markersize=4)
        axes[0].set_title('Characters per Second')
        axes[0].set_ylabel('Chars/s')
        axes[0].grid(True)

        # Plot 2: Total Characters (Cumulative)
        axes[1].plot(timestamps, total_chars, linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Total Characters')
        axes[1].set_ylabel('Characters')
        axes[1].grid(True)

        # Plot 3: Active Sessions
        axes[2].plot(timestamps, active_sessions, linewidth=2, marker='o', markersize=4)
        axes[2].set_title('Active Sessions')
        axes[2].set_ylabel('Sessions')
        axes[2].grid(True)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Configuration
    log_file = "api_monitor.jsonl"
    output_file = "api_metrics.png"
    
    # Create and run visualizer
    visualizer = APIMetricsVisualizer(log_file)
    visualizer.create_visualization(output_file)
    print(f"Visualization saved as {output_file}")

if __name__ == "__main__":
    main()

"""
Requirements:

1. Input:
   - JSONL file containing API metrics
   - Each line should be a valid JSON object with fields:
     * timestamp: ISO format timestamp
     * chars_per_second: Current character processing rate
     * total_chars: Cumulative characters processed
     * active_sessions: Number of active sessions

2. Output:
   - High-resolution PNG image (300 DPI)
   - Three vertically stacked line charts
   - Professional visualization style

3. Chart Specifications:
   a) Characters per Second:
      - Shows processing rate over time
      - Y-axis: chars/s
      - X-axis: time in HH:MM:SS
      - Includes grid lines and markers

   b) Total Characters:
      - Shows cumulative character count
      - Y-axis: total characters
      - X-axis: time in HH:MM:SS
      - Includes grid lines and markers

   c) Active Sessions:
      - Shows concurrent sessions
      - Y-axis: number of sessions
      - X-axis: time in HH:MM:SS
      - Includes grid lines and markers

4. Style Requirements:
   - Clean, professional appearance
   - Grid lines for readability
   - Time-based X-axis with proper formatting
   - Clear titles and labels
   - Appropriate figure size (15x12 inches)
   - Consistent styling across all charts

5. Dependencies:
   pip install matplotlib seaborn

6. Usage:
   python visualizer.py

7. Features:
   - Automatic scaling of axes
   - Professional color scheme
   - High-resolution output
   - Clear data point markers
   - Proper timestamp handling
"""