import fire
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import matplotlib.animation as animation
import numpy as np
from typing import List

# Function to process the file and return a dataframe with summed data
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    tokens_latency = []
    tokens_amount = []
    elapsed_seconds_list = []

    for record in data:
        tokens_latency.extend([value for sublist in record['tokens_latency'] for value in sublist])
        tokens_amount.extend([value for sublist in record['tokens_amount'] for value in sublist])
        elapsed_seconds_list.extend([round(record['elapsed_seconds'], 2)] * len([value for sublist in record['tokens_latency'] for value in sublist]))

    df = pd.DataFrame({
        'Token Amount': tokens_amount[:len(tokens_latency)],  # Align lengths
        'Latency (s)': tokens_latency,
        'Elapsed Seconds': elapsed_seconds_list
    })

    # Summing Token Amount for each Elapsed Seconds
    df_sum = df.groupby('Elapsed Seconds', as_index=False).agg({'Token Amount': 'sum', 'Latency (s)': 'mean'})
    
    return df_sum

# Function to generate and save the 3D animation
def generate_animation(files: List[str], labels: List[str], output: str = "3d_plot_animation.mp4"):
    if len(files) != len(labels):
        raise ValueError("The number of files must match the number of labels.")

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta']  # Extendable color list

    # Create the figure for 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the initial state of the plot
    for i, file_path in enumerate(files):
        df_sum = process_file(file_path)
        color = colors[i % len(colors)]  # Cycle through colors if more than defined
        ax.scatter(df_sum['Elapsed Seconds'], df_sum['Latency (s)'], df_sum['Token Amount'], alpha=0.7, color=color, label=labels[i])

    ax.set_xlabel('Elapsed Seconds')
    ax.set_ylabel('Latency (s)')
    ax.set_zlabel('Tokens Amount')
    ax.set_title('3D Animation of Elapsed Seconds, Latency (s), and Token Amount')
    ax.legend()

    # Animation function: rotate the view
    def update_view(frame):
        ax.view_init(elev=20, azim=frame)

    # Set up the animation
    ani = animation.FuncAnimation(fig, update_view, frames=np.arange(0, 360, 1), interval=50)

    # Save the animation as an mp4 file
    ani.save(output, writer='ffmpeg', fps=30)
    print(f"Animation saved as {output}")

# Fire command to allow CLI execution
if __name__ == "__main__":
    fire.Fire(generate_animation)

