import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSONL file
file_path = './api_monitor.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Extract data for all records
tokens_latency = []
tokens_amount = []
elapsed_seconds_list = []

for record in data:
    tokens_latency.extend([value for sublist in record['tokens_latency'] for value in sublist])
    tokens_amount.extend([value for sublist in record['tokens_amount'] for value in sublist])
    elapsed_seconds_list.extend([round(record['elapsed_seconds'], 2)] * len([value for sublist in record['tokens_latency'] for value in sublist]))

# Create a DataFrame for all records
df = pd.DataFrame({
    'Token Amount': tokens_amount[:len(tokens_latency)],  # Align lengths
    'Latency (s)': tokens_latency,
    'Elapsed Seconds': elapsed_seconds_list
})

# We need to sum the `Token Amount` for each `Elapsed Seconds` after flattening the data and aggregating
df_sum = df.groupby('Elapsed Seconds', as_index=False).agg({'Token Amount': 'sum', 'Latency (s)': 'mean'})

# Plot 1: Elapsed Seconds vs Token Amount (Summed for each Elapsed Seconds)
plt.figure(figsize=(10, 6))
plt.scatter(df_sum['Elapsed Seconds'], df_sum['Token Amount'], alpha=0.7)
plt.title('Elapsed Seconds \ Token Amount (Summed for each Elapsed Seconds)')
plt.xlabel('Elapsed Seconds')
plt.ylabel('Tokens Amount')
plt.grid(True)
plt.show()

# Plot 2: Elapsed Seconds vs Tokens Latency (Summed for each Elapsed Seconds)
plt.figure(figsize=(10, 6))
plt.scatter(df_sum['Elapsed Seconds'], df_sum['Latency (s)'], alpha=0.7)
plt.title('Elapsed Seconds \ Tokens Latency (Summed for each Elapsed Seconds)')
plt.xlabel('Elapsed Seconds')
plt.ylabel('Tokens Latency (s)')
plt.grid(True)
plt.show()

# Plot 3: Tokens Latency vs Token Amount (Summed for each Elapsed Seconds)
plt.figure(figsize=(10, 6))
plt.scatter(df_sum['Latency (s)'], df_sum['Token Amount'], alpha=0.7)
plt.title('Tokens Latency \ Token Amount (Summed for each Elapsed Seconds)')
plt.xlabel('Tokens Latency (s)')
plt.ylabel('Tokens Amount')
plt.grid(True)
plt.show()

# Plot 4: 3D plot (Elapsed Seconds, Latency (s), Tokens) (Summed for each Elapsed Seconds)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_sum['Elapsed Seconds'], df_sum['Latency (s)'], df_sum['Token Amount'], alpha=0.7)
ax.set_title('Elapsed Seconds, Latency (s) and Token Amount (Summed for each Elapsed Seconds) (3D)')
ax.set_xlabel('Elapsed Seconds')
ax.set_ylabel('Latency (s)')
ax.set_zlabel('Tokens Amount')
plt.show()

