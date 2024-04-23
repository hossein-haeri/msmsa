import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import sys

# List of dataset names
# wandb_enable = True
datasets = ['Hyper-A', 'Hyper-I', 'Hyper-G', 'Hyper-LN', 'Hyper-RW', 'Hyper-GU']
# datasets = ['Hyper-A']
# datasets = [
#             'Bike (daily)',
#             'Bike (hourly)',
#             'Household energy',
#             'Melbourne housing',
#             'Air quality',
#             # 'Friction',
#             # 'NYC taxi',
#             # 'Teconer_100K',
#             'Teconer_10K'
# ]

# List of methods
methods = ['DTH', 'KSWIN', 'Naive']
# methods = ['DTH']

# List of base learners
base_learners = ['RF']

# Number of repetitions
repetitions = 10

# Function to run the command silently
def run_simulation(dataset, method, base_learner, seed):
    command = f"python sim_runner_v3.py {dataset} {method} {base_learner} {seed}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)

    print(f"Executed: {dataset}, {method}, {base_learner}, seed:{seed}")

# Execute all combinations in parallel
def execute_round(seed):
    with ThreadPoolExecutor() as executor:
        # Create a list of all tasks for the executor
        tasks = []
        for dataset in datasets:
            for method in methods:
                for base_learner in base_learners:
                    tasks.append(executor.submit(run_simulation, dataset, method, base_learner, seed))
        # Wait for all tasks in the current round to complete
        for future in tasks:
            future.result()  # This line will block until the individual task is completed

# Loop for each repetition
for i in range(repetitions):
    seed = np.random.randint(0, 1000)
    print(f"Starting repetition {i+1}")
    execute_round(seed)
    print(f"Finished repetition {i+1}")
