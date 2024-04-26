import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import sys
import time
# List of dataset names
# wandb_enable = True
# datasets = ['Hyper-A', 'Hyper-I', 'Hyper-G', 'Hyper-LN', 'Hyper-RW', 'Hyper-GU']
# # datasets = ['Hyper-A']
# # datasets = [
# #             'Bike (daily)',
# #             'Bike (hourly)',
# #             'Household energy',
# #             'Melbourne housing',
# #             'Air quality',
# #             # 'Friction',
# #             # 'NYC taxi',
# #             # 'Teconer_100K',
# #             'Teconer_10K'
# # ]

# # List of methods
# methods = ['DTH']
# # methods = ['DTH']

# # List of base learners
# base_learners = ['RF']
epsilons = [0.7]
priors = [0.4, 0.5, 0.6]
# Number of repetitions
repetitions = 10

# Function to run the command silently
def run_simulation(epsilon, prior, seed):
    command = f"python sim_runner_tunner.py {epsilon} {prior} {seed}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
    print(f"Executed: {epsilon} {prior}, seed:{seed}")
    time.sleep(0.1)

# Execute all combinations in parallel
def execute_round(seed):
    with ThreadPoolExecutor() as executor:
        # Create a list of all tasks for the executor
        tasks = []
        for epsilon in epsilons:
            for prior in priors:
                # for base_learner in base_learners:
                    tasks.append(executor.submit(run_simulation, epsilon, prior, seed))
        # Wait for all tasks in the current round to complete
        for future in tasks:
            future.result()  # This line will block until the individual task is completed

# Loop for each repetition
for i in range(repetitions):
    seed = np.random.randint(0, 1000)
    print(f"Starting repetition {i+1}")
    execute_round(seed)
    print(f"Finished repetition {i+1}")
