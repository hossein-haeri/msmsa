import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import sys
import time


# List of dataset names
datasets = [
            # 'Hyper-HT',
            'Hyper-ND',

            # 'Hyper-A',
            # 'Hyper-I',
            # 'Hyper-G',
            # 'Hyper-LN',
            # 'Hyper-RW',
            # 'Hyper-GU',
              
            # 'Bike (daily)',
            # 'Bike (hourly)',
            # 'Household energy',
            # 'Melbourne housing',
            # 'Air quality',

            # 'Friction',
            # 'NYC taxi',
            # 'Teconer_100K',clea
            # 'Teconer_10K'
]

# Tag for the run
tag = 'test3'

# List of methods
methods = [
            # 'DTH',
            'KSWIN',
            'MSMSA',
            # 'ADWIN',
            # 'PH',
            # 'DDM',
            # 'Naive',
            ]

# List of base learners
base_learners = ['LNR']

verbose = False

wandb_log = True

repetitions = 50

# initial seed
initial_seed = 1100



# Function to run the command silently
def run_simulation(dataset, method, base_learner, seed, wandb_log, tag):
    command = f'python sim_runner_v3.py "{dataset}" {method} {base_learner} {seed} {wandb_log} {tag}'
    if verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Finished: {dataset}, {method}, {base_learner}, seed:{seed}")
    # time.sleep(0.1)

# Execute all combinations in parallel
def execute_round(seed):
    with ThreadPoolExecutor() as executor:

        # Create a list of all tasks for the executor
        tasks = []
        # Number of repetitions
        
        for i in range(repetitions):
            seed += 1
            for dataset in datasets:
                for method in methods:
                    for base_learner in base_learners:
                        print(f"Starting run {i+1} with seed {seed}")
                        tasks.append(executor.submit(run_simulation, dataset, method, base_learner, seed, wandb_log, tag))
                        # print(f"Finished repetition {i+1}")
        # Wait for all tasks in the current round to complete
        for future in tasks:
            future.result()  # This line will block until the individual task is completed



execute_round(initial_seed)

