import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import numpy as np
import os
import sys
import time

# List of dataset names
datasets = [
    # 'Hyper-RG',
    # 'Hyper-HT',
    # 'Hyper-ND',
    # 'Hyper-A',
    # 'Hyper-I',
    # 'Hyper-G',
    # 'Hyper-LN',
    # 'Hyper-RW',
    # 'Hyper-GU',
    'Bike (daily)',
    'Bike (hourly)',
    'Household energy',
    'Melbourne housing',
    'Air quality',
    # 'Friction',
    # 'NYC taxi',
    # 'Teconer_100K',
    'Teconer_10K'
]

# Tag for the run
# tag = 'final_hyper_msmsa'
# tag = 'final_real_msmsa'
# tag = 'final_sythetic_dth_v2'
# tag = 'final_real_dth_v2'

# tag = 'final_regional_drift'
# tag = 'final_real'

tag = 'tmi_vs_ptmi_v2'


# List of methods
methods = [
    'PTMI',
    'TMI',
    # 'KSWIN',
    # 'MSMSA',
    # 'ADWIN',
    # 'PH',
    # 'DDM',
    # 'Naive',
]

# List of base learners
base_learners = ['RF']

verbose = True

wandb_log = True

repetitions = 1

# initial seed
initial_seed = 1000

# Maximum number of scripts running simultaneously
max_running_scripts = 5  # Adjust this value as needed

# Semaphore to limit concurrent executions
semaphore = Semaphore(max_running_scripts)

# Function to run the command silently
def run_simulation(dataset, method, base_learner, seed, wandb_log, tag):
    command = f'python sim_runner_v3.py "{dataset}" {method} {base_learner} {seed} {wandb_log} {tag}'
    if verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Finished: {dataset}, {method}, {base_learner}, seed:{seed}")
    # time.sleep(0.1)
    semaphore.release()

# Execute all combinations in parallel
def execute_round(seed):
    with ThreadPoolExecutor() as executor:
        # Create a list of all tasks for the executor
        tasks = []
        for i in range(repetitions):
            seed += 1
            for dataset in datasets:
                for method in methods:
                    for base_learner in base_learners:
                        print(f"Starting run {i+1} with seed {seed}")
                        semaphore.acquire()  # Acquire semaphore before starting a task
                        future = executor.submit(run_simulation, dataset, method, base_learner, seed, wandb_log, tag)
                        tasks.append(future)
        
        # Wait for all tasks in the current round to complete
        for future in as_completed(tasks):
            future.result()  # This line will block until the individual task is completed

execute_round(initial_seed)