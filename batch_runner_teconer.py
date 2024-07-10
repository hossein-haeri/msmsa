import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import numpy as np
# import os
# import sys
# import time


tag = 'teconer_final_v3'

# List of dataset names
datasets = [
    # 'Teconer_Jan_10K',
    # 'Teconer_Downtown_10K',
    # 'Teconer_Jan_100K',
    # 'Teconer_Jan_1M',
    # 'Teconer_Downtown_100K',
    # 'Teconer_Downtown_1M',
    # 'teconer_helsinki_jan2018_100K'
    # 'teconer_helsinki_jan2018_1M'
    'teconer_helsinki_jan2018'

]

preview_druations = [1*60, 5*60, 10*60, 30*60]

# preview_druations = [1*60]
epsilons = [0.9]
# List of base learners
base_learners = ['DT']
# List of methods
methods = [
    # 'PTMI',
    'TMI',
    # 'KSWIN',
    # 'MSMSA',
    # 'ADWIN',
    # 'PH',
    # 'DDM',
    # 'Naive',
]



verbose = True

wandb_log = True

repetitions = 1

# initial seed
initial_seed = 1000

# Maximum number of scripts running simultaneously
max_running_scripts = 13  # Adjust this value as needed

# Semaphore to limit concurrent executions
semaphore = Semaphore(max_running_scripts)

# Function to run the command silently
def run_simulation(dataset, method, base_learner, epsilon, preview_druation, seed, wandb_log, tag):
    command = f'python sim_runner_teconer_trip_based_v2.py "{dataset}" {method} {base_learner} {epsilon} {preview_druation} {seed} {wandb_log} {tag}'
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
                    for preview_druation in preview_druations:
                        for epsilon in epsilons:
                            for base_learner in base_learners:
                                print(f"Starting run {i+1} with seed {seed}")
                                semaphore.acquire()  # Acquire semaphore before starting a task
                                future = executor.submit(run_simulation, dataset, method, base_learner, epsilon, preview_druation, seed, wandb_log, tag)
                                tasks.append(future)
        
        # Wait for all tasks in the current round to complete
        for future in as_completed(tasks):
            future.result()  # This line will block until the individual task is completed

execute_round(initial_seed)