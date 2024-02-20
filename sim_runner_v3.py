from concurrent.futures import ThreadPoolExecutor
from time import gmtime, strftime
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import stream_generator
import learning_models
from datasets.data_loader import load_dataset
from baselines_v2 import davar_reg
from baselines_v2 import ddm_reg
from baselines_v2 import adwin_reg
from baselines_v2 import kswin_reg 
from baselines_v2 import ph_reg
from baselines_v2 import naive_reg
from baselines_v2 import aue_reg as aue
import msmsa_v2 as msmsa
import wandb
import os

def rescale(y, scaler):
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))

def run(model, online_model, dataset, dataset_configs):
    data_X, data_y, scaler_y = load_dataset(dataset,
                                            hyperplane_dimension=dataset_configs['dim'],
                                            noise_var=dataset_configs['noise_var'],
                                            stream_size=dataset_configs['stream_size'],
                                            drift_probability=dataset_configs['drift_prob'])
    
    pred_y = 0
    update_info_list = []
    validation_mae_list = []
    y_pred_list = []
    y_list = []
    val_horizon_list = []
    
    for k, (X, y) in enumerate(zip(data_X, data_y)):
        try:
            pred_y = online_model.predict_online_model(X)[0]
        except:
            pred_y = pred_y

        validation_mae = np.absolute(y - pred_y)
        update_info = online_model.update_online_model(X, y)
        
        validation_mae_list.append(validation_mae)
        y_pred_list.append(pred_y)
        y_list.append(y)
        val_horizon_list = online_model.get_val_horizon()

    y_rescaled = rescale(y_list, scaler_y)
    pred_y_rescaled = rescale(y_pred_list, scaler_y)

    run_summary = { 'dataset': dataset,
                    'stream_size': len(data_y),
                    'method': online_model.method_name,
                    'learning_model': type(model.model).__name__,
                    'MAE': np.mean(np.absolute(y_rescaled - pred_y_rescaled)),
                    'memory_len': np.mean(val_horizon_list),
                    'TargetMean': np.mean(y_rescaled),
                    'TargetSTD': np.std(y_rescaled),
                }
    return run_summary

wandb_log = False
wandb_logrun = False
pickle_log = True

datasets = [
            'Bike (daily)',
            # 'Bike (hourly)',
            # 'Household energy',
            # 'Melbourn housing',
            # 'Air quality',
            # 'Friction',
            # 'NYC taxi',
            # 'Teconer',
            # 'Metro'
                ]

dataset_configs = {'noise_var':     None,
                   'stream_size':   None,
                   'drift_prob':    None,
                   'dim':           None}

base_learners = [
            learning_models.Linear(),
            learning_models.DecissionTree(),
        ]

noise_vars = ['-1']
num_monte = 1

logs = pd.DataFrame()

def run_simulation(monte, base_learner, dataset, noise_var, online_models, dataset_configs):
    logs_local = pd.DataFrame()
    for online_model in online_models:
        online_model.base_learner = base_learner
        run_summary = run(online_model=online_model,
                          dataset=dataset,
                          model=base_learner,
                          dataset_configs=dataset_configs)
        logs_local = pd.concat([logs_local, pd.DataFrame([run_summary])], ignore_index=True)
        if wandb_log:
            wandb.log(run_summary)
            wandb.finish(quiet=True)
    return logs_local

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = []
    for monte in range(num_monte):
        for base_learner in base_learners:
            for dataset in datasets:
                for noise_var in noise_vars:
                    dataset_configs['noise_var'] = noise_var
                    online_models = [
                                msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8),
                                kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                                adwin_reg.ADWIN(delta=0.002),
                                ddm_reg.DDM(alpha_w=2, alpha_d=3),
                                ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                                naive_reg.Naive()
                                ]
                    futures.append(executor.submit(run_simulation, monte, base_learner, dataset, noise_var, online_models, dataset_configs))
    for future in tqdm(futures):
        logs = pd.concat([logs, future.result()], ignore_index=True)
        # print(logs)

# pickle the logs
if pickle_log:
    with open('bike.pkl', 'wb') as f:
        pickle.dump(logs, f)
