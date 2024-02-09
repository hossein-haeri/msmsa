from time import gmtime, strftime
from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# %matplotlib qt
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
# import msmsa_plus_v2
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
    
    # results = np.zeros([len(data_X),4])
    # valid_model = None
    # y = 0
    pred_y = 0
    # params = []
    update_info_list = []
    validation_mae_list = []
    y_pred_list = []
    y_list = []
    val_horizon_list = []
    for k, (X, y) in enumerate(zip(data_X, data_y)):
        try:
            pred_y = online_model.predict_online_model(X)[0]
            # print('predication succeeded')
        except:
            # if prediction fails, use the previous prediction
            pred_y = pred_y
            # print('predication failed')

        # pred_y = float(pred_y)
        validation_mae = np.absolute(y - pred_y)
        update_info = online_model.update_online_model(X, y)
        
        validation_mae_list.append(validation_mae)
        y_pred_list.append(pred_y)
        y_list.append(y)
        val_horizon_list = online_model.get_val_horizon()

    y_rescaled = rescale(y_list, scaler_y)
    pred_y_rescaled = rescale(y_pred_list, scaler_y)
    # mae_inv = np.absolute(y_inv - pred_y_inv)
    # y_bar_inv = np.mean(y_inv)

    run_summary = { 'dataset': dataset,
                    'stream_size': len(data_y),
                    'method': online_model.method_name,
                    'learning_model': type(model.model).__name__,
                    'MAE': np.mean(np.absolute(y_rescaled - pred_y_rescaled)),
                    'memory_len': np.mean(val_horizon_list),
                    # 'noise_var': dataset_configs['noise_var'],
                    # 'STD': np.std(update_info_list),
                    # 'RMSE': np.sqrt(np.mean(update_info_list**2)),
                    # 'RRSE': np.sqrt(np.sum(update_info_list**2)/np.sum((data_y - np.mean(data_y))**2)),
                    'TargetMean': np.mean(y_rescaled),
                    'TargetSTD': np.std(y_rescaled),
                    # 'MeanValidityHorizon': np.mean(update_info_list),
                }
    return run_summary

wandb_log = False
wandb_logrun = False
pickle_log = True

datasets = [
            # 'Bike (daily)',
            # 'Bike (hourly)',
            # 'Household energy',
            # 'Melbourn housing',
            # 'Air quality',
            # 'Friction',
            # 'NYC taxi',
            # 'Teconer',
            'Metro'
                ]

# datasets = ['Hyper-A',
#             'Hyper-I',
#             'Hyper-G',
#             'Hyper-LN',
#             'Hyper-RW',
#             'Hyper-GU'
#                ]


# dataset_configs = {'noise_var': 0,
#                    'stream_size': 1_000,
#                    'drift_prob':0.01,
#                    'dim': 10}

dataset_configs = {'noise_var':     None,
                   'stream_size':   None,
                   'drift_prob':    None,
                   'dim':           None}

# datasets = ['Household energy']
# dataset_name = datasets[1]

# model = learning_models.Linear()
# model = learning_models.DecissionTree()
# # model = learning_models.KNN()
# # model = learning_models.SVReg()
# # model = learning_models.Polynomial()
base_learners = [
            learning_models.Linear(),
            learning_models.DecissionTree(),
            # learning_models.SVReg()
        ]

# noise_vars = [0, 1, 2, 3, 4, 5]
noise_vars = ['-1']
num_monte = 1

logs = pd.DataFrame()
for monte in tqdm(range(num_monte)):
    # print('------ NUMBER OF MONTE SIMS: ', monte, '/', num_monte)
    for base_learner in tqdm(base_learners, leave=False):
        for dataset in tqdm(datasets, leave=False):
            for noise_var in tqdm(noise_vars, leave=False):
                dataset_configs['noise_var'] = noise_var
                online_models = [
                            # msmsa_plus.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8),
                            # aue.AUE(min_memory_len=10, batch_size=20),
                            msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8),
                            # davar_reg.DAVAR(lam=10),
                            kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                            adwin_reg.ADWIN(delta=0.002),
                            ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            naive_reg.Naive()
                            ]
                for online_model in online_models:
                    online_model.base_learner = base_learner

                for online_model in tqdm(online_models, leave=False):
                    run_summary = run(    
                            online_model=online_model,
                            dataset=dataset,
                            model=base_learner,
                            dataset_configs=dataset_configs,
                            )
                    print(run_summary)
                    # use pd concat to append run_summary to logs
                    logs = pd.concat([logs, pd.DataFrame([run_summary])], ignore_index=True)
                    
                    if wandb_log:
                        wandb.log(run_summary)
                        wandb.finish(quiet=True)

    # pickle the logs every 10 monte sims
    if pickle_log and monte % 1 == 0:
        # with open('teconer_100K.pkl', 'wb') as f:
        with open('metro.pkl', 'wb') as f:
            pickle.dump(logs, f)


    