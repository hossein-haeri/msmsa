from time import gmtime, strftime
from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
# %matplotlib qt
import seaborn as sns
from scipy.ndimage import gaussian_filter


# import stream_generator
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
import msmsa_plus_v2 as msmsa_plus
import neural_net_base_learner
import wandb
import os


def rescale(y, scaler):
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))


def run(model, online_model, dataset, dataset_configs):
        
    data_X, data_y, scaler_X, scaler_y = load_dataset(dataset,
                                                hyperplane_dimension=dataset_configs['dim'],
                                                noise_var=dataset_configs['noise_var'],
                                                stream_size=dataset_configs['stream_size'],
                                                drift_probability=dataset_configs['drift_prob'])
    

    if 'Teconer_' in dataset:
        online_model.anchor_samples = data_X

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
    validity_horizon_list = []
    # error_list = []
    for k, (X, y) in enumerate(tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))):
        
        if X[0] < 0:
            y = 0

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
        # error_list.append(np.absolute(y - pred_y))

        # validity_horizon_list.append(len(online_model.memory))
        if online_model.method_name == 'MSMSA':
            validity_horizon_list.append(online_model.validity_horizon)
        elif online_model.method_name == 'MSMSA+':
            validity_horizon_list.append(np.mean(online_model.validity_horizon))
        else:
            validity_horizon_list.append(len(online_model.memory))

    y_rescaled = rescale(y_list, scaler_y)
    # y_rescaled = y_list
    pred_y_rescaled = rescale(y_pred_list, scaler_y)
    error_list = np.absolute(y_rescaled - pred_y_rescaled)
    # mae_inv = np.absolute(y_inv - pred_y_inv)
    # y_bar_inv = np.mean(y_inv)

    
    run_summary = { 'dataset': dataset,
                    'stream_size': len(data_y),
                    'method': online_model.method_name,
                    'learning_model': type(model.model).__name__,
                    'MAE': np.mean(error_list),
                    # 'noise_var': dataset_configs['noise_var'],
                    # 'STD': np.std(update_info_list),
                    # 'RMSE': np.sqrt(np.mean(update_info_list**2)),
                    # 'RRSE': np.sqrt(np.sum(update_info_list**2)/np.sum((data_y - np.mean(data_y))**2)),
                    'TargetMean': np.mean(y_rescaled),
                    'TargetSTD': np.std(y_rescaled),
                    'Error': error_list,
                    'Predictions': pred_y_rescaled,
                    'ValidityHorizon': validity_horizon_list,
                    'MeanValidityHorizon': np.mean(validity_horizon_list),
                }
    # if dataset == 'Teconer':
    #     return run_summary, pred_y_rescaled
    return run_summary, pred_y_rescaled, validity_horizon_list

wandb_log = False
wandb_logrun = False
pickle_log = True


# ################ REAL DATA #################
# datasets = [
#             # 'Bike (daily)',
#             'Bike (hourly)',
#             'Household energy',
#             'Melbourn housing',
#             # 'Air quality',
#             # 'Friction',
#             # 'NYC taxi',
#             # 'Teconer_100K',
#             # 'Teconer_10K'
#                 ]
# dataset_configs = {'noise_var':     None,
#                    'stream_size':   None,
#                    'drift_prob':    None,
#                    'dim':           None}
# noise_vars = ['-1']


################# SYNTHETIC DATA #################
datasets = ['Hyper-A',
            'Hyper-I',
            'Hyper-G',
            'Hyper-LN',
            'Hyper-RW',
            'Hyper-GU'
               ]

dataset_configs = {'noise_var': None,
                   'stream_size': 1_000,
                   'drift_prob':0.01,
                   'dim': 10}
noise_vars = [2]


# datasets = ['Household energy']
# dataset_name = datasets[1]

# model = learning_models.Linear()
# model = learning_models.DecissionTree()
# # model = learning_models.KNN()
# # model = learning_models.SVReg()
# # model = learning_models.Polynomial()

base_learners = [
            # learning_models.Linear(),
            learning_models.DecissionTree(),
            # learning_models.SVReg(),
            # learning_models.NeuralNet()
            # neural_net_base_learner.DNNRegressor()
        ]

# noise_vars = [0, 1, 2, 3, 4, 5]


num_monte = 1

logs = pd.DataFrame()
for monte in tqdm(range(num_monte), position=0, leave=True):
    # print('------ NUMBER OF MONTE SIMS: ', monte, '/', num_monte)
    for base_learner in tqdm(base_learners, leave=False, disable=True):
        for dataset in tqdm(datasets, leave=False, disable=True):
            for noise_var in tqdm(noise_vars, leave=False, disable=True):
                dataset_configs['noise_var'] = noise_var
                online_models = [
                            msmsa_plus.MSMSA_plus(min_memory_len=10, update_freq_factor=1, lam=0.8, max_horizon=500, continuous_model_fit=False),
                            # aue.AUE(min_memory_len=10, batch_size=20),
                            msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8, max_horizon=500, continuous_model_fit=False),
                            # davar_reg.DAVAR(lam=10),
                            kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                            # adwin_reg.ADWIN(delta=0.002),
                            # ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            # ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            # naive_reg.Naive()
                            ]
                for online_model in online_models:
                    online_model.base_learner = base_learner

                for online_model in tqdm(online_models, leave=False, disable=True):
                # for online_model in tqdm(online_models, leave=False, disable=True):
                
                    (run_summary, predictions, val_horizon) = run(    
                            online_model=online_model,
                            dataset=dataset,
                            model=base_learner,
                            dataset_configs=dataset_configs,
                            )
                    # print run_summary fields of interest
                    # print(
                    #     'Learning Model:', run_summary['learning_model'],
                    #     'Dataset:', run_summary['dataset'],
                    #     'Method:', run_summary['method'],
                    #     'MAE:', run_summary['MAE']
                    #       )
                          
                    # print(run_summary)
                    # use pd concat to append run_summary to logs
                    logs = pd.concat([logs, pd.DataFrame([run_summary])], ignore_index=True)
                    
                    if wandb_log:
                        wandb.log(run_summary)
                        wandb.finish(quiet=True)

# pickle the logs every 10 monte sims
# if pickle_log:
#     with open(dataset+'_run_summary.pkl', 'wb') as f:
#         pickle.dump(logs, f)
#     with open(dataset+'_predictions.pkl', 'wb') as f:
#         pickle.dump(predictions, f)
#     with open(dataset+'_val_horizon.pkl', 'wb') as f:
#         pickle.dump(val_horizon, f)


# make 3 subplots

fig, axs = plt.subplots(1, 2, figsize=(10, 10))

# plot MAE for every method and every dataset on sns barplot
# sns.set_theme(style="whitegrid")
axs[0] = sns.barplot(x="dataset", y="MAE", data=logs, hue="method", errorbar="sd",ax=axs[0])

# plot validity horizon for every method and every dataset on sns barplot
axs[1] = sns.barplot(x="dataset", y="MeanValidityHorizon", data=logs, hue="method", errorbar="sd", ax=axs[1])
plt.show()



    