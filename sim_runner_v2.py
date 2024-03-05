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
from utility.logger import Logger, Plotter


# class Logger:
#     def __init__(self):
#         y = []
#         y_pred = []
#         val_horizon = []
#         mae = []


def rescale(y, scaler):
    return np.array(y)
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))


def run(model, online_model, dataset_name, synthetic_param):
        
    data_X, data_y, scaler_X, scaler_y = load_dataset(dataset_name, synthetic_param)

    logger = Logger()
    # if 'Teconer_' in dataset_name:
    #     online_model.anchor_samples = data_X

    # log = {'y': [], 'y_pred': [], 'error': [], 'update_info': [], 'val_horizon': []}
    y_pred = 0

    if 'MSMSA' in online_model.method_name:
        online_model.max_horizon = len(data_y)
    for k, (X, y) in enumerate(tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))):
        
        # if X[0] < 0:
        #     y = 0

        try:
            y_pred = online_model.predict_online_model(X)[0]
            # print('predication succeeded')
        except:
            # if prediction fails, use the previous prediction
            y_pred = y_pred
            # print('predication failed')


        
        
        online_model.update_online_model(X, y)
        
        # validity_horizon_list.append(len(online_model.memory))
        if online_model.method_name == 'MSMSA':
            num_train_samples = online_model.validity_horizon
        elif online_model.method_name == 'MSMSA+':
            num_train_samples = np.mean(online_model.validity_horizon)
        else:
            num_train_samples = len(online_model.memory)


        logger.log(y, y_pred, num_train_samples=num_train_samples)

    logger.sclaer_y = scaler_y
    logger.scaler_X = scaler_X
    logger.summary['dataset_name'] = dataset_name
    logger.summary['method_name'] = online_model.method_name
    logger.summary['base_learner_name'] = type(model).__name__
    logger.synthetic_param = synthetic_param
    # logger.method_name = online_model.method_name
    # logger.base_learner_name = type(model).__name__
    # logger.synthetic_param = synthetic_param
    logger.finish()

    return logger

wandb_log = False
wandb_logrun = False
pickle_log = True

synthetic_param = None


# # ################ REAL DATA #################
# datasets = [
#             'Bike (daily)',
#             'Bike (hourly)',
#             'Household energy',
#             'Melbourn housing',
#             'Air quality',
#             # 'Friction',
# #             # 'NYC taxi',
# #             # 'Teconer_100K',
# #             # 'Teconer_10K'
#                 ]
# noise_vars = [-1]


############## SYNTHETIC DATA #################
datasets = [
            'Hyper-A',
            'Hyper-I',
            # 'Hyper-G',
            # 'Hyper-LN',
            # 'Hyper-RW',
            # 'Hyper-GU',
            # 'SimpleHeterogeneous',
               ]

synthetic_param = {'noise_var': None,
                   'stream_size': 1_000,
                   'drift_prob':0.01,
                   'dim': 10}
noise_vars = [0.01]

## noise_vars = [0, 1, 2, 3, 4, 5]

base_learners = [
            # learning_models.Linear(),
            learning_models.DecissionTree(),
            # learning_models.SVReg(),
            # learning_models.NeuralNet()
            # neural_net_base_learner.DNNRegressor()
        ]


num_monte = 3

logs = []
for monte in tqdm(range(num_monte), position=0, leave=True):
    for base_learner in tqdm(base_learners, leave=False, disable=True):
        for dataset_name in tqdm(datasets, leave=False, disable=True):
            for noise_var in tqdm(noise_vars, leave=False, disable=True):
                if synthetic_param is not None:
                    synthetic_param['noise_var'] = noise_var
                online_models = [
                            msmsa_plus.MSMSA_plus(min_memory_len=10, num_anchors=50, lam=.8, max_horizon=1000, continuous_model_fit=True),
                            # aue.AUE(min_memory_len=10, batch_size=20),
                            msmsa.MSMSA(min_memory_len=10, lam=.8, max_horizon=1000, continuous_model_fit=True),
                            # davar_reg.DAVAR(lam=10),
                            # kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                            # adwin_reg.ADWIN(delta=0.002),
                            # ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            # ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            # naive_reg.Naive()
                            ]
                for online_model in online_models:
                    online_model.base_learner = base_learner

                for online_model in tqdm(online_models, leave=False, disable=True):
                # for online_model in tqdm(online_models, leave=False, disable=True):
                
                    log = run(    
                            online_model=online_model,
                            dataset_name=dataset_name,
                            model=base_learner,
                            synthetic_param=synthetic_param,
                            )
                    logs.append(log)



print(logs[0].summary)
# pickle logs
with open('logs.pkl', 'wb') as f:
        pickle.dump(logs, f)

# fig, axs = plt.subplots(4, 1, figsize=(10, 10))
# # plot MAE for every method and every dataset on sns barplot
# axs[0] = sns.barplot(x="dataset_name", y="MAE", data=logs, hue="method", errorbar="sd",ax=axs[0])
# # plot validity horizon for every method and every dataset on sns barplot
# axs[1] = sns.barplot(x="dataset_name", y="MeanValidityHorizon", data=logs, hue="method", errorbar="sd", ax=axs[1])
# # plot 'Error' for every method in a plt.line
# for i, method in enumerate(online_models):
#     axs[2].plot(logs['Error'][i], label=method.method_name, alpha=0.5)
#     axs[3].plot(logs['ValidityHorizon'][i], label=method.method_name)
# # axs[2].set_title('Prediction Error Over Time')
# axs[2].set_xlabel('Time')
# axs[2].set_ylabel('MAE')
# axs[2].legend()
# axs[3].set_xlabel('Time')
# axs[3].set_ylabel('Training Horizon')
# axs[3].legend()
# plt.show()



    