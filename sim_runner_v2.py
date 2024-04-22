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
import dth
import neural_net_base_learner
import wandb
import os
from utility.utilities import Logger, Plotter


# class Logger:
#     def __init__(self):
#         y = []
#         y_pred = []
#         val_horizon = []
#         mae = []


def rescale(y, scaler):
    # return np.array(y)
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).squeeze()


def run(model, online_model, dataset_name, synthetic_param):
        
    data_X, data_y, scaler_X, scaler_y = load_dataset(dataset_name, synthetic_param)

    logger = Logger()
    logger.method_name = online_model.method_name

    y_pred = 0

    if 'MSMSA' in online_model.method_name:
        online_model.max_horizon = len(data_y)

    pbar = tqdm(zip(data_X, data_y), leave=False, disable=False, total=len(data_y))

    for k, (X, y) in enumerate(pbar):
    # for k, (X, y) in enumerate(pbar(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))):
        X_agg = np.append(X,k)

        # PREDICTION
        if 'DTH' in online_model.method_name:
            # print((len(online_model.memory))) 
            y_pred = online_model.predict_online_model([X], k)[0]
        else:
            y_pred = online_model.predict_online_model(X_agg)[0]

        

        # ADD SAMPLE AND UPDATE MODEL
        if 'DTH' in online_model.method_name:
            # online_model.add_sample([X[1:]], y, X[0])
            online_model.add_sample([X], y, k)
            online_model.update_online_model()
        else:
            online_model.update_online_model(X_agg, y)
        
        # LOGGING
        if online_model.method_name == 'MSMSA':
            num_train_samples = online_model.validity_horizon
        elif online_model.method_name == 'MSMSA+':
            num_train_samples = np.mean(online_model.validity_horizon)
            logger.val_hor.append(online_model.validity_horizon)
        else:
            num_train_samples = len(online_model.memory)

        # retransform y and y_pred back to original scale
        y = rescale(y, scaler_y)
        y_pred = rescale(y_pred, scaler_y)

        logger.log(y, y_pred, num_train_samples=num_train_samples)
        logger.X.append(X)

        pbar.set_postfix(MemSize=num_train_samples, Ratio=(num_train_samples+1)/(k+1))
        
        

    logger.sclaer_y = scaler_y
    logger.scaler_X = scaler_X
    logger.summary['dataset_name'] = dataset_name
    logger.summary['method_name'] = online_model.method_name
    logger.summary['base_learner_name'] = type(model).__name__
    logger.synthetic_param = synthetic_param
    if 'MSMSA+' in online_model.method_name:
        logger.anchors = online_model.anchors
    # logger.method_name = online_model.method_name
    # logger.base_learner_name = type(model).__name__
    # logger.synthetic_param = synthetic_param
    logger.finish()

    return logger

wandb_log = False
wandb_logrun = False
pickle_log = True

synthetic_param = None


# ################ REAL DATA #################
datasets = [
            # 'Bike (daily)',
            # 'Bike (hourly)',
            # 'Household energy',
            # 'Melbourne housing',
            # 'Air quality',
            # 'Friction',
            'NYC taxi',
#             # 'Teconer_100K',
            # 'Teconer_10K'
                ]
noise_vars = [-1]


############# SYNTHETIC DATA #################
# datasets = [
#             'Hyper-A',
#             # 'Hyper-I',
#             # 'Hyper-G',
#             # 'Hyper-LN',
#             # 'Hyper-RW',
#             # 'Hyper-GU',
#             # 'SimpleHeterogeneous',
#                ]

# synthetic_param = {'noise_var': None,
#                    'stream_size': 1_000,
#                    'drift_prob':0.01,
#                    'dim': 10}
# noise_vars = [0.1]

## noise_vars = [0, 1, 2, 3, 4, 5]

base_learners = [
            # learning_models.Linear(),
            # learning_models.DecissionTree(max_depth=5),
            learning_models.RandomForest(n_estimators=10, max_depth=5),
            # learning_models.SVReg(),
            # learning_models.NeuralNet()
            # neural_net_base_learner.DNNRegressor()
        ]


num_monte = 5

logs = []
for monte in tqdm(range(num_monte), position=0, leave=True):
    for base_learner in tqdm(base_learners, leave=False, disable=True):
        for dataset_name in tqdm(datasets, leave=False, disable=True):
            for noise_var in tqdm(noise_vars, leave=False, disable=True):
                if synthetic_param is not None:
                    synthetic_param['noise_var'] = noise_var
                online_models = [
                            dth.DTH(    epsilon=0.9,
                                        num_sub_learners=0,
                                        min_new_samples_for_base_learner_update=1,
                                        min_new_samples_for_pruining=1,
                                        multi_threading_sub_learners=False,
                                        pruning_disabled=False,
                                        num_pruning_threads=1,
                                        max_elimination=100,
                                        use_sublearners_as_baselearner=False,
                                        max_investigated_samples = 1000),
                            # msmsa_plus.MSMSA_plus(min_memory_len=10, num_anchors=50, lam=.8, max_horizon=1000, continuous_model_fit=True),
                            # aue.AUE(min_memory_len=10, batch_size=20),
                            # msmsa.MSMSA(min_memory_len=10, lam=.8, max_horizon=1000, continuous_model_fit=True),
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
                
                    log = run(    
                            online_model=online_model,
                            dataset_name=dataset_name,
                            model=base_learner,
                            synthetic_param=synthetic_param,
                            )
                    logs.append(log)
                    print(log.summary)
# pickle logs
with open('logs_.pkl', 'wb') as f:
        pickle.dump(logs, f)

pltr = Plotter()
pltr.plot_loggers(logs)





    