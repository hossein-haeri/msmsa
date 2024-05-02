from time import gmtime, strftime
from tqdm import tqdm
import sys
import os

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
from utility.utilities import Logger, Plotter


from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def rescale(y, scaler):
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).squeeze()


def run(online_model_name, base_learner_name, dataset_name, synthetic_param, seed=None):
    
    data_X, data_y, scaler_X, scaler_y, hyper_w = load_dataset(dataset_name, synthetic_param, seed=int(seed))

    if base_learner_name == 'RF':
        # base_learner = learning_models.RandomForest(n_estimators=20, bootstrap=True, n_jobs=-1, max_depth=7)
        base_learner = RandomForestRegressor(n_estimators=20, max_depth=7, n_jobs=-1, bootstrap=True)
    elif base_learner_name == 'LNR':
        # base_learner = learning_models.Linear()
        base_learner = Ridge(alpha=0.1, fit_intercept = True)
    elif base_learner_name == 'DT':
        # base_learner = learning_models.DecissionTree(max_depth=7)
        base_learner = DecisionTreeRegressor(max_depth=7)
    elif base_learner_name == 'SVR':
        # base_learner = learning_models.SVReg()
        base_learner = SVR(kernel='rbf', C=10, gamma=0.3, epsilon=.1)
    elif base_learner_name == 'NN':
        # base_learner = learning_models.NeuralNet()
        base_learner = neural_net_base_learner.RegressionNN(    hidden_layers=[50, 50],
                                                                input_dim=data_X.shape[1]+1, 
                                                                output_dim=1,
                                                                dropout=0.1, 
                                                                learning_rate=0.01, 
                                                                epochs=10)
    else:
        print('Base learner not found')


    if online_model_name == 'MSMSA':
        online_model = msmsa.MSMSA()
    elif online_model_name == 'MSMSA+':
        online_model = msmsa_plus.MSMSA_plus()
    elif online_model_name == 'DAVAR':
        online_model = davar_reg.DAVAR(lam=10)
    elif online_model_name == 'KSWIN':
        online_model = kswin_reg.KSWIN()
    elif online_model_name == 'ADWIN':
        online_model = adwin_reg.ADWIN()
    elif online_model_name == 'DDM':
        online_model = ddm_reg.DDM()
    elif online_model_name == 'PH':
        online_model = ph_reg.PH()
    elif online_model_name == 'Naive':
        online_model = naive_reg.Naive()
    elif online_model_name == 'AUE':
        online_model = aue.AUE(min_memory_len=10, batch_size=20)
    elif online_model_name == 'DTH':
        online_model = dth.DTH()
    else:
        print('Online learner not found')

    online_model.base_learner = base_learner

    logger = Logger()

    y_pred = 0

    if 'MSMSA' in online_model.method_name:
        online_model.max_horizon = len(data_y)

    stream_bar = tqdm(zip(data_X, data_y), leave=False, disable=False, total=len(data_y))

    for k, (X, y) in enumerate(stream_bar):

        X_with_time = np.append(k/len(data_y),X).reshape(1,-1)

        y_pred = online_model.predict_online_model(X_with_time)[0]

        if k%100 == 0:
            online_model.update_online_model(X_with_time, y, fit_base_learner=True)
        else:
            online_model.update_online_model(X_with_time, y, fit_base_learner=False)
    
        # retransform y and y_pred back to original scale
        y = rescale(y, scaler_y)
        y_pred = rescale(y_pred, scaler_y)

        # num_train_samples = len(online_model.samples)
        num_train_samples = online_model.get_num_samples()
        logger.log(y, y_pred, num_train_samples=num_train_samples)
        logger.X.append(X)
        stream_bar.set_postfix(MemSize=online_model.X.shape[0], KeepRatio=(num_train_samples)/(k+1))
        if wandb_log:
            if 'Teconer' in dataset_name:
                wandb.log({
                            'run_abs_error': np.abs(y - y_pred),
                            'run_y': y,
                            'run_y_pred': y_pred,
                            'num_train_samples': num_train_samples,
                            })
            elif 'Hyper' in dataset_name:
                wandb.log({
                            'run_abs_error': np.abs(y - y_pred),
                            'run_y': y,
                            'run_y_pred': y_pred,
                            'run_w': hyper_w[k],
                            'num_train_samples': num_train_samples,
                            })
            else:
                wandb.log({
                            'run_abs_error': np.abs(y - y_pred),
                            'run_y': y,
                            'run_y_pred': y_pred,
                            'num_train_samples': num_train_samples,
                            })
            # wandb.log({'run_abs_error': np.abs(y - y_pred)})
        

    logger.sclaer_y = scaler_y
    logger.scaler_X = scaler_X
    logger.method_name = online_model.method_name
    logger.hyperparams = online_model.hyperparams
    logger.summary['mean_memory_size'] = np.mean(logger.num_train_samples_list)
    logger.summary['MAE'] = np.mean(logger.errors)
    logger.summary['RMSE'] = np.sqrt(np.mean(np.square(logger.errors)))
    logger.summary['MAPE'] = np.mean(np.abs(np.array(logger.errors) / np.array(logger.y)))
    logger.summary['R2'] = 1 - np.sum(np.square(logger.errors)) / np.sum(np.square(np.array(logger.y) - np.mean(logger.y)))

    # logger.finish()
    
    return logger



wandb_log = True



# get base_learner_name from the argument of the script
dataset_name = sys.argv[1]
online_model_name = sys.argv[2]
base_learner_name = sys.argv[3]
seed = sys.argv[4]
if len(sys.argv) > 5:
    tag = sys.argv[5]



if 'Hyper' in dataset_name:
    synthetic_param = {'noise_var': 0.01, # [0, 1, 2, 3, 4, 5]
                       'stream_size': 1_000,
                       'drift_prob':0.005,
                       'dim': 5}
else:
    synthetic_param = None






config={
        "dataset": dataset_name,
        "method": online_model_name,
        "base_learner": base_learner_name,
        "seed": seed,
        "synthetic_params": synthetic_param,
        "tag": tag,
        # "hyperparams": online_model.hyperparams,
    }

if wandb_log:
        wandb_run = wandb.init(project='stream_learning', entity='haeri-hsn', config=config)


# print(config)
log = run(
        online_model_name=online_model_name,
        dataset_name=dataset_name,
        synthetic_param=synthetic_param,
        base_learner_name=base_learner_name,
        seed=seed
        )



# config['summary'] = log.summary
config.update(log.summary)
config.update(log.hyperparams)

# print(config)
if wandb_log:
    # pickle logs
    directory = 'pickled_logs'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # File path for the pickle file
    file_path = os.path.join(directory, str(wandb_run.name) + '.pkl')
    with open(file_path, 'wb') as f:
            pickle.dump(log, f)


    # pltr = Plotter()
    # pltr.plot_loggers(log)

    # log into wandb
    # wandb.log(config)
    wandb.config.update(config)
    wandb.finish()






    