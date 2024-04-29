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




def rescale(y, scaler):
    # return np.array(y)
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).squeeze()


def run(online_model_name, base_learner_name, dataset_name, synthetic_param, seed=None):
        
    data_X, data_y, scaler_X, scaler_y, hyper_w = load_dataset(dataset_name, synthetic_param, seed=int(seed))
    if base_learner_name == 'RF':
        base_learner = learning_models.RandomForest(n_estimators=50, bootstrap=True, n_jobs=-1, max_depth=7)
    elif base_learner_name == 'LNR':
        base_learner = learning_models.Linear()
    elif base_learner_name == 'DT':
        base_learner = learning_models.DecissionTree(max_depth=5)
    elif base_learner_name == 'SVR':
        base_learner = learning_models.SVReg()
    elif base_learner_name == 'NN':
        base_learner = learning_models.NeuralNet()
    else:
        print('Base learner not found')


    # num_feautres = 8
    # num_samples = 10000
    num_feautres = len(data_X[0])+1
    num_samples = len(data_X)


    if online_model_name == 'MSMSA':
        online_model = msmsa.MSMSA()
    elif online_model_name == 'MSMSA+':
        online_model = msmsa_plus.MSMSA_plus()
    elif online_model_name == 'DAVAR':
        online_model = davar_reg.DAVAR(lam=10)
    elif online_model_name == 'KSWIN':
        online_model = kswin_reg.KSWIN(max_num_samples=num_samples, num_features=num_feautres)
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
        online_model = dth.DTH(max_num_samples=num_samples, num_features=num_feautres)
    else:
        print('Online learner not found')


    online_model.base_learner = base_learner



    logger = Logger()
    logger.method_name = online_model.method_name

    y_pred = 0

    if 'MSMSA' in online_model.method_name:
        online_model.max_horizon = len(data_y)

    stream_bar = tqdm(zip(data_X, data_y), leave=False, disable=False, total=len(data_y))

    for k, (X, y) in enumerate(stream_bar):

        X_with_time = np.append(k/len(data_y),X).reshape(1,-1)

        y_pred = online_model.predict_online_model(X_with_time)[0]

        online_model.update_online_model(X_with_time, y)
    
        
        # retransform y and y_pred back to original scale
        y = rescale(y, scaler_y)
        y_pred = rescale(y_pred, scaler_y)

        # num_train_samples = len(online_model.samples)
        num_train_samples = online_model.get_num_samples()
        logger.log(y, y_pred, num_train_samples=num_train_samples)
        logger.X.append(X)
        stream_bar.set_postfix(MemSize=num_train_samples, Ratio=(num_train_samples)/(k+1))
        if wandb_log:
            if 'Hyper' in dataset_name:
                wandb.log({
                        'run_abs_error': np.abs(y - y_pred),
                        'run_y': y,
                        'run_y_pred': y_pred,
                        'run_w': hyper_w[k],
                        'run_memory_size': num_train_samples,
                        })
            else:
                wandb.log({'run_abs_error': np.abs(y - y_pred)})
        

    logger.sclaer_y = scaler_y
    logger.scaler_X = scaler_X
    logger.finish()
    return logger



wandb_log = True



# get base_learner_name from the argument of the script
dataset_name = sys.argv[1]
online_model_name = sys.argv[2]
base_learner_name = sys.argv[3]
seed = sys.argv[4]
# if additional argument is given get it as the tag for wandb
# if len(sys.argv) > 5:
#     wandb_run.tags = sys.argv[5:]



if 'Hyper' in dataset_name:
    synthetic_param = {'noise_var': 0.01, # [0, 1, 2, 3, 4, 5]
                       'stream_size': 1_000,
                       'drift_prob':0.01,
                       'dim': 5}
else:
    synthetic_param = None






config={
        "dataset": dataset_name,
        "method": online_model_name,
        "base_learner": base_learner_name,
        "seed": seed,
        "synthetic_params": synthetic_param,
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






    