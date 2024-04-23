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


def run(model, online_model, dataset_name, synthetic_param, seed=None):
        
    data_X, data_y, scaler_X, scaler_y, hyper_w = load_dataset(dataset_name, synthetic_param, seed=int(seed))

    logger = Logger()
    logger.method_name = online_model.method_name

    y_pred = 0

    if 'MSMSA' in online_model.method_name:
        online_model.max_horizon = len(data_y)

    stream_bar = tqdm(zip(data_X, data_y), leave=False, disable=False, total=len(data_y))

    for k, (X, y) in enumerate(stream_bar):
    # for k, (X, y) in enumerate(stream_bar(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))):
        X_with_time = np.append(k,X).reshape(1,-1)

        # PREDICTION
        y_pred = online_model.predict_online_model(X_with_time)[0]

        # ADD SAMPLE AND UPDATE MODEL
        if 'DTH' in online_model.method_name:
            # online_model.add_sample([X[1:]], y, X[0])
            online_model.add_sample([X], y, k)
            online_model.update_online_model()
        else:
            online_model.update_online_model(X_with_time[0], y)
        
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

        stream_bar.set_postfix(MemSize=num_train_samples, Ratio=(num_train_samples+1)/(k+1))
        
        wandb.log({'run_mae': np.abs(y - y_pred)})
        if 'Hyper' in dataset_name:
            wandb.log({'w': hyper_w})
        

    logger.sclaer_y = scaler_y
    logger.scaler_X = scaler_X
    logger.summary['dataset_name'] = dataset_name
    logger.summary['method_name'] = online_model.method_name
    logger.summary['base_learner_name'] = type(model).__name__
    logger.synthetic_param = synthetic_param
    logger.summary['seed'] = seed
    if 'MSMSA+' in online_model.method_name:
        logger.anchors = online_model.anchors
    logger.finish()
    return logger


wandb_run = wandb.init(project='stream_learning', entity='haeri-hsn')



# get base_learner_name from the argument of the script
dataset_name = sys.argv[1]
online_model_name = sys.argv[2]
base_learner_name = sys.argv[3]
seed = sys.argv[4]
# if additional argument is given get it as the tag for wandb
if len(sys.argv) > 5:
    wandb_run.tags = sys.argv[5:]



if 'Hyper' in dataset_name:
    synthetic_param = {'noise_var': 0.1, # [0, 1, 2, 3, 4, 5]
                       'stream_size': 1_000,
                       'drift_prob':0.01,
                       'dim': 1}
else:
    synthetic_param = None


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
    online_model = dth.DTH(epsilon=.99)
else:
    print('Online learner not found')



online_model.base_learner = base_learner


log = run(
        online_model=online_model,
        dataset_name=dataset_name,
        model=base_learner,
        synthetic_param=synthetic_param,
        seed=seed
        )

print(log.summary)



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

wandb.log(log.summary)
wandb.finish()

# print('done')
# print(log.summary)





    