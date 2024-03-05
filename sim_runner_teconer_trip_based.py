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

def rescale_features(X, scaler):
    return scaler.inverse_transform(X)



def run(model, online_model, dataset_name, synthetic_param):
        

    data_X, data_y, scaler_X, scaler_y, trip_ids = load_dataset(dataset_name, synthetic_param)
    

    if 'Teconer_' in dataset_name:
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

    predicted_trip_ids = []

    trips = []
    # error_list = []
    num_records = len(data_y)

    for k, (X, y) in enumerate(tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))):
        

        if k < num_records - 1e5:
            online_model.add_sample(X, y)
            continue

        if online_model.method_name == 'Naive':
            if trip_ids[k] not in predicted_trip_ids and k > num_records - 1e5:
                update_info = online_model.update_online_model(X, y)
            else:
                online_model.add_sample(X, y)

        else:
            update_info = online_model.update_online_model(X, y)

        

        

        if trip_ids[k] not in predicted_trip_ids:


            # build X_trip and y_trip from data_X and data_y where X[0] == trip_id
            X_trip = data_X[trip_ids == trip_ids[k]]
            y_trip = data_y[trip_ids == trip_ids[k]]

            
            # predict the whole trip
            pred_trip = online_model.predict_online_model(X_trip)
            # append the predicted trip to the predictions list
            predicted_trip_ids.append(trip_ids[k])
            y_trip_rescaled = rescale(y_trip, scaler_y)
            pred_trip_rescaled = rescale(pred_trip, scaler_y)
            X_trip_rescaled = rescale_features(X_trip, scaler_X)
            trips.append([trip_ids[k], X_trip_rescaled, y_trip_rescaled, pred_trip_rescaled])

         
    with open('trips_road_piece_msmsa.pkl', 'wb') as f:
        pickle.dump(trips, f)


wandb_log = False
wandb_logrun = False
pickle_log = True


################ REAL DATA #################
datasets = [
            # 'Teconer_full',
            # 'Teconer_10K',
            'Teconer_road_piece'
                ]
dataset_configs = {'noise_var':     None,
                   'stream_size':   None,
                   'drift_prob':    None,
                   'dim':           None}
noise_vars = ['-1']




base_learners = [
            # learning_models.Linear(),
            learning_models.DecissionTree(),
            # learning_models.SVReg(),
            # learning_models.NeuralNet()
            # neural_net_base_learner.DNNRegressor()
        ]


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
                            msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8, max_horizon=2000, continuous_model_fit=False),
                            # davar_reg.DAVAR(lam=10),
                            kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                            # adwin_reg.ADWIN(delta=0.002),
                            # ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            # ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            naive_reg.Naive()
                            ]
                for online_model in online_models:
                    online_model.base_learner = base_learner

                for online_model in tqdm(online_models, leave=False, disable=True):
                # for online_model in tqdm(online_models, leave=False, disable=True):
                
                    run(    
                            online_model=online_model,
                            dataset_name=dataset,
                            model=base_learner,
                            synthetic_param=dataset_configs,
                            )



    