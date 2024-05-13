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
from sklearn.ensemble import RandomForestRegressor

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
import dth



def rescale(y, scaler):
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))


def rescale_features(X, scaler):
    return scaler.inverse_transform(X)


def run(online_model, dataset_name, synthetic_param):
        

    data_X, data_y, scaler_X, scaler_y, trip_ids = load_dataset(dataset_name, synthetic_param)
 
    print('number of unique trips: ', len(np.unique(trip_ids)))
    # if 'Teconer_' in dataset_name:
    #     online_model.anchor_samples = data_X

    predicted_trip_ids = []

    trips = []
    # error_list = []
    num_records = len(data_y)

    # start_from_k = 0
    # end_at_k = 1_000_000
    num_preview_samples = 50_000

    # # trim the data from the start and end
    # data_X = data_X[start_from_k:end_at_k]
    # data_y = data_y[start_from_k:end_at_k]
    # trip_ids = trip_ids[start_from_k:end_at_k]

    # randomly select a subset of the data but keep the order
    # idx = np.random.choice(len(data_y), 1_000_000, replace=False)
    # idx.sort()  # This will sort the indices to maintain the original order
    # data_X = data_X[idx]
    # data_y = data_y[idx]
    # trip_ids = trip_ids[idx]

    stream_bar = tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))
    for k, (X, y) in enumerate(stream_bar):


        X = X.reshape(1,-1)
        
        if trip_ids[k] not in predicted_trip_ids and k <= num_preview_samples:
            # add the sample to the memory and fit the model
            online_model.update_online_model(X, y, fit_base_learner=True)
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
            trips.append([  trip_ids[k],
                            X_trip_rescaled,
                            y_trip_rescaled,
                            pred_trip_rescaled,
                            online_model.get_num_samples(),])
        else:
            # just add the sample to the memory but do not fit the model
            online_model.update_online_model(X, y, fit_base_learner=False)
        stream_bar.set_postfix(MemSize=online_model.X.shape[0], Ratio=(online_model.get_num_samples())/(k+1), NumTrips=len(predicted_trip_ids))

    with open('trips_downtown_full_epsilon6.pkl', 'wb') as f:
    # with open('trips_100K.pkl', 'wb') as f:
        pickle.dump(trips, f)


wandb_log = True
wandb_logrun = False
pickle_log = True


################ REAL DATA #################
datasets = [
            'Teconer_downtown'
            # 'Teconer_full',
            # 'Teconer_100K',
            # 'Teconer_road_piece'
                ]
dataset_configs = {'noise_var':     None,
                   'stream_size':   None,
                   'drift_prob':    None,
                   'dim':           None}
noise_vars = ['-1']




base_learners = [
            RandomForestRegressor(n_estimators=20, max_depth=7, n_jobs=-1, bootstrap=True),
            # learning_models.Linear(),
            # learning_models.DecissionTree(),
            # learning_models.SVReg(),
            # learning_models.NeuralNet()
            # neural_net_base_learner.DNNRegressor()
            # neural_net_base_learner.RegressionNN(    hidden_layers=[50, 50],
                                                                # input_dim=7, 
                                                                # output_dim=1,
                                                                # dropout=0.1, 
                                                                # learning_rate=0.01, 
                                                                # epochs=10)
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
                            # msmsa_plus.MSMSA_plus(min_memory_len=10, update_freq_factor=1, lam=0.8, max_horizon=500, continuous_model_fit=False),
                            # aue.AUE(min_memory_len=10, batch_size=20),
                            # msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8, max_horizon=2000, continuous_model_fit=False),
                            # davar_reg.DAVAR(lam=10),
                            dth.DTH(epsilon=0.6),
                            # kswin_reg.KSWIN(),
                            # adwin_reg.ADWIN(delta=0.002),
                            # ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            # ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            # naive_reg.Naive()
                            
                            ]
                for online_model in online_models:
                    online_model.base_learner = base_learner

                for online_model in tqdm(online_models, leave=False, disable=True):
                # for online_model in tqdm(online_models, leave=False, disable=True):
                
                    run(    
                            online_model=online_model,
                            dataset_name=dataset,
                            synthetic_param=dataset_configs,
                            )



    