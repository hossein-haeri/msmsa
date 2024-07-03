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
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
import msmsa_v3 as msmsa
import msmsa_plus_v2 as msmsa_plus
import neural_net_base_learner
import wandb
import os
import temporal_model_inference as tmi



def rescale(y, scaler):
    return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))


def rescale_features(X, scaler):
    return scaler.inverse_transform(X)


def run(online_models, dataset_name):
        

    data_X, data_y, trip_ids = load_dataset(dataset_name)

    for online_model in online_models:
        if 'MSMSA' in online_model.method_name:
            online_model.max_horizon = len(data_y)
            data_X_without_time = data_X[:, 1:]
            online_model.initialize_anchors(data_X_without_time)
    print('number of unique trips: ', len(np.unique(trip_ids)))
    # if 'Teconer_' in dataset_name:
    #     online_model.anchor_samples = data_X

    predicted_trip_ids = []

    trips = []
    # error_list = []
    num_records = len(data_y)

    # start_from_k = 0
    # end_at_k = 1_000_000
    # num_preview_samples = 80_000

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

        if k%1 == 0:
            pass
        else:
            continue

        X = X.reshape(1,-1)
        X_without_time = X[:, 1:]
        
        if trip_ids[k] not in predicted_trip_ids:
            # add the sample to the memory and fit the model
            for online_model in online_models:
                if 'TMI' in online_model.method_name:
                    online_model.update_online_model(X, y, fit_base_learner=True)
                else:
                    online_model.update_online_model(X_without_time, y, fit_base_learner=True)

            # build X_trip and y_trip from data_X and data_y where X[0] == trip_id
            X_trip = data_X[trip_ids == trip_ids[k]]
            X_trip_without_time = X_trip[:, 1:]
            y_trip = data_y[trip_ids == trip_ids[k]]

            pred_trip = []

            # predict the whole trip
            for online_model in online_models:
                if 'TMI' in online_model.method_name:
                    pred_trip.append(online_model.predict_online_model(X_trip))
                else:
                    pred_trip.append(online_model.predict_online_model(X_trip_without_time))
            predicted_trip_ids.append(trip_ids[k])
            num_train_samples = []
            for online_model in online_models:
                num_train_samples.append(online_model.get_num_samples())

            trips.append([  trip_ids[k],
                            X_trip,
                            y_trip,
                            pred_trip,
                            num_train_samples
                            ])
        else:
            # just add the sample to the memory but do not fit the model
            for online_model in online_models:
                if 'TMI' in online_model.method_name:
                    online_model.update_online_model(X, y, fit_base_learner=False)
                else:
                    online_model.update_online_model(X_without_time, y, fit_base_learner=False)
        # if len(online_models) > 1:
        #     stream_bar.set_postfix(MemSize=online_models[0].X.shape[0], KeepRatio=(online_models[0].get_num_samples())/(k+1), NumTrips=len(predicted_trip_ids))
        # else:
        stream_bar.set_postfix(NumTrips=len(predicted_trip_ids))
    print(len(trips))
    with open('trips_100K_all.pkl', 'wb') as f:
    # with open('trips_downtown_full.pkl', 'wb') as f:
    # with open('trips_100K.pkl', 'wb') as f:
        pickle.dump(trips, f)



pickle_log = True



dataset = 'Teconer_100K'
            # 'Teconer_100K'
            # 'Teconer_downtown'
            # 'Teconer_full',
            # 'Teconer_100K',
            # 'Teconer_1M',
            # 'Teconer_road_piece'
                # ]

online_models = [
                    tmi.TMI(epsilon=0.9),
                    tmi.TMI(probabilistic_prediction='ensemble', epsilon=0.9),
                    msmsa.MSMSA(),
                    kswin_reg.KSWIN(),
                    adwin_reg.ADWIN(),
                    ddm_reg.DDM(),
                    ph_reg.PH(),
                    naive_reg.Naive(),

                 
                 ]
for online_model in online_models:
    online_model.base_learner = RandomForestRegressor(n_estimators=50, max_depth=7, n_jobs=4, bootstrap=True, max_samples=.9)
    # online_model.base_learner = DecisionTreeRegressor(max_depth=5)


run(online_models=online_models, dataset_name=dataset)