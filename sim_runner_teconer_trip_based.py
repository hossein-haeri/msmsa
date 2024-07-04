from time import gmtime, strftime
from tqdm import tqdm
import sys
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

def make_trip_id_dict(trip_ids):
    # Get the unique trip_ids
    unique_trip_ids = np.unique(trip_ids)
    # Create a dictionary to hold the boolean vectors
    trip_id_dict = {}
    # Iterate through the unique trip_ids and create the boolean vectors
    for trip_id in unique_trip_ids:
        trip_id_dict[trip_id] = (trip_ids == trip_id)
    return trip_id_dict

def run(online_models, dataset_name, preview_duration, run_name=None):
        
    if run_name is None:
        run_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    data_X, data_y, trip_ids = load_dataset(dataset_name)

    data_y_pred = np.zeros((len(online_models), len(data_y)))
    data_y_pred[:] = np.nan

    trip_id_dict = make_trip_id_dict(trip_ids)

    # calculate the average records per trip
    avg_records_per_trip = len(data_X)/len(np.unique(trip_ids))
    print('avg records per trip: ', avg_records_per_trip)
    # make a is_predicted np array to keep track of the records that have been predicted (boolean)
    is_predicted = np.zeros_like(trip_ids, dtype=bool)

    for online_model in online_models:
        if 'MSMSA' in online_model.method_name:
            online_model.max_horizon = len(data_y)
            data_X_without_time = data_X[:, 1:]
            online_model.initialize_anchors(data_X_without_time)

    print('number of unique trips: ', len(np.unique(trip_ids)))

    # make an array with empy lists to hold the trip_preds
    trips = []

    # preview_duration = 60

    # num_preview_samples = 80_000
    num_preview_samples = 0


    num_aheads = []
    stream_bar = tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))
    for k, (X, y) in enumerate(stream_bar):
        
        if k%1 == 0:
            pass
        else:
            continue

        X = X.reshape(1,-1)
        X_without_time = X[:, 1:]
        current_abs_time = X[0, 0]
        

        for online_model in online_models:
            if 'TMI' in online_model.method_name:
                online_model.update_online_model(X, y, fit_base_learner=True)
            else:
                online_model.update_online_model(X_without_time, y, fit_base_learner=True)


        if is_predicted[k] == False:
            # get ahead indices from X_trip
            ahead_indices = np.where((data_X[:, 0] >= current_abs_time) & (data_X[:, 0] < current_abs_time + preview_duration) & (trip_id_dict[trip_ids[k]]))[0]
            num_aheads.append(len(ahead_indices))

            X_ahead = data_X[ahead_indices]
            X_ahead_without_time = X_ahead[:, 1:]
            # make is_predicte True for the ahead indices
            is_predicted[ahead_indices] = True


            pred_trip = []
            for i, online_model in enumerate(online_models):
                if 'TMI' in online_model.method_name:
                    # pred_trip.append(online_model.predict_online_model(X_ahead))
                    data_y_pred[i, ahead_indices] = online_model.predict_online_model(X_ahead)
                else:
                    # pred_trip.append(online_model.predict_online_model(X_ahead_without_time))
                    data_y_pred[i, ahead_indices] = online_model.predict_online_model(X_ahead_without_time)


            # num_train_samples = []
            
            # for online_model in online_models:
            #     num_train_samples.append(online_model.get_num_samples())


                                                                                         
         
   

    # put data_X, data_y, data_y_pred in a df
    df = pd.DataFrame(data_X, columns=['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months'])
    df['TripID'] = trip_ids
    df['Friction (measured)'] = data_y
    
                                                                                    
    for i, online_model in enumerate(online_models):
        df[online_model.method_name] = data_y_pred[i]

    # calculate the overal MAE, RMSE, and R2 for all the records in df
    mae = np.mean(np.abs(df['Friction (measured)'] - df[online_model.method_name]))
    rmse = np.sqrt(np.mean(np.square(df['Friction (measured)'] - df[online_model.method_name])))
    r2 = 1 - np.sum(np.square(df['Friction (measured)'] - df[online_model.method_name])) / np.sum(np.square(df['Friction (measured)'] - np.mean(df['Friction (measured)'])))
         

    summary = {
        'average_records_per_trip': len(data_X)/len(np.unique(trip_ids)),
        'average_ahead_records': np.mean(num_aheads),
        'preview_duration': preview_duration,
        'learning_model': online_models[0].base_learner.__class__.__name__,
        'dataset': dataset_name,
        'online_models': [online_model.method_name for online_model in online_models],
        'metric_MAE': mae,
        'metric_RMSE': rmse,
        'metric_R2': r2,
    }
    # put all online methods hyperparams in the summary dict with a key of the method name
    for online_model in online_models:
        summary[online_model.method_name] = online_model.hyperparams
    


    with open('Teconer_results/'+run_name+'.pkl', 'wb') as f:
        pickle.dump((df,summary) , f)
    return summary



# {dataset}" {method} {base_learner} {seed} {wandb_log} {tag}
# get base_learner_name from the argument of the script
dataset_name = sys.argv[1]
online_model_name = sys.argv[2]
base_learner_name = sys.argv[3]
epsilon = float(sys.argv[4])
preview_duration = int(sys.argv[5])
seed = int(sys.argv[6])
# get wandb_log True/False from argument 5 of the script
wandb_log = sys.argv[7] == 'True'
if len(sys.argv) > 8:
    tags = sys.argv[8:]
else:
    tags = None



pickle_log = True

if online_model_name == 'TMI':
    online_models = [tmi.TMI(epsilon=epsilon)]
elif online_model_name == 'PTMI':
    online_models = [tmi.TMI(epsilon=epsilon, probabilistic_prediction='ensemble')]
elif online_model_name == 'MSMSA':
    online_models = [msmsa.MSMSA()]
elif online_model_name == 'KSWIN':
    online_models = [kswin_reg.KSWIN()]
elif online_model_name == 'ADWIN':
    online_models = [adwin_reg.ADWIN()]
elif online_model_name == 'DDM':
    online_models = [ddm_reg.DDM()]
elif online_model_name == 'PH':
    online_models = [ph_reg.PH()]
elif online_model_name == 'Naive':
    online_models = [naive_reg.Naive()]


if base_learner_name == 'RF':
    for online_model in online_models:
        online_model.base_learner = RandomForestRegressor(n_estimators=50, max_depth=7, n_jobs=4, bootstrap=True, max_samples=.9)
elif base_learner_name == 'DT':
    for online_model in online_models:
        online_model.base_learner = DecisionTreeRegressor(max_depth=5)



# dataset = 'Teconer_1M'
#             # 'Teconer_100K'
#             # 'Teconer_downtown'
#             # 'Teconer_full',
#             # 'Teconer_100K',
#             # 'Teconer_1M',
#             # 'Teconer_road_piece'
#                 # ]

# online_models = [
#                     tmi.TMI(epsilon=0.6),
#                     # tmi.TMI(probabilistic_prediction='ensemble', epsilon=0.9),
#                     # msmsa.MSMSA(),
#                     # kswin_reg.KSWIN(),
#                     # adwin_reg.ADWIN(),
#                     # ddm_reg.DDM(),
#                     # ph_reg.PH(),
#                     # naive_reg.Naive(),

                 
#                  ]
# for online_model in online_models:
#     # online_model.base_learner = RandomForestRegressor(n_estimators=100, max_depth=7, n_jobs=4, bootstrap=True, max_samples=.9)
#     online_model.base_learner = DecisionTreeRegressor(max_depth=5)
config = {
    'dataset': dataset_name,
    'online_model': online_model_name,
    'base_learner': base_learner_name,
    'epsilon': epsilon,
    'seed': seed,
    'wandb_log': wandb_log,
    'tags': tags
}

if wandb_log:
        wandb_run = wandb.init(project='stream_learning', entity='haeri-hsn', config=config, tags=tags)
        # get the run name
        run_name = wandb_run.name
        summary = run(online_models=online_models, dataset_name=dataset_name, preview_duration=preview_duration, run_name=run_name)
else:
    summary = run(online_models=online_models, dataset_name=dataset_name, preview_duration=preview_duration)

if wandb_log:
    wandb.log(summary)
    wandb_run.finish()