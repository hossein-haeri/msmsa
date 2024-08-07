from time import gmtime, strftime
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# %matplotlib qt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# import stream_generator
# import learning_models
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
# import neural_net_base_learner
import wandb
# import os
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

def run(online_model, dataset_name, preview_duration, columns, run_name=None):
        
    if run_name is None:
        run_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())



    if dataset_name == 'teconer_helsinki_jan2018':
        # unpikcle the dataset
        with open('datasets/teconer_helsinki_jan2018_df.pkl', 'rb') as f:
            df = pickle.load(f)
    elif dataset_name == 'teconer_helsinki_jan2018_10K':
        with open('datasets/teconer_helsinki_jan2018_df.pkl', 'rb') as f:
            df = pickle.load(f)
            # randomly sample 10K records from the dataset
            df = df.sample(n=10000, random_state=seed)
    elif dataset_name == 'teconer_helsinki_jan2018_100K':
        with open('datasets/teconer_helsinki_jan2018_df.pkl', 'rb') as f:
            df = pickle.load(f)
            # randomly sample 100K records from the dataset
            df = df.sample(n=100000, random_state=seed)
    elif dataset_name == 'teconer_helsinki_jan2018_1M':
        with open('datasets/teconer_helsinki_jan2018_df.pkl', 'rb') as f:
            df = pickle.load(f)
            # randomly sample 1M records from the dataset
            df = df.sample(n=1000000, random_state=seed)
    
    df = df.sort_values(by='UnixTime')

    num_records = len(df)
    trip_ids = df['TripID'].to_numpy(dtype=int)
    # print('number of unique trips: ', len(np.unique(trip_ids)))

    data_y_pred = np.zeros(num_records)
    data_y_pred[:] = np.nan
    trip_id_dict = make_trip_id_dict(trip_ids)

    # calculate the average records per trip
    avg_records_per_trip = num_records/len(np.unique(trip_ids))
    print('avg records per trip: ', avg_records_per_trip)
    # make a is_predicted np array to keep track of the records that have been predicted (boolean)
    is_predicted = np.zeros_like(trip_ids, dtype=bool)

    data_X = df[columns].to_numpy()
    data_y = df['Friction'].to_numpy()

    del df
    # if 'MSMSA' in online_model.method_name:
    #     online_model.max_horizon = num_records
    #     data_X_without_time = data_X[:, 1:]
    #     online_model.initialize_anchors(data_X_without_time)

    
    num_preview_samples = 0
    num_preview_points = []
    num_training_samples = []

    stream_bar = tqdm(zip(data_X, data_y),leave=False, disable=False, total=len(data_y))


    for k, (X, y) in enumerate(stream_bar):

    # for k in tqdm(range(num_records)):

        # read the kth row of the csv file 'datasets/teconer_helsinki_jan2018.csv'
        if k%1 == 0:
            pass
        else:
            continue

        X = X.reshape(1,-1)
        X_without_time = X[:, 1:]
        current_abs_time = X[0, 0]

        # predict the records that are in the preview window
        if is_predicted[k] == False:
            if 'TMI' in online_model.method_name and k > 0:
                online_model.update_online_model()
            # get preview indices from X_trip
            preview_indices = np.where((data_X[:, 0] >= current_abs_time) & (data_X[:, 0] < current_abs_time + preview_duration) & (trip_id_dict[trip_ids[k]]))[0]
            num_preview_points.append(len(preview_indices))

            X_preview = data_X[preview_indices]
            X_preview_without_time = X_preview[:, 1:]
            # make is_predicte True for the preview indices
            is_predicted[preview_indices] = True

            
            if 'TMI' in online_model.method_name:
                data_y_pred[preview_indices] = online_model.predict_online_model(X_preview)
            else:
                data_y_pred[preview_indices] = online_model.predict_online_model(X_preview_without_time)

        # update the online models with the current record
        if 'TMI' in online_model.method_name:
            online_model.update_online_model(X, y, fit_base_learner=False)
        else:
            online_model.update_online_model(X_without_time, y, fit_base_learner=True)
        
        # get the number of samples in the memory (tarining set size)
        if 'MSMSA' in online_model.method_name:
            num_training_samples.append(online_model.validity_horizon)
        else:
            num_training_samples.append(online_model.get_num_samples())


        # display num_training_samples[-1] in the progress bar
        stream_bar.set_description(f'num_training_samples: {num_training_samples[-1]}')                                                                      
         
   

    # put data_X, data_y, data_y_pred in a df
    df = pd.DataFrame(data_X, columns=columns)
    df['TripID'] = trip_ids
    df['Friction (measured)'] = data_y
    df[online_model.method_name] = data_y_pred

    # calculate the overal MAE, RMSE, and R2 for all the records in df

    mae = np.mean(np.abs(df['Friction (measured)'] - df[online_model.method_name]))
    rmse = np.sqrt(np.mean(np.square(df['Friction (measured)'] - df[online_model.method_name])))
    r2 = 1 - np.sum(np.square(df['Friction (measured)'] - df[online_model.method_name])) / np.sum(np.square(df['Friction (measured)'] - np.mean(df['Friction (measured)'])))


    summary = {
        'average_records_per_trip': len(data_X)/len(np.unique(trip_ids)),
        'average_preview_records': np.mean(num_preview_points),
        # 'training_size_log': num_training_samples,
        'average_training_size': np.mean(num_training_samples),
        'preview_window': preview_duration,
        'learning_model': online_model.base_learner.__class__.__name__,
        'dataset': dataset_name,
        'metric_MAE': mae,
        'metric_RMSE': rmse,
        'metric_R2': r2,
    }



    # put all online methods hyperparams in the summary dict with a key of the method name
    summary[online_model.method_name] = online_model.hyperparams
    


    with open('Teconer_results/'+run_name+'.pkl', 'wb') as f:
        pickle.dump((df, summary) , f)
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
    online_model = tmi.TMI(epsilon=epsilon, max_elimination=1000)
elif online_model_name == 'PTMI':
    online_model = tmi.TMI(epsilon=epsilon, probabilistic_prediction='ensemble')
elif online_model_name == 'MSMSA':
    online_model = msmsa.MSMSA()
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


if base_learner_name == 'RF':
    online_model.base_learner = RandomForestRegressor(n_estimators=10, max_depth=7, n_jobs=-1, bootstrap=True, max_samples=.8, min_samples_leaf=5)
elif base_learner_name == 'DT':
    online_model.base_learner = DecisionTreeRegressor(max_depth=7, min_samples_leaf=5)



config = {
    'dataset': dataset_name,
    'online_model': online_model_name,
    'base_learner': base_learner_name,
    'epsilon': epsilon,
    'seed': seed,
    'wandb_log': wandb_log,
    'tags': tags
}

# columns = ['UnixTime','Latitude', 'Longitude','Height','Speed', 'Direction', 'Ta', 'Tsurf', 'S1', 'S2', 'S3', 'S9', 'S10', 'S11', 'Hour']
columns = ['UnixTime','Latitude', 'Longitude','Height','Speed', 'Direction', 'Ta', 'Tsurf','Hour']

if wandb_log:
        wandb_run = wandb.init(project='stream_learning', entity='haeri-hsn', config=config, tags=tags)
        # get the run name
        run_name = wandb_run.name
        summary = run(online_model=online_model, dataset_name=dataset_name, preview_duration=preview_duration, columns=columns, run_name=run_name)
else:
    summary = run(online_model=online_model, dataset_name=dataset_name, preview_duration=preview_duration, columns=columns)

if wandb_log:
    wandb.log(summary)
    wandb_run.finish()