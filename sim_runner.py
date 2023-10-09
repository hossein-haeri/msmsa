
from itertools import repeat
from time import gmtime, strftime
from tqdm import tqdm
import sys
import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# %matplotlib qt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import stream_generator
import learning_models
from baselines import davar_reg
from baselines import ddm_reg
from baselines import adwin_reg
from baselines import kswin_reg 
from baselines import ph_reg
from baselines import naive_reg
import msmsa
import msmsa_plus
import wandb
import os




# dataset_name = datasets[0]
def load_dataset(dataset_name, hyperplane_dimension=10,stream_size=10000, noise_var=0.2, drift_probability=0.05):


    if dataset_name == 'Teconer':
        df = pd.read_csv('Teconer.csv', usecols = ['Date', 'Time', 'Latitude', 'Longitude', 'Friction', 'Tsurf', 'Ta', 'SensorName', 'VehicleID', 'Speed']).dropna()
        # sort the df by date and time
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values(by=['Date', 'Time'])
        data_X = df[['Latitude', 'Longitude','Tsurf', 'Ta']].to_numpy()
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'NYC taxi':
        df = pd.read_csv('nyc_taxi_train.csv').dropna()
        df = df.reset_index().rename(columns={'index': 'index_col'})
        # create an additional distance feature
        df['dist'] = np.sqrt((df['pickup_longitude']-df['dropoff_longitude'])**2 + (df['pickup_latitude']-df['dropoff_latitude'])**2)
        # given pickup_date time feature (3/14/2016  5:24:55 PM), create additional features as time of day in hours
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        # sort every row by pickup_datetime
        df = df.sort_values(by='pickup_datetime')
        # create a new feature using pickup_datetime as its hour
        df['pickup_hour'] = df['pickup_datetime'].dt.hour + df['pickup_datetime'].dt.minute/60
        # create two new features using pickup_hour as its sin and cos
        df['pickup_hour_sin'] = np.sin(df['pickup_hour']*(2.*np.pi/24))
        df['pickup_hour_cos'] = np.cos(df['pickup_hour']*(2.*np.pi/24))
        data_y = df['trip_duration'].to_numpy()[:10_000]
        data_X = df[['index_col','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]
        # data_X = df[['dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]

    if dataset_name == 'Household energy':
        df = pd.read_csv('household_power_consumption.csv').dropna()
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_y = df['Sub_metering_3'].to_numpy()[:10_000]
        data_X = df[['index_col','Global_active_power','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2']].to_numpy()[:10_000]

    if dataset_name == 'Bike (daily)':
        df = pd.read_csv('datasets/bike_day.csv')
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['index_col','workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
        # data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
        data_y = df['cnt'].to_numpy()

    if dataset_name == 'Bike (hourly)':
        df = pd.read_csv('bike_hour.csv')
        data_X = df[['workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
        # data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
        data_y = df['cnt'].to_numpy()

    if dataset_name == 'Melbourn housing':
        df = pd.read_csv('Melbourne_housing_full_sorted.csv').dropna()
        data_y = df['Price'].to_numpy()
        data_X = df[['Lattitude','Longtitude','YearBuilt','BuildingArea','Landsize','Car','Bathroom','Bedroom2','Distance']].to_numpy()
        # data_X = df[['YearBuilt','BuildingArea','Landsize','Car','Bathroom','Bedroom2','Distance']].to_numpy()
        # data_X = df[['Lattitude','Longtitude']].to_numpy()

    if dataset_name == 'Air quality':
        df = pd.read_csv('AirQualityUCI.csv').dropna()
        df.drop(df[(df['CO(GT)'] == -200)].index, inplace=True)
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['index_col','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']].to_numpy()
        data_y = df['CO(GT)'].to_numpy()

    if dataset_name == 'Friction':
        df = pd.read_csv('friction_2.csv', usecols = ['Latitude','Longitude','Height','Tsurf','Ta','Friction']).dropna()
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['index_col','Latitude','Longitude','Height','Tsurf','Ta']].to_numpy()
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'Hyper-A':
        stream = stream_generator.hyper_abrupt(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var, drift_probability=drift_probability)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])

    if dataset_name == 'Hyper-I':
        stream = stream_generator.hyper_incremental(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var, drift_probability=drift_probability)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])     

    if dataset_name == 'Hyper-G':
        stream = stream_generator.hyper_gradual(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var, drift_probability=drift_probability)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])

    if dataset_name == 'Hyper-GU':
        stream = stream_generator.hyper_gaussian(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var, drift_probability=drift_probability)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])

    if dataset_name == 'Hyper-LN':
        stream = stream_generator.hyper_linear(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var, drift_probability=drift_probability)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])

    if dataset_name == 'Hyper-RW':
        stream = stream_generator.hyper_random_walk(hyperplane_dimension=hyperplane_dimension, stream_size=stream_size, noise_var=noise_var)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])

    scaler_X = StandardScaler()
    data_X = scaler_X.fit_transform(data_X)
    scaler_y = StandardScaler()
    data_y = scaler_y.fit_transform(data_y.reshape(-1, 1)).squeeze()
    return data_X, data_y, scaler_y

def run(model, validator, dataset, dataset_configs, wandb_logrun):
        
    data_X, data_y, scaler_y = load_dataset(dataset,
                                                hyperplane_dimension=dataset_configs['dim'],
                                                noise_var=dataset_configs['noise_var'],
                                                stream_size=dataset_configs['stream_size'],
                                                drift_probability=dataset_configs['drift_prob'])
    
    results = np.zeros([len(data_X),4])
    valid_model = None
    y = 0
    pred_y = 0
    params = []
    for k, (X, y) in enumerate(zip(data_X, data_y)):
        try:
            pred_y = valid_model.predict(X)
        except: 
            pred_y = pred_y
        pred_y = float(pred_y)
        e = float(np.absolute(y - pred_y))
        validator.add_sample([X, y])
        valid_model, val_horizon = validator.update_(model, e)
        results[k,:] = [e, val_horizon, y, pred_y]
        # print(y, pred_y)
        # print(scaler.inverse_transform(y), scaler.inverse_transform(pred_y))
        if wandb_log and wandb_logrun:
            wandb.log({
                'error': float(y - pred_y),
                'abs_error': e,
                'validity_horizon': val_horizon,
                'y': y,
                'pred_y': pred_y
            })

    # print scaler_y mean and scale
    # print(scaler_y.mean_, scaler_y.scale_)

    # e_inv = scaler_y.inverse_transform(results[:,0].reshape(-1, 1))
    y_inv = scaler_y.inverse_transform(results[:,2].reshape(-1, 1))
    pred_y_inv = scaler_y.inverse_transform(results[:,3].reshape(-1, 1))
    e_inv = np.absolute(y_inv - pred_y_inv)
    y_bar_inv = np.mean(y_inv)

    run_summary = { 'dataset': dataset,
                    'stream_size': len(data_y),
                    'method': validator.method_name,
                    'learning_model': type(model.model).__name__,
                    'noise_var': noise_var,
                    'MAE': np.mean(e_inv),
                    'STD': np.std(e_inv),
                    'RMSE': np.sqrt(np.mean(e_inv**2)),
                    'RRSE': np.sqrt(np.sum(e_inv**2)/np.sum((y_inv - y_bar_inv)**2)),
                    'TargetMean': y_bar_inv,
                    'TargetSTD': np.std(y_inv),
                    'MeanValidityHorizon': np.mean(results[:,1]),
                }
    return run_summary

wandb_log = False
wandb_logrun = False
pickle_log = True

datasets = [
            'Bike (daily)',
            # 'Bike (hourly)',
            # 'Household energy',
            # 'Melbourn housing',
            # 'Air quality',
            # 'Friction',
            # 'NYC taxi',
            # 'Teconer'
                ]

# datasets = ['Hyper-A',
#             'Hyper-I',
#             'Hyper-G',
#             'Hyper-LN',
#             'Hyper-RW',
#             'Hyper-GU'
#                ]


# dataset_configs = {'noise_var': 0,
#                    'stream_size': 1_000,
#                    'drift_prob':0.01,
#                    'dim': 10}

dataset_configs = {'noise_var': '-1',
                   'stream_size': '-1',
                   'drift_prob':'-1',
                   'dim': '-1'}

# datasets = ['Household energy']
# dataset_name = datasets[1]

# model = learning_models.Linear()
# model = learning_models.DecissionTree()
# # model = learning_models.KNN()
# # model = learning_models.SVReg()
# # model = learning_models.Polynomial()
models = [learning_models.Linear(),
          learning_models.DecissionTree(),
        #   learning_models.SVReg()
        ]

# noise_vars = [0, 1, 2, 3, 4, 5]
noise_vars = ['-1']
num_monte = 10

logs = pd.DataFrame()
for monte in tqdm(range(num_monte)):
    # print('------ NUMBER OF MONTE SIMS: ', monte, '/', num_monte)
    for model in tqdm(models, leave=False):
        for dataset in tqdm(datasets, leave=False):
            for noise_var in tqdm(noise_vars, leave=False):
                dataset_configs['noise_var'] = noise_var
                validators = [
                            # msmsa_plus.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8),
                            msmsa.MSMSA(min_memory_len=10, update_freq_factor=1, lam=0.8),
                            # davar_reg.DAVAR(lam=10),
                            kswin_reg.KSWIN(alpha=0.005, window_size=100, stat_size=30, min_memory_len=10),
                            # adwin_reg.ADWIN(delta=0.002),
                            # ddm_reg.DDM(alpha_w=2, alpha_d=3),
                            # ph_reg.PH(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10),
                            # naive_reg.Naive()
                            ]

                for validator in tqdm(validators, leave=False):
                    run_summary = run(    
                            validator=validator,
                            dataset=dataset,
                            model=model,
                            dataset_configs=dataset_configs,
                            wandb_logrun=wandb_logrun
                            )
                    
                    if wandb_log:
                        wandb.init( project='stream_learning',
                                    mode="offline",
                                    group=dataset,
                                    job_type=validator.method_name,
                                    tags=[type(model.model).__name__],
                                    config={'validator': validator.hyperparams, 
                                            'validator_name': validator.method_name,   
                                            'dataset': dataset_configs
                                            }     
                                        )
                    
                    # use pd concat to append run_summary to logs
                    logs = pd.concat([logs, pd.DataFrame([run_summary])], ignore_index=True)
                    
                    if wandb_log:
                        wandb.log(run_summary)
                        wandb.finish(quiet=True)
    
    # pickle the logs every 10 monte sims
    if pickle_log and monte % 1 == 0:
        with open('logs_msmsa_plus_test.pkl', 'wb') as f:
            pickle.dump(logs, f)


    