import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import stream_generator
import torch

# dataset_name = datasets[0]
def load_dataset(dataset_name, synthetic_param=None, seed=None):
    # if synthetic_param is not None:
    #     synthetic_param['hyperplane_dimension']=10
    #     synthetic_param['stream_size']=10000
    #     synthetic_param['noise_var']=0.2
    #     synthetic_param['drift_probability']=0.05

    data_w = None
    trip_ids = None

    # if dataset_name == 'Teconer':
    #     # check if the dataset is already downloaded if not download it
    #     if not os.path.exists('datasets/Teconer.csv'):
    #         print('Downloading Teconer dataset...')
    #         os.system('wget https://drive.google.com/file/d/1uEMMEfowbi5m8I2zBay4o7x53eClTE5B/view?usp=drive_link -P datasets/')
    #     # read the dataset
    #     df = pd.read_csv('datasets/Teconer.csv', usecols = ['Date', 'Time', 'Latitude', 'Longitude', 'Friction', 'Tsurf', 'Ta', 'SensorName', 'VehicleID', 'Speed']).dropna()
    #     # sort the df by date and time
    #     df['Date'] = pd.to_datetime(df['Date'])
    #     df['Time'] = pd.to_datetime(df['Time'])
    #     df = df.sort_values(by=['Date', 'Time'])
    #     data_X = df[['Latitude', 'Longitude','Tsurf', 'Ta']].to_numpy()
    #     data_y = df['Friction'].to_numpy()
    if dataset_name == 'Teconer_road_piece':
        df = pd.read_csv('datasets/Teconer_road_piece_full.csv').dropna()
        # pickle the df 
        # df.to_pickle(dataset_name+'_records.pkl')
        # print(df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed']].head())
        data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Month ']].to_numpy(dtype=float)
        # data_X = df[['AbsoluteTime','Latitude']].to_numpy(dtype=float)
        trip_ids = df['TripID'].to_numpy(dtype=int)
        data_y = df['Friction'].to_numpy()

         # Move tensors to GPU if CUDA is available
        # if torch.cuda.is_available():
        #     # Convert entire dataset to PyTorch tensors
        #     data_X = torch.tensor(data_X, dtype=torch.float32)
        #     data_y = torch.tensor(data_y, dtype=torch.float32)
        
        #     data_X = data_X.cuda()
        #     data_y = data_y.cuda()
    



    if dataset_name == 'Teconer_10K':
        df = pd.read_csv('datasets/Teconer_2018_Jan_light_10K.csv').dropna()
        # pickle the df 
        # df.to_pickle(dataset_name+'_records.pkl')
        # print(df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed']].head())
        # data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        data_X = df[['Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        # trip_ids = df['TripID'].to_numpy(dtype=int)
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'Teconer_100K':
        df = pd.read_csv('datasets/Teconer_2018_Jan_light_100K.csv').dropna()
        # pickle the df 
        print(df.columns)
        # df.to_pickle(dataset_name+'_records.pkl')
        # data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        data_X = df[['Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        trip_ids = df['TripID'].to_numpy(dtype=int)
        # print(trip_ids)
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'Teconer_1M':
        df = pd.read_csv('datasets/Teconer_2018_Jan_light_1M.csv').dropna()
        # pickle the df 
        df.to_pickle(dataset_name+'_records.pkl')
        data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        trip_ids = df['TripID'].to_numpy(dtype=int)
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'Teconer_downtown':
        # df = pd.read_csv('datasets/Teconer_downtown_10K.csv').dropna()
        # df = pd.read_csv('datasets/Teconer_downtown_100K.csv').dropna()
        df = pd.read_csv('datasets/Teconer_downtown_full.csv').dropna()
        # pickle the df 
        df.to_pickle(dataset_name+'_records.pkl')
        data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed','Months']].to_numpy()
        # data_X = df[['AbsoluteTime','Latitude', 'Longitude','Tsurf', 'Ta','Hours','Speed']].to_numpy()
        # data_X = df[['Latitude', 'Longitude']].to_numpy()
        trip_ids = df['TripID'].to_numpy(dtype=int)
        data_y = df['Friction'].to_numpy()


    if dataset_name == 'NYC taxi':
        df = pd.read_csv('datasets/nyc_taxi_train.csv').dropna()
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
        data_X = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]
        data_X = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]
        # data_X = df[['dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]

    if dataset_name == 'Household energy':
        df = pd.read_csv('datasets/household_power_consumption.csv').dropna()
        # df = df.reset_index().rename(columns={'index': 'index_col'})
        data_y = df['Sub_metering_3'].to_numpy()[:10_000]
        data_X = df[['Global_active_power','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2']].to_numpy()[:10_000]

    if dataset_name == 'Bike (daily)':
        df = pd.read_csv('datasets/bike_day.csv')
        # df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
        # data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
        data_y = df['cnt'].to_numpy()

    if dataset_name == 'Bike (hourly)':
        df = pd.read_csv('datasets/bike_hour.csv')
        data_X = df[['workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
        # data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
        data_y = df['cnt'].to_numpy()

    if dataset_name == 'Melbourne housing':
        # df = pd.read_csv('datasets/Melbourne_housing_full_sorted.csv').dropna()
        # data_y = df['Price'].to_numpy()
        # data_X = df[['Lattitude','Longtitude','YearBuilt','BuildingArea','Landsize','Car','Bathroom','Bedroom2','Distance']].to_numpy()
        df = pd.read_csv('datasets/melbourne_housing_clean.csv').dropna()
        data_y = df['Price'].to_numpy()
        data_X = df[['Lattitude','Longtitude','YearBuilt','BuildingArea','Landsize','Car','Bathroom','Bedroom2','Distance']].to_numpy()


    if dataset_name == 'Air quality':
        df = pd.read_csv('datasets/AirQualityUCI.csv').dropna()
        df.drop(df[(df['CO(GT)'] == -200)].index, inplace=True)
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']].to_numpy()
        data_y = df['CO(GT)'].to_numpy()

    if dataset_name == 'datasets/Friction':
        df = pd.read_csv('friction_2.csv', usecols = ['Latitude','Longitude','Height','Tsurf','Ta','Friction']).dropna()
        df = df.reset_index().rename(columns={'index': 'index_col'})
        data_X = df[['Latitude','Longitude','Height','Tsurf','Ta']].to_numpy()
        data_y = df['Friction'].to_numpy()

    if dataset_name == 'Hyper-A':
        stream = stream_generator.hyper_abrupt(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-I':
        stream = stream_generator.hyper_incremental(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-G':
        stream = stream_generator.hyper_gradual(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-GU':
        stream = stream_generator.hyper_gaussian(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-LN':
        stream = stream_generator.hyper_linear(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-RW':
        stream = stream_generator.hyper_random_walk(synthetic_param, seed=seed)
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'Hyper-HT':
        stream = stream_generator.hyper_abrupt_half_drift(synthetic_param, seed=seed)
        # print(stream[0])
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])


    if dataset_name == 'Hyper-ND':
        stream = stream_generator.hyper_noise_drift(synthetic_param, seed=seed)
        # print(stream[0])
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        data_w = np.array([item[2] for item in stream])

    if dataset_name == 'SimpleHeterogeneous':
        stream = stream_generator.simple_heterogeneous(synthetic_param, seed=seed)
        # print(stream[0])
        data_X = np.array([item[0] for item in stream])
        data_y = np.array([item[1] for item in stream])
        # data_w = np.array([item[2] for item in stream])

    # if 'Hyper' in dataset_name:
        # include time (index) as a feature
    # data_X = np.column_stack((np.arange(len(data_X)), data_X))
    # scaler_X = StandardScaler()

    # scaler_X = MinMaxScaler()
    # data_X = scaler_X.fit_transform(data_X)
    # scaler_y = StandardScaler()
    # data_y = scaler_y.fit_transform(data_y.reshape(-1, 1)).squeeze()

    if 'Teconer' in dataset_name:
        return data_X, data_y, trip_ids

    # else:
    #     return data_X, data_y, scaler_X, scaler_y
    return data_X, data_y, data_w

