import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# dataset_name = datasets[0]
def load_dataset(dataset_name, hyperplane_dimension=10,stream_size=10000, noise_var=0.2, drift_probability=0.05):


    if dataset_name == 'Teconer':
        # check if the dataset is already downloaded if not download it
        if not os.path.exists('datasets/Teconer.csv'):
            print('Downloading Teconer dataset...')
            os.system('wget https://drive.google.com/file/d/1uEMMEfowbi5m8I2zBay4o7x53eClTE5B/view?usp=drive_link -P datasets/')
        # read the dataset
        df = pd.read_csv('datasets/Teconer.csv', usecols = ['Date', 'Time', 'Latitude', 'Longitude', 'Friction', 'Tsurf', 'Ta', 'SensorName', 'VehicleID', 'Speed']).dropna()
        # sort the df by date and time
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values(by=['Date', 'Time'])
        data_X = df[['Latitude', 'Longitude','Tsurf', 'Ta']].to_numpy()
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
        data_X = df[['index_col','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]
        # data_X = df[['dist','pickup_hour_sin','pickup_hour_cos']].to_numpy()[:10_000]

    if dataset_name == 'Household energy':
        df = pd.read_csv('datasets/household_power_consumption.csv').dropna()
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
        df = pd.read_csv('datasets/bike_hour.csv')
        data_X = df[['workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
        # data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
        data_y = df['cnt'].to_numpy()

    if dataset_name == 'Melbourn housing':
        df = pd.read_csv('datasets/Melbourne_housing_full_sorted.csv').dropna()
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