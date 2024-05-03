import torch
import pickle
from tqdm import tqdm

def load_dataset(name, params):
    # Dummy function for context
    pass

def rescale(data, scaler):
    # Convert tensor to numpy array for scaling
    data = data.numpy()
    return scaler.inverse_transform(data)

def rescale_features(data, scaler):
    # Convert tensor to numpy array for scaling
    data_X_np = data.numpy()
    return scaler.inverse_transform(data_X_np)

def run(online_model, dataset_name, synthetic_param):
    data_X, data_y, scaler_X, scaler_y, trip_ids = load_dataset(dataset_name, synthetic_param)
    predicted_trip_ids = set()

    trips = []
    num_records = len(data_y)

    idx = torch.randperm(data_y.size(0))[:1_000_000]
    sorted_idx, _ = idx.sort()
    data_X = data_X[sorted_idx]
    data_y = data_y[sorted_idx]
    trip_ids = trip_ids[sorted_idx]

    stream_bar = tqdm(zip(data_X, data_y), leave=False, disable=False, total=len(data_y))
    for k, (X, y) in enumerate(stream_bar):
        X = X.reshape(1, -1)

        if trip_ids[k] not in predicted_trip_ids:
            online_model.update_online_model(X, y, fit_base_learner=True)
            X_trip = data_X[trip_ids == trip_ids[k]]
            y_trip = data_y[trip_ids == trip_ids[k]]

            pred_trip = online_model.predict_online_model(X_trip)
            predicted_trip_ids.add(trip_ids[k])
            y_trip_rescaled = rescale(y_trip, scaler_y)
            pred_trip_rescaled = rescale(pred_trip, scaler_y)
            X_trip_rescaled = rescale_features(X_trip, scaler_X)

            trips.append([trip_ids[k], X_trip_rescaled, y_trip_rescaled, pred_trip_rescaled, online_model.get_num_samples()])
        else:
            online_model.update_online_model(X, y, fit_base_learner=False)

        stream_bar.set_postfix(MemSize=online_model.X.shape[0], Ratio=(online_model.get_num_samples()) / (k + 1), NumTrips=len(predicted_trip_ids))

    with open('trips_road_piece_dth_with_preview.pkl', 'wb') as f:
        pickle.dump(trips, f)
