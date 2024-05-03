import numpy as np


class Memory:
    def __init__(self, max_num_samples=None, num_features=None):
        # make X as an array of shape (num_samples, num_features) with float64 dtype
        # self.X = np.zeros((max_num_samples, num_features), dtype=np.float64)
        # # make y as an array of shape (num_samples,) with float64 dtype
        # self.y = np.zeros((max_num_samples,), dtype=np.float64)
        # self.active_indices = np.zeros((max_num_samples,), dtype=np.bool_)
        self.next_available_spot_in_memory = 0
        self.current_time = None
        self.base_learner_is_fitted = False
        self.base_learner = None
        self.is_first_sample = True

    def construct_memory(self, X):
        max_num_samples = 1_00
        # get num features from the first sample in the X
        num_features = X.shape[0]
        self.X = np.zeros((max_num_samples, num_features), dtype=np.float64)
        self.y = np.zeros((max_num_samples,), dtype=np.float64)
        self.active_indices = np.zeros((max_num_samples,), dtype=np.bool_)

    # double the size of the memory if it is full
    def extend_memory(self):
        self.condense_memory()
        self.X = np.concatenate([self.X, np.zeros_like(self.X)], axis=0)
        self.y = np.concatenate([self.y, np.zeros_like(self.y)], axis=0)
        self.active_indices = np.concatenate([self.active_indices, np.zeros_like(self.active_indices)], axis=0)

    # condense the memory to the active number of samples in the memory
    def condense_memory(self):
        self.X = self.X[self.active_indices]
        self.y = self.y[self.active_indices]
        self.active_indices = np.ones_like(self.y, dtype=np.bool_)
        self.next_available_spot_in_memory = len(self.active_indices)
        # self.next_available_spot_in_memory = self.X.shape[0]

    def add_sample(self,X_with_time, y):
        if self.is_first_sample:
            self.construct_memory(X_with_time[0])
            self.is_first_sample = False
        elif self.next_available_spot_in_memory == self.X.shape[0]:
            self.extend_memory()
        X_with_time = np.array(X_with_time).squeeze()
        y = np.array(y).squeeze()
        t = X_with_time[0]
        if self.current_time is None:
                self.current_time = t
        if self.current_time < t:
                self.current_time = t

        self.X[self.next_available_spot_in_memory] = X_with_time
        self.y[self.next_available_spot_in_memory] = y
        self.active_indices[self.next_available_spot_in_memory] = True
        self.next_available_spot_in_memory += 1

    def add_sample_bulk(self, X_with_time, y):
        num_new_samples = X_with_time.shape[0]
        memory_size = self.X.shape[0]
        while num_new_samples > memory_size - self.next_available_spot_in_memory:
            self.extend_memory()
        self.X[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = X_with_time
        self.y[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = y
        self.active_indices[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = True
        self.next_available_spot_in_memory += num_new_samples


    def get_num_samples(self):
        if self.is_first_sample:
            return 0
        return np.sum(self.active_indices)
        
    def get_X(self):
        return self.X[self.active_indices, 1:]
    
    def get_y(self):
        return self.y[self.active_indices]
    
    def get_t(self):
        return self.X[self.active_indices, 0]
    
    def get_X_with_time(self):
        return self.X[self.active_indices]


    

    def predict_online_model(self, X):
        if self.base_learner_is_fitted:
            return self.base_learner.predict(X)
        elif self.get_num_samples() > 0:
            return [np.mean(self.get_y())]
        else:
            return [0]
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))
    

    def fit_to_memory(self):
        if self.get_num_samples() < 1:
            print('No active samples in memory to fit')
            return
        self.base_learner.fit(self.get_X_with_time(), self.get_y())
        self.base_learner_is_fitted = True


    def forget_before(self, num_keep_samples):
        num_samples = self.get_num_samples()
        if num_samples > num_keep_samples:
            self.active_indices[:num_samples - num_keep_samples] = False
        else:
            self.active_indices[:] = False



