import numpy as np
from sklearn.preprocessing import StandardScaler



class Memory:
    def __init__(self, max_num_samples=None, num_features=None):
        # make X as an array of shape (num_samples, num_features) with float64 dtype
        # self.X = np.zeros((max_num_samples, num_features), dtype=np.float64)
        # # make y as an array of shape (num_samples,) with float64 dtype
        # self.y = np.zeros((max_num_samples,), dtype=np.float64)
        # self.is_actives = np.zeros((max_num_samples,), dtype=np.bool_)
        self.next_available_spot_in_memory = 0
        self.current_time = None
        self.base_learner_is_fitted = False
        self.base_learner = None
        self.is_first_sample = True
        self.max_model_memory_len = 10
        

    def construct_memory(self, X):
        max_num_samples = 1_00
        # get num features from the first sample in the X
        num_features = X.shape[0]
        self.X = np.zeros((max_num_samples, num_features), dtype=np.float64)
        self.y = np.zeros((max_num_samples,), dtype=np.float64)
        self.is_actives = np.zeros((max_num_samples,), dtype=np.bool_)

    # double the size of the memory if it is full
    def extend_memory(self):
        self.condense_memory()
        self.X = np.concatenate([self.X, np.zeros_like(self.X)], axis=0)
        self.y = np.concatenate([self.y, np.zeros_like(self.y)], axis=0)
        self.is_actives = np.concatenate([self.is_actives, np.zeros_like(self.is_actives)], axis=0)

    # condense the memory to the active number of samples in the memory
    def condense_memory(self):
        self.X = self.X[self.is_actives]
        self.y = self.y[self.is_actives]
        self.is_actives = np.ones_like(self.y, dtype=np.bool_)
        self.next_available_spot_in_memory = len(self.is_actives)
        # self.next_available_spot_in_memory = self.X.shape[0]

    def add_sample(self,X_with_time, y):
        if self.is_first_sample:
            self.construct_memory(X_with_time[0])
            self.is_first_sample = False
        elif self.next_available_spot_in_memory == self.X.shape[0]:
            self.extend_memory()
        X_with_time = np.array(X_with_time).squeeze()
        y = np.array(y).squeeze()

        if X_with_time.ndim == 0:
            t = None
            self.current_time = None
        else:
            t = X_with_time[0]
            if self.current_time is None:
                    self.current_time = t
            if self.current_time < t:
                    self.current_time = t

        self.X[self.next_available_spot_in_memory] = X_with_time
        self.y[self.next_available_spot_in_memory] = y
        self.is_actives[self.next_available_spot_in_memory] = True
        self.next_available_spot_in_memory += 1

    def add_sample_bulk(self, X_with_time, y):
        num_new_samples = X_with_time.shape[0]
        memory_size = self.X.shape[0]
        while num_new_samples > memory_size - self.next_available_spot_in_memory:
            self.extend_memory()
        self.X[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = X_with_time
        self.y[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = y
        self.is_actives[self.next_available_spot_in_memory:self.next_available_spot_in_memory + num_new_samples] = True
        self.next_available_spot_in_memory += num_new_samples


    def get_num_samples(self):
        if self.is_first_sample:
            return 0
        return np.sum(self.is_actives)
        
    # def get_X(self):
    #     return self.X[self.is_actives, 1:]
    
    # def get_y(self):
    #     return self.y[self.is_actives]
    
    # def get_t(self):
    #     return self.X[self.is_actives, 0]
    
    # def get_X_with_time(self):
    #     return self.X[self.is_actives]

    # def get_X(self, only_last=None):
    #     if only_last is None:
    #         return self.X[self.is_actives]
    #     else:
    #         active_indices = np.flatnonzero(self.is_actives)
    #         active_indices = active_indices[-only_last:]
    #         return self.X[active_indices]
    
    def get_y(self, only_last=None):
        if only_last is None:
            return self.y[self.is_actives]
        else:
            active_indices = np.flatnonzero(self.is_actives)
            active_indices = active_indices[-only_last:]
            return self.y[active_indices]
    
    def get_t(self, only_last=None):
        if only_last is None:
            return self.X[self.is_actives, 0]
        else:
            active_indices = np.flatnonzero(self.is_actives)
            active_indices = active_indices[-only_last:]
            return self.X[self.is_actives, 0]
    
    def get_X(self, with_time=True, only_last=None):
        if only_last is None:
            if with_time:
                return self.X[self.is_actives]
            else:
                return self.X[self.is_actives, 1:]
        else:
            active_indices = np.flatnonzero(self.is_actives)
            active_indices = active_indices[-only_last:]
            if with_time:
                return self.X[active_indices]
            else:
                return self.X[active_indices, 1:]


    

    def predict_online_model(self, X):
        if self.base_learner_is_fitted:
            return self.base_learner.predict(X)
        elif self.get_num_samples() > 0:
            return [np.mean(self.get_y())]
        else:
            return [0]
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))
    

    def fit_to_memory(self, only_last=None, add_to_model_memory=False):
        if self.get_num_samples() < 1:
            print('No active samples in memory to fit')
            return
        if only_last is not None:
            self.base_learner.fit(self.get_X(only_last=only_last), self.get_y(only_last=only_last))
        else:
            self.base_learner.fit(self.get_X(), self.get_y())
        
        self.base_learner_is_fitted = True
        # return self.base_learner


    def forget_before(self, num_keep_samples):
        num_samples = self.get_num_samples()
        if num_samples > num_keep_samples:
            self.is_actives[:num_samples - num_keep_samples] = False
        else:
            self.is_actives[:] = False



