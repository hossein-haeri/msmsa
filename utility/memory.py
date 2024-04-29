import numpy as np


class Memory:
    def __init__(self, max_num_samples=None, num_features=None):
        # make X as an array of shape (num_samples, num_features) with float64 dtype
        self.X = np.zeros((max_num_samples, num_features), dtype=np.float64)
        # make y as an array of shape (num_samples,) with float64 dtype
        self.y = np.zeros((max_num_samples,), dtype=np.float64)

        self.active_indices = np.zeros((max_num_samples,), dtype=np.bool_)
        self.next_spot = 0

        self.current_time = None
        self.base_learner_is_fitted = False
        self.base_learner = None


    def add_sample(self,X_with_time, y):
        X_with_time = np.array(X_with_time).squeeze()
        y = np.array(y).squeeze()
        t = X_with_time[0]
        # X = X_with_time[1:]
        if self.current_time is None:
                self.current_time = t
        if self.current_time < t:
                self.current_time = t

        self.X[self.next_spot] = X_with_time
        self.y[self.next_spot] = y
        self.active_indices[self.next_spot] = True
        self.next_spot += 1
        # if self.next_spot == len(self.X):
        #     self.next_spot = 0


    def get_num_samples(self):
        return np.sum(self.active_indices)
        

    def get_X(self):
        # if len(self.samples) == 0:
        #     return np.array([])
        # return np.array([sample.X for sample in self.samples])
        return self.X[self.active_indices, 1:]
    
    def get_y(self):
        # if len(self.samples) == 0:
        #     return np.array([])
        # return np.array([sample.y for sample in self.samples])
        return self.y[self.active_indices]
    
    def get_t(self):
        # if len(self.samples) == 0:
        #     return np.array([])
        # return np.array([sample.t for sample in self.samples])
        # return the first column of X for all samples
        return self.X[self.active_indices, 0]
    
    def get_X_with_time(self):
        # if len(self.samples) == 0:
        #     return np.array([])
        # return np.array([np.append(sample.t, sample.X) for sample in self.samples])
        return self.X[self.active_indices]

    def get_X_with_current_time(self):
        X_with_current_time = self.get_X_with_time()
        # make all elements in the first column of X_with_current_time equal to the current time
        X_with_current_time[:, 0] = self.current_time
        return X_with_current_time
    

    def predict_online_model(self, X):
        if self.base_learner_is_fitted:
            return self.base_learner.model.predict(X)
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
        self.base_learner.model.fit(self.get_X_with_time(), self.get_y())
        self.base_learner_is_fitted = True


    def forget_before(self, num_keep_samples):
        num_samples = self.get_num_samples()
        if num_samples > num_keep_samples:
            self.active_indices[:num_samples - num_keep_samples] = False
        else:
            self.active_indices[:] = False



