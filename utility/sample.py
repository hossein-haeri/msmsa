import numpy as np

class Sample:
    def __init__(self, X, y, t, id=None):
        self.id = id
        self.X = X
        self.y = y
        self.t = t
        self.expiration_probability = 0.5

        self.mu_o = None
        self.mu_c = None
        self.sigma_o = None
        self.sigma_c = None

    def X_with_time(self):
        return np.append(self.t, self.X)
    
    def X_with_current_time(self, current_time):
        return np.append(current_time, self.X)


class Memory:
    def __init__(self, max_memory_size=None):
        self.max_memory_size = max_memory_size
        self.samples = []
        self.current_time = None
        self.sample_id_counter = 0
        self.base_learner_is_fitted = False
        self.base_learner = None


    def add_sample(self,X_with_time, y):
        X_with_time = np.array(X_with_time).squeeze()
        y = np.array(y).squeeze()
        t = X_with_time[0]
        X = X_with_time[1:]
        if self.current_time is None:
                self.current_time = t
        if self.current_time < t:
                self.current_time = t
        self.samples.append(Sample(X, y, t, self.sample_id_counter))
        self.sample_id_counter += 1
        # else:
        #     for i in range(len(y)):
        #         self.add_sample(X[i], y[i], t[i])

    def get_X(self):
        if len(self.samples) == 0:
            return np.array([])
        return np.array([sample.X for sample in self.samples])
    
    def get_y(self):
        if len(self.samples) == 0:
            return np.array([])
        return np.array([sample.y for sample in self.samples])
    
    def get_t(self):
        if len(self.samples) == 0:
            return np.array([])
        return np.array([sample.t for sample in self.samples])
    
    def get_X_with_time(self):
        if len(self.samples) == 0:
            return np.array([])
        return np.array([np.append(sample.t, sample.X) for sample in self.samples])

    def get_X_with_current_time(self):
        if len(self.samples) == 0:
            return np.array([])
        return np.array([np.append(self.current_time, sample.X) for sample in self.samples])
    

    def predict_online_model(self, X):
        if self.base_learner_is_fitted:
            return self.base_learner.model.predict(X)
        elif len(self.samples) > 0:
            return [np.mean(self.get_y())]
        else:
            return [0]
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))
    

    def fit_to_memory(self):
        if len(self.samples) < 1:
            print('No samples in memory to fit')
            return
        self.base_learner.model.fit(self.get_X_with_time(), self.get_y())
        self.base_learner_is_fitted = True


