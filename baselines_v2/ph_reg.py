import numpy as np
# from skmultiflow.drift_detection import PageHinkley
from river.drift import PageHinkley

class PH(PageHinkley):
    def __init__(self,min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10):
        super().__init__(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001)
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'PH'
        self.min_memory_len = min_memory_len

        self.hyperparams = {
                            'min_instances':min_instances,
                            'delta':delta,
                            'threshold':threshold,
                            'alpha':alpha,
                            'min_memory_len':min_memory_len,
                
                            }

    def add_sample(self, X, y):
        self.memory.append((X, y))

    def detect(self, error):
        # self.add_element(error)
        self.update(error)
        self.change_flag = self.drift_detected
        self.update_memory()
        return False, self.change_flag

    def update_memory(self):
        if self.change_flag:
            self.memory = self.memory[-self.min_memory_len:]

    def get_recent_data(self):
        return self.memory

    def reset_detector(self):
        self.reset()
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []

    # def update_(self, model, error):
    #     self.detect(error)
    #     model.reset()
    #     model.fit(self.memory)
    #     return model, len(self.memory)
    

    def update_online_model(self, X, y):
        self.add_sample(X, y)
        if self.base_learner_is_fitted:
            y_hat = self.predict_online_model(X)
        else:
            y_hat = 0
        error = self.mean_absoulte_error(y, y_hat)
        self.detect(error)
        self.base_learner.reset()
        self.base_learner.fit(self.memory)
        if len(self.memory) > 1:
            self.base_learner_is_fitted = True
        return None
    
    def predict_online_model(self, X):
        return self.base_learner.predict(X)
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))