import numpy as np
import copy
# from skmultiflow.drift_detection import KSWIN as KolmogorovSmirnovWIN

from river.drift import KSWIN as KolmogorovSmirnovWIN

class KSWIN(KolmogorovSmirnovWIN):
    def __init__(self,alpha=0.005, window_size=100, stat_size=30, min_memory_len=10):
        super().__init__(alpha=alpha, window_size=window_size, stat_size=stat_size)
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'KSWIN'
        self.min_memory_len = min_memory_len
        self.hyperparams = {'window_size':window_size,
                            'stat_size':stat_size,
                            'alpha':alpha,
                            'min_memory_len':min_memory_len
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
    
    def get_val_horizon(self):
        return len(self.memory)

    def reset_detector(self):
        self.reset()
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []

    # def update_(self, model, error):
    #     self.detect(error)
    #     # model_ = copy.copy(model)

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