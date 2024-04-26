import numpy as np
# import copy
from utility.sample import Memory
# from skmultiflow.drift_detection import KSWIN as KolmogorovSmirnovWIN

from river.drift import KSWIN as KolmogorovSmirnovWIN

class KSWIN(KolmogorovSmirnovWIN, Memory):
    def __init__(self,alpha=0.005, window_size=100, stat_size=30, min_memory_len=10):
        super().__init__(alpha=alpha, window_size=window_size, stat_size=stat_size)
        Memory.__init__(self)

        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'KSWIN'
        self.min_memory_len = min_memory_len
        self.hyperparams = {'window_size':window_size,
                            'stat_size':stat_size,
                            'alpha':alpha,
                            'min_memory_len':min_memory_len
                            }

    def detect(self, error):
        self.update(error)
        self.change_flag = self.drift_detected
        self.update_memory()
        return False, self.change_flag

    def update_memory(self):
        if self.change_flag:
            self.samples = self.samples[-self.min_memory_len:]

    def reset_detector(self):
        self.reset()
        self.samples = []
        self.change_flag = False
        self.change_flag_history = []
    
    def update_online_model(self, X, y):
        self.add_sample(X, y)

        if self.base_learner_is_fitted:
            y_hat = self.predict_online_model(X)
        else:
            y_hat = 0
        error = self.mean_absoulte_error(y, y_hat)
        self.detect(error)
        # self.base_learner.reset()
        # self.base_learner.model.fit(self.get_X_with_time(), self.get_y())
        self.fit_to_memory()
        if len(self.samples) > 1:
            self.base_learner_is_fitted = True
            
    
