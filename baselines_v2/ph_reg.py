import numpy as np
# from skmultiflow.drift_detection import PageHinkley
from river.drift import PageHinkley
from utility.memory import Memory

class PH(PageHinkley, Memory):
    def __init__(self,min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10):
        super().__init__(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001)
        Memory.__init__(self)

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

    def detect(self, error):
        self.update(error)
        self.change_flag = self.drift_detected
        self.update_memory()
        return False, self.change_flag

    def update_memory(self):
        if self.change_flag:
            self.forget_before(self.min_memory_len)

    def reset_detector(self):
        self.reset()
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []


    def update_online_model(self, X, y, fit_base_learner=True):
        self.add_sample(X, y)
        if self.base_learner_is_fitted:
            y_hat = self.predict_online_model(X)
        else:
            y_hat = 0
        error = self.mean_absoulte_error(y, y_hat)
        self.detect(error)
        if fit_base_learner:
            self.fit_to_memory()
