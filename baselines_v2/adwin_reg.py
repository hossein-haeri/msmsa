import numpy as np
# from skmultiflow.drift_detection import ADWIN as AdaptiveWindowing
from river.drift import ADWIN as AdaptiveWindowing

class ADWIN(AdaptiveWindowing):
    def __init__(self,delta=0.002):
        super().__init__(delta=delta)
        self.model = None
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'ADWIN'
        self.hyperparams = {'delta':delta
                            }
    def add_sample(self, sample):
        self.memory.append(sample)

    def detect_(self, error):
        self.update(np.absolute(error))
        self.change_flag = self.drift_detected
        self.update_memory()
        return False, self.change_flag

    def update_memory(self):
        if self.change_flag:
            # self.memory = self.memory[-self._width:]
            w = (int(self.width))
            self.memory = self.memory[-w:]

    def get_recent_data(self):
        return self.memory
    
    def reset_detector(self):
        self.reset()
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        
    def update_online_model(self, sample):
        self.add_sample(sample)
        y = sample[1]
        y_hat = self.predict_online_model(sample[0])
        error = self.mean_absoulte_error(y, y_hat)
        self.detect_(error)
        self.model.reset()
        self.model.fit(self.memory)
        return None
    
    def predict_online_model(self, X, y):
        return self.model.predict(X)
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))