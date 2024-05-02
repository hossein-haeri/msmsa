import numpy as np
# from skmultiflow.drift_detection import ADWIN as AdaptiveWindowing
from river.drift import ADWIN as AdaptiveWindowing
from utility.memory import Memory

class ADWIN(AdaptiveWindowing, Memory):
    def __init__(self,delta=0.002):
        super().__init__(delta=delta)
        Memory.__init__(self)
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'ADWIN'
        self.hyperparams = {'delta':delta
                            # 'min_memory_len':10
                            }
        
    # def add_sample(self, X, y):
    #     self.memory.append((X, y))

    def detect_(self, error):
        self.update(np.absolute(error))
        self.change_flag = self.drift_detected
        self.update_memory()
        return False, self.change_flag

    def update_memory(self):
        if self.change_flag:
            w = (int(self.width))
            # self.memory = self.memory[-w:]
            self.forget_before(w)

    # def get_recent_data(self):
    #     return self.memory
    
    # def get_val_horizon(self):
    #     return len(self.memory)
    
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
        self.detect_(error)
        if fit_base_learner:
            self.fit_to_memory()
        # self.base_learner.reset()
        # self.base_learner.fit(self.memory)
        # if len(self.memory) > 1:
        #     self.base_learner_is_fitted = True
    
    # def predict_online_model(self, X):
    #     return self.base_learner.predict(X)
    
    
    # def mean_absoulte_error(self, y_true, y_pred):
    #     return np.mean(np.absolute(y_true - y_pred))