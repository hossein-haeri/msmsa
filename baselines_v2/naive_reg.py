import numpy as np
from utility.sample import Memory

class Naive(Memory):
    def __init__(self):
        super().__init__()
        self.method_name = 'Naive'
        self.t = 0
        self.hyperparams = {
            }
   
    def detect(self, error):
        self.t += 1
        
    def update_memory(self):
        pass
    
    def reset(self):
        self.t = 0

    def reset_detector(self):
        self.__init__(self)
        
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
        # if len(self.samples) > 1:
        #     self.base_learner_is_fitted = True
