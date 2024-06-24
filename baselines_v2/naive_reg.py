import numpy as np
from utility.memory import Memory

class Naive(Memory):
    def __init__(self):
        super().__init__()
        self.method_name = 'Naive'
        self.t = 0
        self.hyperparams = {
            }

    def update_memory(self):
        pass

    def reset_detector(self):
        self.__init__(self)

    def update_online_model(self, X, y, fit_base_learner=True):
        self.add_sample(X, y)
        self.t += 1
        if fit_base_learner:
            self.fit_to_memory()
