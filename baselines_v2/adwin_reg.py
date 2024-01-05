import numpy as np
# from skmultiflow.drift_detection import ADWIN as AdaptiveWindowing
from river.drift import ADWIN as AdaptiveWindowing

class ADWIN(AdaptiveWindowing):
    def __init__(self,delta=0.002):
        super().__init__(delta=delta)
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'ADWIN'
        self.hyperparams = {'delta':delta
                            }
    def add_sample(self, sample):
        # self.add_element(sample[1])
        self.memory.append(sample)

    def detect_(self, error):
        # self.add_element(np.absolute(error))
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
        
    # def update(self, model, error):
    def update_(self, model, error):
        self.detect_(error)
        model.reset()
        model.fit(self.memory)
        return model, len(self.memory)