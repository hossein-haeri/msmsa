import numpy as np
import copy
# from skmultiflow.drift_detection import KSWIN as KolmogorovSmirnovWIN

from river.drift import KSWIN as KolmogorovSmirnovWIN

class KSWIN(KolmogorovSmirnovWIN):
    def __init__(self,alpha=0.005, window_size=100, stat_size=30, min_memory_len=10):
        super().__init__(alpha=alpha, window_size=window_size, stat_size=stat_size)
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'KSWIN'
        self.min_memory_len = min_memory_len
        self.hyperparams = {
                    }

    def add_sample(self, sample):
        self.memory.append(sample)

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

    def update_(self, model, error):
        self.detect(error)
        # model_ = copy.copy(model)

        model.reset()
        model.fit(self.memory)
        return model, len(self.memory)
    
