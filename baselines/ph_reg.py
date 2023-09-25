import numpy as np
# from skmultiflow.drift_detection import PageHinkley
from river.drift import PageHinkley

class PH(PageHinkley):
    def __init__(self,min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001, min_memory_len=10):
        super().__init__(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001)
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        self.method_name = 'PH'
        self.min_memory_len = min_memory_len

        self.hyperparams = {
                            }

    def add_sample(self, sample):
        # self.add_element(sample[1])
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
        model.reset()
        model.fit(self.memory)
        return model, len(self.memory)