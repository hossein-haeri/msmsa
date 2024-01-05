import numpy as np

class Naive:
    def __init__(self):
        self.memory = []
        self.method_name = 'Naive'
        self.t = 0
        self.hyperparams = {
            }

    def add_sample(self, sample):
        self.memory.append(sample)
        
    def detect(self, error):
        self.t += 1
        
    def update_memory(self):
        pass
    
    def reset(self):
        self.t = 0

    def reset_detector(self):
        self.__init__(self)
        

    def get_recent_data(self):
        return self.memory

    def update_(self, model, error):
        self.detect(error)
        model.reset()
        model.fit(self.memory)
        return model, self.t
