import numpy as np

class Sample:
    def __init__(self, X, y, t, id=None):
        self.id = id
        self.X = X
        self.y = y
        self.t = t
        self.expiration_probability = 0.1

        self.mu_o = None
        self.mu_c = None
        self.sigma_o = None
        self.sigma_c = None

    def X_with_time(self):
        return np.append(self.t, self.X)
    
    def X_with_current_time(self, current_time):
        return np.append(current_time, self.X)
    



class Memory:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)