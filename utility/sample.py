import numpy as np

class Sample:
    def __init__(self, X, y, t, id=None):
        self.id = id
        self.X = X
        self.y = y
        self.t = t

    def X_with_time(self):
        return np.append(self.X, self.t)

