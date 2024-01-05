import numpy as np


class AVAR:
    def __init__(self, max_memory_size):
        self.memory = []
        self.z = [0]
        self.q = 10
        # list of potential recency horizons to be evaluated (is adaptively adjusted)
        self.taus = [1]
        # the maximum number of samples
        self.memory_horizon = max_memory_size
        self.recency_horizon = 0
        self.av = []
        self.method_name = 'DAVAR'
    def add_sample(self, sample):
        self.memory.append(sample)
        self.z.append(self.z[-1] + sample)

    def get_allanvar(self, y=None, taus=None, is_dynamic=False):
        # if signal is given entirely, first add to the memory one by one
        if y is not None:
            self.memory = []
            for sample in y:
                self.add_sample(sample)
        # if dynamic allan variance, forget sampels older than the memory horizon
        if is_dynamic:
            self.memory = self.memory[-self.memory_horizon:]
            self.z = self.z[-self.memory_horizon:]
        if taus is not None:
            self.taus = taus
        else:
            self.taus = list(range(1, min(int(self.memory_horizon / 2), int(len(self.memory) / 2))))
        self.av = []
        for m in self.taus:
            n = len(self.z)
            cumsum = 0
            cnt = 0
            for k in range(max(1, n - self.q * m), n - 2 * m):
                cumsum += (self.z[k + 2 * m] - 2 * self.z[k + m] + self.z[k]) ** 2
                cnt += 1

            self.av.append(cumsum / (2 * (m ** 2) * cnt))

    def get_recency_horizon(self):
        self.get_allanvar(is_dynamic=True)
        if len(self.av) <= 1:
            return 1
        indx = list(self.av).index(min(self.av))
        self.recency_horizon = self.taus[indx]
        self.memory_horizon = max(3 * self.recency_horizon, 10)
        return self.recency_horizon

    def detect(self, error):
        self.get_recency_horizon()
        return False, False

    def get_recent_data(self):
        return self.memory[-self.recency_horizon:]
    
    def reset_detector(self):
        self.memory = []
        self.z = [0]
        self.q = 10
        # list of potential recency horizons to be evaluated (is adaptively adjusted)
        self.taus = [1]
        self.recency_horizon = 0
        self.av = []