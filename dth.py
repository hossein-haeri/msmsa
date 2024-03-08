
import numpy as np
import learning_models
import copy
import sys
import torch
import torch.nn as nn

# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt

class DTH:

    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self):
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.prune_learner = learning_models.DecissionTree()
        self.memory = []
        # self.train_memory = []
        self.current_time = 0
        self.time_in_features = True
        self.method_name = 'DTH'



    def add_sample(self,X, y):
        if self.time_in_features:
            sample_time = X[0]
            if self.current_time < sample_time:
                self.current_time = sample_time
            
        self.memory.append((X, y))
        # self.train_memory.append((X, y))
        # if len(self.memory) > self.memory_size+1:
        #     self.memory[-self.memory_size:]


    def update_online_model(self, X, y):
        self.add_sample(X, y)
        self.prune_memory()
        self.base_learner.reset()
        self.base_learner.fit(self.memory)
        self.base_learner_is_fitted = True
        return None
    

    def predict_online_model(self, X):
        return self.base_learner.predict(X)
    

    def prune_memory(self):
        ''' Assuming that samples in memory are sorted in time '''

        # shuffle the the sample objects in memory
        np.random.shuffle(self.memory)

        # split the memory into num_prun_models batches
        num_prun_models = 5
        memory_batches = np.array_split(self.memory, num_prun_models)

        # fit a prune_learner to each part
        prune_learners = []
        for batch in memory_batches:
            prune_learner = copy.deepcopy(self.prune_learner)
            prune_learner.fit(batch)
            prune_learners.append(prune_learner)

        for sample in self.memory:
            for prune_learner in prune_learners:
                if not prune_learner.is_valid(sample):
                    self.memory.remove(sample)
                    break







    def train_horizon(self, X):
        ''' Place Holder '''
        return 100
        return self.current_time/2
    

    def predict_online_model(self, X):
        if self.base_learner_is_fitted:
            return self.base_learner.predict(X)
        else: # raise error 'no model is fitted'
            raise ValueError('model is not fitted yet bro!')
        





