
import numpy as np
import learning_models
import copy
import sys
import torch
import torch.nn as nn

# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt

class DTH:

    def __init__(self, num_features):
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.memory = []
        self.train_memory = []





    def add_sample(self,X, y):
        self.memory.append((X, y))
        if len(self.memory) > self.memory_size+1:
            self.memory[-self.memory_size:]


    def update_online_model(self, X, y):
        self.add_sample(X, y)
        self.update_train_memory()
        self.base_learner.reset()
        self.base_learner.fit(self.train_memory)
        return None
    

    def predict_online_model(self, X):
        return self.base_learner.predict(X)
    

    def update_train_memory(self):
        self.train_memory = self.memory[-self.memory_size:]





