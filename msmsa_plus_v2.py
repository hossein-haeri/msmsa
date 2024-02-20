
import numpy as np
import learning_models
from collections import deque
import copy
import sys

from scipy.ndimage import gaussian_filter


class MSMSA_plus:

    def __init__(self, anchor_samples=None, lam=0.1, min_memory_len=10, update_freq_factor=1, num_anchors = 100, max_horizon=1000):
        self.anchor_samples = anchor_samples
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.t = 0
        self.num_candids = 100
        self.num_anchors = num_anchors
        self.hor_candids = np.unique([max(int(1.15**j), min_memory_len) for j in range(1, self.num_candids+1)])
        # self.hor_candids = np.unique([max(int(2**(j)), min_memory_len) for j in range(1, self.num_candids+1)])
        # drop horizon candidates that are greater than max_horizon
        self.hor_candids = self.hor_candids[self.hor_candids <= max_horizon]
        # print(self.hor_candids)
        # self.num_candids = 500
        # self.hor_candids = range(min_memory_len,self.num_candids)
        # print(self.hor_candids)
        self.num_candids = len(self.hor_candids)
        self.validity_horizon = np.ones(self.num_anchors, dtype=int)
        self.memory_size = np.max(self.hor_candids)
        self.errors = []
        self.memory = []
        self.models = [[]]*self.num_candids
        self.models_ = [[]]*self.num_anchors
        self.method_name = 'MSMSA+'
        self.anchors = None
        self.first_sample = True
        self.num_features = None
        self.update_freq_factor = update_freq_factor  # every tau/update_freq_factor timestep a new model is trained
        self.lam = lam
        self.avars = np.empty([self.num_candids, self.num_anchors])
        self.avars[:] = np.nan
        self.max_indicator_memory = 2*max(self.hor_candids)
        self.indicators = np.empty(shape=(self.max_indicator_memory,self.num_candids,self.num_anchors))
        self.indicators[:] = np.nan
        self.hyperparams = {'lam':lam,
                            'num_anchors': self.num_anchors,
                            'update_freq_factor': update_freq_factor,
                            }

    def add_sample(self,X, y):
        if self.first_sample:
            self.num_features = X.shape[0]
            self.initialize_anchors()
            self.first_sample = False

        self.memory.append((X, y))
        if len(self.memory) > self.memory_size+1:
            self.memory[-self.memory_size:]
        self.t += 1

    def update_online_model(self, X, y):
        self.add_sample(X, y)
        if self.t > 1:
            for i, tau in enumerate(self.hor_candids):
                update_period = max(1,int(tau/self.update_freq_factor))
                # if self.t%update_period == 0 or self.update_freq_factor == -1:
                if self.t%tau == 0 or self.update_freq_factor == -1:
                
                    # train a new model using last tau samples and append it to the right side of the que
                    self.base_learner.reset()
                    self.base_learner.fit(self.memory[-tau:])
                    self.models[i] = self.base_learner

                    # calculate model indicators and store it
                    current_indicators = self.get_model_indicators(self.base_learner)
                    
                    # self.indicators[self.t%self.max_indicator_memory, i, :] = current_indicators
                    

                    # recall model indicators calculated tau timestep before
                    if self.update_freq_factor == -1:
                        previous_indicators = self.indicators[(self.t%self.max_indicator_memory)-tau, i, :]
                    else:
                        # print(self.indicators[(self.t%self.max_indicator_memory), i, :])
                        previous_indicators = self.indicators[(self.t%self.max_indicator_memory)-1, i, :]
                    
                    previous_indicators = self.indicators[0, i, :]
                    # print('current_indicators: ',current_indicators[:10])
                    # print('previous_indicators: ',previous_indicators[:10])

                    if not np.isnan(self.avars[i,0]):
                        self.avars[i,:] = (1-self.lam) * self.avars[i,:] + self.lam  * (current_indicators - previous_indicators)**2
                    else:
                        self.avars[i,:] = (current_indicators - previous_indicators)**2

                    self.indicators[0, i, :] = current_indicators

                    


        for k in range(self.num_anchors):
            # find the minimum and the validity horizon
            if not all(np.isnan(v) for v in self.avars[:,k]): # check for warm start condition
                idx = self.index_of_minimum(self.avars[:,k])
                # print('\n PRINT------------',self.validity_horizon)

                self.validity_horizon[k] = min(self.t, self.hor_candids[idx])
                # self.base_learner.reset()
                # self.base_learner.fit(self.memory[-self.validity_horizon[k]:])
                # self.models_[k] = self.base_learner
                # return model, self.validity_horizon[k]
            else: # means all values in avars are nan and hence we are in the cold start period
                self.validity_horizon[k] = self.t
                # self.base_learner.reset()
                # self.base_learner.fit(self.memory[-self.validity_horizon[k]:])
                # self.models_[k] = self.base_learner
            # self.models[k] = self.base_learner
        return None

    

    def get_allan_variance(self, params):
        params = np.array(params)
        avar = []
        if len(params) < 2:
            return None
        for i in range(len(params)-1):
            avar.append( ((params[i,:] - params[i+1, :])**2).mean() )
        return np.mean(avar)
    

    def replace_nan_with_max(self, x):
        if np.isnan(x).all():
            return x
        else:
            max_val = np.nanmax(x)
            x[np.isnan(x)] = max_val
            return x


    def get_avar_between_models(self, model, model_):
        """ compares the output of the anchor points across two consective models"""
        Y_ = model_.model.predict(self.anchors)
        Y = model.model.predict(self.anchors)
        avars = (Y_ - Y)**2 / len(Y)
        return avars
    

    def get_model_indicators(self, model):
        """ calculates the indicators by evaluating the model across anchor points"""
        return model.model.predict(self.anchors)

    def initialize_anchors(self):
        # self.anchors = np.random.uniform(low=-1, high=1, size=(self.num_anchors, self.num_features))
        if self.anchor_samples is not None:
            # sample the anchor points from the anchor_samples wihtout replacement if number of anchor points is less than the number of samples
            if self.num_anchors < self.anchor_samples.shape[0]:
                idx = np.random.choice(self.anchor_samples.shape[0], self.num_anchors, replace=False)
                self.anchors = self.anchor_samples[idx]
            else:
                self.anchors = self.anchor_samples
        else:
            self.anchors = np.random.normal(0, scale=1, size=(self.num_anchors, self.num_features))

        # sample 


    def index_of_minimum(self, arr): # written by chat-GPT!
        # create a boolean mask to exclude NaN values
        mask = ~np.isnan(arr)
        # use the masked array to find the index of the minimum value
        min_index = np.nanargmin(arr[mask])
        # use the mask to adjust the index to the original array
        adjusted_index = np.flatnonzero(mask)[min_index]
        # loop over the array to check if there are multiple occurrences of the minimum value
        for i in range(adjusted_index + 1, len(arr)):
            if arr[i] == arr[adjusted_index]:
                adjusted_index = i
        return adjusted_index
    

    def predict_online_model(self, X):
        # check and see which anchor point is closest to the input X size=(self.num_features)
        # calculate the distance between X and all anchor points
        dists = np.linalg.norm(self.anchors - X, axis=1)
        # find the index of the anchor point that is closest to X
        idx = np.argmin(dists)
        # identify the the validity horizon of the closest anchor point
        validity_horizon = self.validity_horizon[idx]

        # return self.model_[idx].predict(X)
        return self.model[idx].predict(X), validity_horizon
    
        
