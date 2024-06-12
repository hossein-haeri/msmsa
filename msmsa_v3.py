import numpy as np
import learning_models
from collections import deque
import copy
import sys
from utility.memory import Memory
import pickle
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class MSMSA(Memory):

    def __init__(self, lam=0.8, min_memory_len=10, num_anchors = 100, max_horizon=4000):
        Memory.__init__(self)
        self.method_name = 'MSMSA'
        self.lam = lam
        self.min_memory_len = min_memory_len
        self.num_anchors = num_anchors
        self.t = 0
        # self.num_candids = 1000
        self.hyperparams = {'lam':lam,
                            'num_anchors': self.num_anchors,
                            'min_memory_len': self.min_memory_len,
                            'max_horizon': max_horizon,
                            }
        self.initialize_horizon_candidates(min_horizon=self.min_memory_len, max_horizon=max_horizon)
        
        # self.initialize_anchors()
        # self.models = [[]]*self.num_candids
        self.avars = np.empty([self.num_candids, self.num_anchors])
        self.avars[:] = np.nan
        self.first_sample = True



    def initialize_horizon_candidates(self, min_horizon, max_horizon):
        # self.hor_candids = list(np.unique([max(int(1.15**j), min_horizon) for j in range(1, self.num_candids+1)]))
        # self.hor_candids = [i for i in self.hor_candids if i <= max_horizon]
        candid = min_horizon
        self.hor_candids = []
        while candid <= max_horizon:
            self.hor_candids.append(candid)

            # candid = int(2*candid)
            candid = int(1.10*candid)
            # candid = candid + 1

        # self.hor_candids = np.arange(min_horizon, max_horizon, 1, dtype=int)
        # self.hor_candids = np.linspace(min_horizon, max_horizon, num=num_candids, dtype=int)
        # add 'linear_hor_candids' to the hyperparams
        self.num_candids = len(self.hor_candids)
        self.hyperparams['hor_candids'] = self.hor_candids
        self.hyperparams['num_candids'] = self.num_candids
        
        # self.hor_candids = np.array(self.hor_candids)
        

    def initialize_anchors(self, num_features, use_prior_anchors=None):
        if use_prior_anchors is not None:
            with open('melbourne_anchor_samples.pkl', 'rb') as f:
                anchors = pickle.load(f)
            # select a random subset of the prior anchors
            self.anchors = anchors['uniform']
        else:
            # self.anchors = np.random.uniform(low=0, high=1, size=(self.num_anchors, num_features))
            self.anchors = np.random.normal(0, scale=1, size=(self.num_anchors, num_features))
        self.anochor_preds = np.zeros((self.num_candids, self.num_anchors))

    def update_online_model(self, X, y, fit_base_learner=True):
        self.add_sample(X, y)
        if self.first_sample:
            self.initialize_anchors(X.shape[1])
            self.models = [copy.deepcopy(self.base_learner) for _ in range(self.num_candids)]
            self.first_sample = False

        self.t += 1
        for tau_indx, tau in enumerate(self.hor_candids):
            # if there are enough number of new samples to build a model across this horizon
            if self.t%tau == 0:
                # create a new model across this horizon (tau)
                self.models[tau_indx].fit(self.get_X(only_last=tau), self.get_y(only_last=tau))
                # get the prediction of the model at the anchor points
                new_anchor_preds = self.models[tau_indx].predict(self.anchors).squeeze()
                
                if not np.isnan(self.avars[tau_indx,0]): 
                    # update the allan variance value for this horizon
                    self.avars[tau_indx,:] = (1-self.lam) * self.avars[tau_indx,:] + self.lam  * (new_anchor_preds - self.anochor_preds[tau_indx, :])**2
                else: # if this is the first time we are updating the allan variance just use the new predictions (no need for rolling exp average)
                    self.avars[tau_indx,:] = (new_anchor_preds - self.anochor_preds[tau_indx, :])**2
                
                # update the anchor predictions
                self.anochor_preds[tau_indx, :] = new_anchor_preds

        # average between all anchor points
        self.avars_scalarized = np.mean(self.avars, axis=1)
        self.update_validity_horizon()
        
        if fit_base_learner:
            # fit the base learner only for the samples in the memory that are within the validity horizon
            self.fit_to_memory(only_last=self.validity_horizon)


    def update_validity_horizon(self):
        if not all(np.isnan(v) for v in self.avars_scalarized): # check for warm start condition
            idx = self.index_of_minimum(self.avars_scalarized)
            self.validity_horizon_index = max(idx+1, len(self.hor_candids)-1)
            self.validity_horizon = self.hor_candids[idx]

        else: # means all values in avars are nan and hence we are in the pure cold start period
            self.validity_horizon = self.t


    def index_of_minimum(self, arr): 
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
            

        