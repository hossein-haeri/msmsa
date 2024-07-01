import numpy as np
import learning_models
from collections import deque
import copy
import sys
from utility.memory import Memory
import pickle
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

class MSMSA(Memory):

    def __init__(self, lam=0.8, min_memory_len=10, num_anchors = 100, max_horizon=5000):
        Memory.__init__(self)
        self.method_name = 'MSMSA'
        self.lam = lam
        self.min_memory_len = min_memory_len
        self.num_anchors = num_anchors
        self.t = 0
        self.b = None
        self.use_prior_anchors = 'normal'
        self.hyperparams = {'lam':lam,
                            'num_anchors': self.num_anchors,
                            'min_memory_len': self.min_memory_len,
                            'max_horizon': max_horizon,
                            'b': self.b,
                            'anchors_distribution': self.use_prior_anchors,
                            'method_name':self.method_name
                            }
        self.initialize_horizon_candidates(min_horizon=self.min_memory_len, max_horizon=max_horizon)
        
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
            if self.b is None:
                candid = candid = candid + 1
            elif self.b > 1:
                candid = int(self.b*candid)
            else:
                raise ValueError('b should be greater than 1 (exponental) or None (all possible horizons)')


        # self.hor_candids = np.arange(min_horizon, max_horizon, 1, dtype=int)
        # self.hor_candids = np.linspace(min_horizon, max_horizon, num=num_candids, dtype=int)
        # add 'linear_hor_candids' to the hyperparams
        self.num_candids = len(self.hor_candids)
        self.hyperparams['hor_candids'] = self.hor_candids
        self.hyperparams['num_candids'] = self.num_candids
        
        self.hor_candids = np.array(self.hor_candids)
    
    def initialize_anchors(self, data_X=None, feature_bounds=None):
        # feature_bounds is a 2D array with shape (num_features, 2) where the first column is the min value of each feature and the second column is the max value of each feature
        if data_X is None and feature_bounds is None:
            raise ValueError('Either data_X or feature_bounds should be provided')
        if data_X is None and feature_bounds is not None:
            num_features = len(feature_bounds)
            self.anchors = np.random.uniform(feature_bounds[:,0], feature_bounds[:,1], (self.num_anchors, num_features))
        if data_X is not None:
            num_features = data_X.shape[1]
            if self.use_prior_anchors == 'uniform':
                # get the min and max values of each feature in data_X
                min_values = np.min(data_X, axis=0)
                max_values = np.max(data_X, axis=0)

                # create the anchors by sampling uniformly from the min and max values
                self.anchors = np.random.uniform(min_values, max_values, (self.num_anchors, num_features))
            if self.use_prior_anchors == 'normal':
                # get the mean and std values of each feature in data_X
                mean_values = np.mean(data_X, axis=0)
                std_values = np.std(data_X, axis=0)
                # create the anchors by sampling normally from the mean and std values
                self.anchors = np.random.normal(mean_values, std_values, (self.num_anchors, num_features))
            if self.use_prior_anchors == 'exact':
                # sample n random samples from the dataset without replacement
                self.anchors = data_X[np.random.choice(data_X.shape[0], self.num_anchors, replace=False), :]
        

        self.previous_anchor_preds = np.empty([self.num_candids, self.num_anchors])
        self.previous_anchor_preds[:] = np.nan
        
        
        self.avars_scalarized = np.empty((self.num_candids))
        self.avars_scalarized[:] = np.nan
        




    def update_online_model(self, X, y, fit_base_learner=True):
        # # drop the first feature which is the time
        # X = X[:,1:]
        self.add_sample(X, y)
        if self.first_sample:
            # self.initialize_anchors(X.shape[1])
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
                    self.avars[tau_indx,:] = (1-self.lam) * self.avars[tau_indx,:] + self.lam  * (new_anchor_preds - self.previous_anchor_preds[tau_indx, :])**2
                else: # if this is the first time we are updating the allan variance just use the new predictions (no need for rolling exp average)
                    self.avars[tau_indx,:] = (new_anchor_preds - self.previous_anchor_preds[tau_indx, :])**2

                self.previous_anchor_preds[tau_indx, :] = new_anchor_preds

                self.avars_scalarized[tau_indx] = np.mean(self.avars[tau_indx, :])


        # print('ts: ', self.t, 'avars: \n', tabulate(self.avars, tablefmt="grid"))
        # average between all anchor points

        # self.avars_scalarized = np.mean(self.avars, axis=1)
        # append the self.avars_scalarized vector to the msmsa_avars.csv file (create one if it does not exist)
        # with open('msmsa_avars.csv', 'a') as f:
        #     # np.savetxt(f, self.avars_scalarized.reshape(1,-1), delimiter=',')
        #     np.savetxt(f, self.avars_scalarized.reshape(1,-1), delimiter=',')
        
        # update the validity horizon
        self.update_validity_horizon()
        
        if fit_base_learner:
            # fit the base learner only for the samples in the memory that are within the validity horizon
            self.fit_to_memory(only_last=self.validity_horizon)


    def update_validity_horizon(self):
        if not all(np.isnan(v) for v in self.avars_scalarized): # check for warm start condition
            # idx = self.index_of_minimum(self.avars_scalarized)
            # self.validity_horizon_index = max(idx+1, len(self.hor_candids)-1)
            # self.validity_horizon = self.hor_candids[idx]

            min_tau, min_tau_index = fit_sigma_function_and_find_min_index(self.hor_candids, self.avars_scalarized)
            self.validity_horizon = min(self.t, min_tau)

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
            

def fit_sigma_function_and_find_min_index(hor_candids, avars_scalarized):
    """
    Fit the function sigma(tau) = theta_1 * tau + theta_2 * (1 / tau)
    and find the index of the minimum sigma value using the fitted model.
    Ignores None or NaN values in avars_scalarized.
    
    Parameters:
    hor_candids (np.ndarray): The tau values.
    avars_scalarized (np.ndarray): The corresponding sigma(tau) values, with potential None or NaN.
    
    Returns:
    tuple: (theta, min_index) where theta is np.ndarray [theta_1, theta_2]
           and min_index is the index of the minimum sigma value in the filtered dataset.
    """
    # Convert avars_scalarized to numpy array and ensure hor_candids is a numpy array
    avars_scalarized = np.array(avars_scalarized)
    hor_candids = np.array(hor_candids)
    
    # Filter out None or NaN values
    valid_indices = ~np.isnan(avars_scalarized) & ~np.isinf(avars_scalarized)
    valid_hor_candids = hor_candids[valid_indices]
    valid_avars_scalarized = avars_scalarized[valid_indices]
    
    # Create the design matrix for the valid values
    A = np.vstack([valid_hor_candids, 1 / valid_hor_candids]).T
    
    # Solve for theta using least squares on valid values
    theta, _, _, _ = np.linalg.lstsq(A, valid_avars_scalarized, rcond=None)
    
    # Compute sigma(tau) using the fitted model for all hor_candids
    sigma_fitted = theta[0] * hor_candids + theta[1] / hor_candids
    

    # Find the index of the minimum sigma value
    min_index = np.argmin(sigma_fitted)
    min_tau = hor_candids[min_index]
    
    return min_tau, min_index
    