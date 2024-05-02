
import numpy as np
import copy
# import sys
# import random
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import Pool
# import torch
# import torch.nn as nn

# from utility.sample import Sample, Memory
from utility.memory import Memory
# import learning_models


        
class DTH(Memory):
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self, 
                 epsilon=0.9,
                 prior=0.5,
                 num_sub_predictions=50,
                 ):
        super().__init__()

        ### hyper-parameters
        self.epsilon = epsilon
        self.prior = prior
        self.num_sub_predictions = num_sub_predictions
        self.hyperparams = {'epsilon':epsilon,
                        'prior':prior,
                        'num_sub_predictions':num_sub_predictions
                          }
        #### initialization
        self.method_name = 'DTH'
        self.sample_id_counter = 0
        self.current_time = 0
        self.first_time = True
        self.model_memory = []

    def update_online_model(self, X, y, fit_base_learner=True, prune_memory=True):
        self.add_sample(X, y)
        if fit_base_learner:
            self.fit_base_learner()
            self.first_time = False
            # if prune_memory and len(self.model_memory) >= 50:
            if prune_memory and self.get_num_samples() >= 10:
                    self.prune_memory()

    def fit_base_learner(self):

            self.fit_to_memory()

            # self.model_memory.append(copy.deepcopy(self.base_learner))
            # if len(self.model_memory) > 50:
            #     self.model_memory.pop(0)
    
    def prune_memory(self):

        X_with_t_o = self.get_X_with_time()
        X_with_t_c = self.get_X_with_current_time()
        y = self.get_y()

        mu_o, sigma_o = self.predict_bulk(X_with_t_o)
        mu_c, sigma_c = self.predict_bulk(X_with_t_c)

        # print(mu_o.shape, sigma_o.shape, mu_c.shape, sigma_c.shape)
        sigma_o = np.maximum(sigma_o, 1e-6)
        sigma_c = np.maximum(sigma_c, 1e-6)

        prob_y_current  = np.exp(-0.5 * ((y - mu_c)/sigma_c)**2) / (sigma_c)
        prob_y_original = np.exp(-0.5 * ((y - mu_o)/sigma_o)**2) / (sigma_o)

        prob_y_current = np.maximum(prob_y_current, 1e-6)
        prob_y_original = np.maximum(prob_y_original, 1e-6)
        

        prob_original_given_y = (prob_y_original * self.prior) / (prob_y_original * self.prior + prob_y_current * (1 - self.prior))
        

        # Create a boolean array where the condition is True for samples that should be deactivated
        to_deactivate = (prob_original_given_y > self.epsilon)

        # get the indices of the active samples
        actives = np.where(self.active_indices)[0]
        if len(to_deactivate) > 0:
            for i in actives[to_deactivate]:
                self.active_indices[i] = False


    def predict_bulk(self, X_batch_with_time):
        if self.base_learner.__class__.__name__ == 'RandomForestRegressor':
            tree_predictions =  [tree.predict(X_batch_with_time) for tree in self.base_learner.estimators_]
            return np.mean(tree_predictions, axis=0), np.std(tree_predictions, axis=0)

        elif self.base_learner.__class__.__name__ == 'RegressionNN':
            return self.base_learner.make_uncertain_predictions(X_batch_with_time, num_samples=self.num_sub_predictions)
            
        else:
            # return NotImplementedError
            raise NotImplementedError

            # y_pred = np.zeros(len(self.model_memory))
            # for i, model in enumerate(self.model_memory):
            #     y_pred[i] = model.predict(X_batch_with_time)[0]
            # return y_pred[-1], np.std(y_pred)


    



