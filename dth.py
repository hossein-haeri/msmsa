
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
                 ):
        super().__init__()

        ### hyper-parameters
        self.epsilon = epsilon
        self.prior = prior
        self.hyperparams = {'epsilon':epsilon,
                        'prior':prior,
                          }
        #### initialization
        self.method_name = 'DTH'
        self.sample_id_counter = 0
        self.current_time = 0
        self.first_time = True
        self.model_memory = []

    def update_online_model(self, X, y, fit_base_learner=True):
        self.add_sample(X, y)
        if fit_base_learner:
            self.fit_base_learner()
            self.first_time = False
            if len(self.model_memory) >= 50:
                self.prune_memory()

    def fit_base_learner(self):
            # X = self.get_X_with_time()
            # y = self.get_y()
            # self.base_learner.model.fit(X, y)
            # self.base_learner_is_fitted = True
            self.fit_to_memory()
            self.model_memory.append(copy.deepcopy(self.base_learner.model))
            if len(self.model_memory) > 50:
                self.model_memory.pop(0)
    
    def prune_memory(self):

        X_with_t_o = self.get_X_with_time()
        X_with_t_c = self.get_X_with_current_time()
        y = self.get_y()

        mu_o, sigma_o = self.predict_bulk(X_with_t_o)
        mu_c, sigma_c = self.predict_bulk(X_with_t_c)

        sigma_o = np.maximum(sigma_o, 1e-6)
        sigma_c = np.maximum(sigma_c, 1e-6)

        prob_y_current  = np.exp(-0.5 * ((y - mu_c)/sigma_c)**2) / (sigma_c)
        prob_y_original = np.exp(-0.5 * ((y - mu_o)/sigma_o)**2) / (sigma_o)

        prob_y_current = np.maximum(prob_y_current, 1e-6)
        prob_y_original = np.maximum(prob_y_original, 1e-6)
        
        # prior = np.array([sample.expiration_probability for sample in self.samples])
        # prior = np.minimum(prior, 0.9)
        # prior = np.maximum(prior, 0.1)

        # print('prior:', prior)
        # print('prob_y_current:', prob_y_current)
        # print('prob_y_original:', prob_y_original)

        # prior = self.prior
        prob_original_given_y = (prob_y_original * self.prior) / (prob_y_original * self.prior + prob_y_current * (1 - self.prior))
        
        # self.prior = prob_original_given_y
        # # print('prob_y_current:', prob_original_given_y)
        # num_removed = 0
        # for i, sample in enumerate(self.samples):
        #     sample.expiration_probability = prob_original_given_y[i]
        #     # if prob_original_given_y[i] > self.epsilon and sigma_c[i] < np.mean(sigma_c):
        #     if prob_original_given_y[i] > self.epsilon:
        #         if num_removed > 5:
        #             self.fit_base_learner()
        #             num_removed = 0
        #         # print('removing:', i, prob_original_given_y[i], sigma_c[i])
        #         self.samples.pop(i)
        #         num_removed += 1

        # Create a boolean array where the condition is True for samples that should be deactivated
        to_deactivate = (prob_original_given_y > self.epsilon)

        # get the indices of the active samples
        actives = np.where(self.active_indices)[0]
        if len(to_deactivate) > 0:
            for i in actives[to_deactivate]:
                self.active_indices[i] = False


    def predict_bulk(self, X_batch_with_time):
        # if self.base_learner.model.__class__.__name__ == 'RandomForestRegressor':
        #     # print('X_batch_with_time shape:', X_batch_with_time.shape)
        #     return self.base_learner.get_sub_predictions(X_batch_with_time)
        # else:
        #     error = 'predict_bulk is not implemented for base_learner: ' + self.base_learner.model.__class__.__name__
        #     raise NotImplementedError(error)

        y_pred = np.zeros(len(self.model_memory))
        for i, model in enumerate(self.model_memory):
            y_pred[i] = model.predict(X_batch_with_time)[0]
        return y_pred[-1], np.std(y_pred)
        # return np.mean(y_pred), np.std(y_pred)
    



