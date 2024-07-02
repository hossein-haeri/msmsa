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
from collections import deque

        
class TMI(Memory):
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self,
                 epsilon=0.95,
                 prior=0.5,
                 num_sub_predictions=20,
                 min_memory_len=10,
                 probabilistic_prediction=False
                 ):
        super().__init__()

        ### hyper-parameters
        self.epsilon = epsilon
        self.prior = prior
        self.probabilistic_prediction = probabilistic_prediction # 'previously_trained_models'/ 'ensemble' / 'drop-out' / False
        self.num_sub_predictions = num_sub_predictions
        self.min_memory_len = min_memory_len
        self.max_elimination_per_pruning = 10


        #### initialization
        if self.probabilistic_prediction == False:
            self.method_name = 'TMI'
        elif self.probabilistic_prediction == 'previously_trained_models':
            self.method_name = 'PTMI-PTM'
        elif self.probabilistic_prediction == 'ensemble':
            self.method_name = 'PTMI-ENS'
        elif self.probabilistic_prediction == 'drop-out':
            self.method_name = 'PTMI-DO'
        
        self.sample_id_counter = 0
        self.current_time = 0
        # self.first_time = True
        self.max_model_memory_len = self.num_sub_predictions
        self.model_memory = deque(maxlen=self.max_model_memory_len)
        
        # make prior a numpy array of size 
        self.prior = prior

        self.hyperparams = {'epsilon':epsilon,
                            'prior':prior,
                            'num_sub_predictions':num_sub_predictions,
                            'min_memory_len':min_memory_len,
                            'max_elimination_per_pruning':self.max_elimination_per_pruning,
                            'probabilistic_prediction':self.probabilistic_prediction,
                            'method_name':self.method_name
                          }


    def get_X_with_current_time(self):
        X_with_current_time = self.get_X()
        X_with_current_time[:, 0] = np.full_like(X_with_current_time[:, 0], self.current_time)
        return X_with_current_time


    def update_online_model(self, X, y, fit_base_learner=True, prune_memory=True):
        
        self.add_sample(X, y)
        if fit_base_learner:
            self.fit_to_memory()
            if self.probabilistic_prediction == 'previously_trained_models':
                self.model_memory.append(copy.deepcopy(self.base_learner))
            if prune_memory:
                    max_eliminations = min(max(0, self.get_num_samples() - self.min_memory_len), self.max_elimination_per_pruning)
                    if max_eliminations > 0:
                        self.prune_memory(max_eliminations)


    def prune_memory(self, max_eliminations):
        elimination_prob = self.assess_memory()
        # Create a boolean array where the condition is True for samples that should be eliminated (deactivated)
        to_deactivate = (self.epsilon < elimination_prob) & (elimination_prob < 1)
        # Get the indices of the active samples in memory
        actives_ind = np.flatnonzero(self.is_actives)
        for i, memory_indx in enumerate(actives_ind[to_deactivate]):
            if i < max_eliminations:
                self.is_actives[memory_indx] = False


    def assess_memory(self, X_with_t_i=None, y=None):

        if X_with_t_i is None:
            X_with_t_i = self.get_X()
            X_with_t_c = self.get_X_with_current_time()
            y = self.get_y()
        else:
            X_with_t_c = X_with_t_i
            X_with_t_c[:, 0] = np.full_like(X_with_t_c[:, 0], self.current_time)


        if self.probabilistic_prediction == False: # deterministic prediction
            
            y_hat_i = self.predict_online_model(X_with_t_i)
            y_hat_c = self.predict_online_model(X_with_t_c)

            loss_i = np.abs(y_hat_i - y)
            loss_c = np.abs(y_hat_c - y)
            prb_y_given_x_and_t_i = (loss_c + 1e-9) / (loss_i + loss_c + 1e-9)


        else: # probablistic prediction
            mu_i, sigma_i = self.predict_bulk(X_with_t_i)
            mu_c, sigma_c = self.predict_bulk(X_with_t_c)

            mu_i = self.predict_online_model(X_with_t_i)
            mu_c = self.predict_online_model(X_with_t_c)
            
            sigma_i = np.maximum(sigma_i, 1e-5)
            sigma_c = np.maximum(sigma_c, 1e-5)

            # print('sigma_i: ', sigma_i)
            # print('sigma_c: ', sigma_c)

            # print('mu_i: ', mu_i)
            # print('mu_c: ', mu_c)

            prob_y_c  = np.exp(-0.5 * ( (y - mu_c)/sigma_c)**2 ) / (sigma_c)
            prob_y_i =  np.exp(-0.5 * ( (y - mu_i)/sigma_i)**2 ) / (sigma_i)

            prb_y_given_x_and_t_i = (prob_y_i) / (prob_y_i + prob_y_c)


            # print(prb_y_given_x_and_t_i)
        return prb_y_given_x_and_t_i


    def predict_bulk(self, X_batch_with_time):
        if self.probabilistic_prediction == 'ensemble':
            # Get predictions from each individual tree and stack them into a single array
            tree_predictions = np.array([tree.predict(X_batch_with_time) for tree in self.base_learner.estimators_])
            # Compute the mean and standard deviation of the predictions
            mean_predictions = np.mean(tree_predictions, axis=0)
            std_predictions = np.std(tree_predictions, axis=0)
            return mean_predictions, std_predictions
        
        elif self.probabilistic_prediction == 'drop-out':
            return self.base_learner.make_uncertain_predictions(X_batch_with_time, num_samples=self.num_sub_predictions)

        elif self.probabilistic_prediction == 'previously_trained_models':
            y_pred = np.zeros((len(self.model_memory), X_batch_with_time.shape[0]))
            for i, model in enumerate(self.model_memory):
                y_pred[i,:] = model.predict(X_batch_with_time)

            # Compute the mean and standard deviation of the predictions
            mean_predictions = np.mean(y_pred, axis=0)
            std_predictions = np.std(y_pred, axis=0)

            return mean_predictions, std_predictions
