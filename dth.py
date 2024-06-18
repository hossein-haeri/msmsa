
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

        
class DTH(Memory):
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self,
                 epsilon=0.9,
                 prior=0.5,
                 num_sub_predictions=20,
                 min_memory_len=10,

                 ):
        super().__init__()

        ### hyper-parameters
        self.epsilon = epsilon
        self.prior = prior
        self.sub_prediction_type = 'no-subprediction' # 'model_memory'/ 'sub_learners' / 'no-subprediction'
        self.num_sub_predictions = num_sub_predictions
        self.min_memory_len = min_memory_len
        self.max_elimination_per_pruning = 10
        # self.max_assessments_per_pruning = 100
        self.hyperparams = {'epsilon':epsilon,
                            'prior':prior,
                            'num_sub_predictions':num_sub_predictions,
                            'min_memory_len':min_memory_len,
                            'max_elimination_per_pruning':self.max_elimination_per_pruning,
                            # 'num_eliminations_per_assessment':self.num_eliminations_per_assessment,
                            # 'max_assessments_per_pruning':self.max_assessments_per_pruning,

                          }
        #### initialization
        self.method_name = 'DTH'
        self.sample_id_counter = 0
        self.current_time = 0
        # self.first_time = True
        self.max_model_memory_len = self.num_sub_predictions
        self.model_memory = deque(maxlen=self.max_model_memory_len)
        
        # make prior a numpy array of size 
        self.prior

    def get_X_with_current_time(self):
        X_with_current_time = self.get_X()
        X_with_current_time[:, 0] = np.full_like(X_with_current_time[:, 0], self.current_time)
        return X_with_current_time
    
    def update_online_model(self, X, y, fit_base_learner=True, prune_memory=True):
        
        self.add_sample(X, y)
        if fit_base_learner:
            self.fit_to_memory()
            if self.sub_prediction_type == 'model_memory':
                self.model_memory.append(copy.deepcopy(self.base_learner))
            if prune_memory:
                    max_eliminations = min(max(0, self.get_num_samples() - self.min_memory_len), self.max_elimination_per_pruning)
                    if max_eliminations > 0:
                        self.prune_memory(max_eliminations)
                        # pass

    # def fit_base_learner(self):
    #         self.fit_to_memory()
    
    # def prune_memory_old(self):

    #     elimination_prob = self.assess_memory()

    #     # Create a boolean array where the condition is True for samples that should be deactivated
    #     to_deactivate = (elimination_prob > self.epsilon)
    #     # count the number of True values in to_deactivate
    #     if len(to_deactivate) > 0:
    #         self.ratio_deactivate = np.sum(to_deactivate)/len(to_deactivate)
    #     # get the indices of the active samples in memory
    #     actives_ind = np.where(self.is_actives)[0]
    #     to_deactivate
    #     if len(to_deactivate) > 0:

    #         # for i, memory_indx in enumerate(actives_ind[to_deactivate]):
    #         #     if len(actives_ind)-i > self.min_memory_len:
    #         #         self.is_actives[memory_indx] = False

    #         num_eliminations = 5
    #         # Get the indices of the samples to deactivate, sorted by their elimination probabilities in descending order
    #         to_deactivate_indices = np.argsort(elimination_prob[to_deactivate])[::-1][:num_eliminations]

    #                 # Deactivate the top 5 samples based on their elimination probabilities
    #         for i in to_deactivate_indices:
    #             if len(actives_ind) - np.sum(~self.is_actives) > self.min_memory_len:
    #                 self.is_actives[actives_ind[to_deactivate][i]] = False

    def prune_memory(self, max_eliminations):
        eliminated_count = 0
        
        elimination_prob = self.assess_memory()

        # Create a boolean array where the condition is True for samples that should be eliminated (deactivated)
        to_deactivate = (elimination_prob > self.epsilon)

        
        # to_deactivate_indices = np.argsort(elimination_prob[to_deactivate])[::-1][:max_eliminations]

        # Check the number of samples eligible for elimination
        # num_eligibles = np.sum(to_deactivate)
        
        # Get the indices of the active samples in memory
        actives_ind = np.flatnonzero(self.is_actives)
        

        for i, memory_indx in enumerate(actives_ind[to_deactivate]):
            if i < max_eliminations:
                self.is_actives[memory_indx] = False


        # Deactivate the top 5 samples based on their elimination probabilities
        # for i in to_deactivate_indices:
        #     # if len(actives_ind) - np.sum(~self.is_actives) > self.min_memory_len:
        #     self.is_actives[actives_ind[to_deactivate][i]] = False
        #     eliminated_count += 1


    def assess_memory(self, X_with_t_o=None, y=None):

        if X_with_t_o is None:
            X_with_t_o = self.get_X()
            X_with_t_c = self.get_X_with_current_time()
            y = self.get_y()
        else:
            X_with_t_c = X_with_t_o
            X_with_t_c[:, 0] = np.full_like(X_with_t_c[:, 0], self.current_time)

        if self.sub_prediction_type == 'no-subprediction': # deterministic prediction
            y_hat_i = self.predict_online_model(X_with_t_o)[0]
            y_hat_c = self.predict_online_model(X_with_t_c)[0]

            d_i = np.linalg.norm(y_hat_i - y)
            d_c = np.linalg.norm(y_hat_c - y)
            prb_y_given_x_and_t_i = 1 - d_i/(d_i + d_c)

        else: # probablistic prediction
            mu_i, sigma_i = self.predict_bulk(X_with_t_o)
            mu_c, sigma_c = self.predict_bulk(X_with_t_c)
            mu_i = self.predict_online_model(X_with_t_o)[0]
            mu_c = self.predict_online_model(X_with_t_c)[0]
            
            sigma_i = np.maximum(sigma_i, 1e-9)
            sigma_c = np.maximum(sigma_c, 1e-9)

            prob_y_c  = np.exp(-0.5 * ((y - mu_c)/sigma_c)**2) / (sigma_c)
            prob_y_i = np.exp(-0.5 * ((y - mu_i)/sigma_i)**2) / (sigma_i)
            prob_y_c = np.maximum(prob_y_c, 1e-9)
            prob_y_i = np.maximum(prob_y_i, 1e-9)
            prb_y_given_x_and_t_i = (prob_y_i * self.prior) / (prob_y_i * self.prior + prob_y_c * (1 - self.prior))
        
        return prb_y_given_x_and_t_i


    def predict_bulk(self, X_batch_with_time):
        if self.base_learner.__class__.__name__ == 'RandomForestRegressor' and self.sub_prediction_type == 'sub_learners':

            scaler = self.base_learner.named_steps['standardscaler']
            rf = self.base_learner.named_steps['randomforestregressor']

                        # Transform the input data using the scaler
            X_scaled = scaler.transform(X_batch_with_time)

            # Get predictions from each individual tree and stack them into a single array
            tree_predictions = np.array([tree.predict(X_scaled) for tree in rf.estimators_])

            # Compute the mean and standard deviation of the predictions
            mean_predictions = np.mean(tree_predictions, axis=0)
            std_predictions = np.std(tree_predictions, axis=0)

            return mean_predictions, std_predictions

        elif self.base_learner.__class__.__name__ == 'RegressionNN':
            return self.base_learner.make_uncertain_predictions(X_batch_with_time, num_samples=self.num_sub_predictions)
            
        elif self.sub_prediction_type == 'model_memory':
            # return NotImplementedError
            # raise NotImplementedError
            # print('NotImplementedError: The base learner is not supported for the bulk prediction.')
            scaler = self.base_learner.named_steps['standardscaler']
            X_scaled = scaler.transform(X_batch_with_time)

            y_pred = np.zeros(len(self.model_memory))
            for i, model in enumerate(self.model_memory):
                y_pred[i] = model.predict(X_scaled)[0]
            
            # Compute the mean and standard deviation of the predictions
            mean_predictions = np.mean(y_pred, axis=0)
            std_predictions = np.std(y_pred, axis=0)

            return mean_predictions, std_predictions


    



