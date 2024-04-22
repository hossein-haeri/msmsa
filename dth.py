
import numpy as np
import copy
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import torch
import torch.nn as nn

from utility.sample import Sample
import learning_models


        
class DTH:
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self, 
                 epsilon=0.9,
                 ):
        ### hyper-parameters
        self.base_learner = learning_models.RandomForest(n_estimators=100, max_depth=7, n_jobs=-1)
        self.epsilon = epsilon

        #### initialization
        self.method_name = 'DTH'
        self.base_learner_is_fitted = False
        self.memory = []
        self.sample_id_counter = 0
        self.current_time = 0
        self.first_time = True

    def add_sample(self,X, y, t):
        if len([y]) == 1:
            if self.current_time < t:
                    self.current_time = t
            self.memory.append(Sample(X[0], y, t, self.sample_id_counter))
            self.sample_id_counter += 1
        else:
            for i in range(len(y)):
                self.add_sample(X[i], y[i], t[i])

    def predict_online_model(self, X):
        return self.base_learner.predict(X)

    def update_online_model(self):
        self.fit_base_learner()
        self.prune_memory()

    def fit_base_learner(self):
            X, y = self.samples2xy(self.memory)
            self.base_learner.model.fit(X, y)
            self.base_learner_is_fitted = True


    def prune_memory(self):
        X_with_t_o, y = self.samples2xy(self.memory, at_current_time=False)
        X_with_t_c, y = self.samples2xy(self.memory, at_current_time=True)

        mu_o, sigma_o = self.predict_bulk(X_with_t_o)
        mu_c, sigma_c = self.predict_bulk(X_with_t_c)

        sigma_o = np.maximum(sigma_o, 0.00001)
        sigma_c = np.maximum(sigma_c, 0.00001)

        prob_y_current  = np.exp(-0.5 * ((y - mu_c) / sigma_c)**2) / (sigma_c)
        prob_y_original = np.exp(-0.5 * ((y - mu_o) / sigma_o)**2) / (sigma_o)

        # prior = np.array([sample.expiration_probability for sample in self.memory])
        # prior = 0.5
        prob_original_given_y = ((prob_y_original * prior) /
                                    (prob_y_original * prior + prob_y_current * (1 - prior)))
        

        for i, sample in enumerate(self.memory):
            sample.expiration_probability = prob_original_given_y[i]
            if prob_original_given_y[i] > self.epsilon:
                self.memory.pop(i) 


    def samples2xy(self, samples, at_current_time=False):
        # print('sample2xy:', samples[0].X)
        if at_current_time:
            # X = np.append(self.current_time, [sample.X for sample in samples])
            X = np.array([sample.X_with_current_time(self.current_time) for sample in samples])
            # print('at current', X)
        else:
            X = np.array([sample.X_with_time() for sample in samples])
            # print('at original',X)
        y = np.array([sample.y for sample in samples])  
        # print('X', X)
        # print('y.shape:', y.shape)
        return X, y
    
    def predict_bulk(self, X_batch_with_time):
        if self.base_learner.model.__class__.__name__ == 'RandomForestRegressor':
            return self.base_learner.get_sub_predictions(X_batch_with_time)
    

    def get_prediction_at(self, X, t):
        return self.base_learner.get_sub_predictions([np.append(X, t)])


    def predict_online_model(self, X_with_time):
        if self.base_learner_is_fitted:
            # X_with_time = np.append(t, X)
            # print('X_with_time:', X_with_time)
            return self.base_learner.model.predict(X_with_time)
        else:
            return [0]
        # if self.use_sublearners_as_baselearner:
        #     if self.base_learner_is_fitted:
        #         return self.get_prediction_at(X, self.current_time)[0]
        #     else:
        #         if len(self.memory) > self.num_sub_learners:
        #             # X = np.array([np.append(sample[0],sample[2]) for sample in self.memory])
        #             # y = np.array([sample[1] for sample in self.memory])
        #             X = np.array([sample.X_with_time() for sample in self.memory])
        #             y = np.array([sample.y for sample in self.memory])
        #             self.base_learner.model.fit(X, y)
        #             self.base_learner_is_fitted = True
        #             return self.predict_online_model(X, t)
        #         elif len(self.memory) > 0:
        #             # return mean of the y values in the memory
        #             return np.mean([sample.y for sample in self.memory])
        #         else:
        #             return [0]
        # else: # use a separate base_learner
        #     if self.base_learner_is_fitted:
        #         X_time_included = np.append(X, t)
        #         return self.base_learner.predict(X_time_included)
        #     elif len(self.memory) > 0:
        #         # X_time_included = np.append(X, t)
        #         # X, y = self.samples2xy(self.memory)
        #         X_time_included = np.array([np.append(sample[0],sample[2]) for sample in self.memory])
        #         y = np.array([sample[1] for sample in self.memory])
        #         self.base_learner.model.fit(X_time_included, y)
        #         self.base_learner_is_fitted = True
        #         return self.predict_online_model(X, t)
        #     else: # raise error 'no model is fitted'
        #         return [0]
    




