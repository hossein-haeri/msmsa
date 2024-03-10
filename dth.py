
import numpy as np
import copy
import sys
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from utility.sample import Sample
import learning_models


class DTH:
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self, epsilon=0.5, num_sub_learners=2, min_new_samples_for_base_learner_update=1, min_new_samples_for_pruining=1, multi_threading_sub_learners=True, num_cold_start_samples=10):
        ### hyper-parameters
        self.base_learner = learning_models.DecissionTree(max_depth=5)
        self.num_sub_learners = num_sub_learners
        self.sub_model = learning_models.DecissionTree(max_depth=3)  # sub-models are used to estimate the uncertainty of the base_learner, they need to be relatively fast or be only a few
        self.epsilon = epsilon
        self.min_new_samples_for_base_learner_update = min_new_samples_for_base_learner_update
        self.min_new_samples_for_pruining = min_new_samples_for_pruining
        self.multi_threading_sub_learners = multi_threading_sub_learners
        self.num_cold_start_samples = num_cold_start_samples
        
       
        #### initialization
        self.method_name = 'DTH'
        self.sub_learners = []
        self.base_learner_is_fitted = False
        self.memory = []
        self.sample_id_counter = 0
        self.current_time = 0
        self.new_samples_count_for_base_learner_update = 0 # number of new samples added to memory since last update
        self.new_samples_count_for_pruining = 0
        for _ in range(self.num_sub_learners):
            self.sub_learners.append(copy.deepcopy(self.sub_model))



    def add_sample(self,X, y, t):
        if len(X) == 1:
            if self.current_time < t:
                    self.current_time = t
            self.memory.append(Sample(X, y, t, id=self.sample_id_counter))
            self.sample_id_counter += 1
            self.new_samples_count_for_base_learner_update += 1
            self.new_samples_count_for_pruining += 1
        else:
            for i in range(len(y)):
                self.add_sample(X[i], y[i], t[i])


    def predict_online_model(self, X):
        return self.base_learner.predict(X)


    def update_online_model(self):
        
        if self.new_samples_count_for_base_learner_update >= self.min_new_samples_for_base_learner_update:
            self.new_samples_count_for_base_learner_update = 0
            # fit the base_learner to the memory
            self.fit_base_learner()

        # if self.num_cold_start_samples < len(self.memory) and self.new_samples_count_for_pruining >= self.min_new_samples_for_pruining:
        if self.new_samples_count_for_pruining >= self.min_new_samples_for_pruining:
            
            self.new_samples_count_for_pruininge = 0
            # prune the memory
            self.prune_memory()

    def fit_base_learner(self):
        # fit the base_learner to the memory
        if len(self.memory) == 0:
            print('no samples in memory to fit the base_learner')
        else:
            print('fitting the base_learner with ', len(self.memory), ' samples')
        X, y = self.samples2xy(self.memory)
        self.base_learner.model.fit(X, y)
        self.base_learner_is_fitted = True


    def prune_memory(self):
        # update sub-learners if memory is large enough
        # if len(self.memory) > self.num_sub_learners*10:
        self.update_sub_learners()
        
        # assess the validity of the samples in memory
        for sample in self.memory:
            stability_at_sampling_time = self.get_prediction_stability_at(sample.X, sample.t)
            stability_at_current_time = self.get_prediction_stability_at(sample.X, self.current_time)    
            prediction_at_sampling_time = self.predict_online_model(sample.X, sample.t)
            prediction_at_current_time = self.predict_online_model(sample.X, self.current_time)

            alpha = np.abs(prediction_at_sampling_time*stability_at_sampling_time - prediction_at_current_time*stability_at_current_time)
            if alpha > self.epsilon:
                self.memory.remove(sample)
    

    def update_sub_learners(self):
        batches = [[] for _ in range(self.num_sub_learners)]
        random.shuffle(self.memory)
        
        for i, sample in enumerate(self.memory):
            batches[i % self.num_sub_learners].append(sample)

        if self.multi_threading_sub_learners:
            # Use ThreadPoolExecutor to manage a pool of threads
            with ThreadPoolExecutor(max_workers=self.num_sub_learners) as executor:
                futures = []
                for i, batch in enumerate(batches):
                    # Submit each training task to the thread pool
                    future = executor.submit(self.train_sub_learner, batch, self.sub_learners[i])
                    futures.append(future)
                # Optionally, you can wait for all futures to complete and check for results
                # This step is not strictly necessary if you don't need to process the results immediately
                for future in futures:
                    result = future.result()  # This will block until the future is complete
                    # Process result if necessary (e.g., check for exceptions)
        else:
            for i, batch in enumerate(batches):
                self.train_sub_learner(batch, self.sub_learners[i])

    def train_sub_learner(self, batch, sub_learner):
        X, y = self.samples2xy(batch)
        sub_learner.model.fit(X, y)


    def samples2xy(self, samples):
        print('START HERE!')
        X = np.array([[sample.X] for sample in samples])
        y = np.array([sample.y for sample in samples])
        X = np.concatenate((X, np.array([sample.t for sample in samples]).reshape(-1, 1)), axis=1)
        return X, y
    

    def get_prediction_stability_at(self, X, t):
        # add time to the features 
        X = np.append(X, t)
        # print('X:', X)
        sub_learner_predictions = []
        for sub_learner in self.sub_learners:
            sub_learner_predictions.append(sub_learner.model.predict([X]))
        return 1/(np.std(sub_learner_predictions)+0.00001)


    def predict_online_model(self, X, t):
        if self.base_learner_is_fitted:
            X_time_included = np.append(X, t)
            return self.base_learner.predict(X_time_included)
        elif len(self.memory) > 0:
            X_time_included = np.append(X, t)
            X, y = self.samples2xy(self.memory)
            self.base_learner.fit(X, y)
            self.base_learner_is_fitted = True
            return self.base_learner.predict(X_time_included)
        else: # raise error 'no model is fitted'
            return 0
    




