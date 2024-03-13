
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

def worker(memory_slice, start_index, epsilon, sub_learners, current_time):
            removal_indices = []
            for i, (X, y, t) in enumerate(memory_slice):
                prediction_at_sampling_time, stability_at_sampling_time = get_prediction_at(X, t, sub_learners)
                prediction_at_current_time, stability_at_current_time = get_prediction_at(X, current_time, sub_learners)
                alpha = np.abs(prediction_at_sampling_time * stability_at_sampling_time - prediction_at_current_time * stability_at_current_time)
                if alpha > epsilon:
                    removal_indices.append(start_index + i)
            return removal_indices
        
def get_prediction_at(X, t, sub_learners):
    # add time to the features 
    X = np.append(X, t)
    # print('X:', X)
    sub_learner_predictions = []
    for sub_learner in sub_learners:
        sub_learner_predictions.append(sub_learner.model.predict([X]))
    return np.mean(sub_learner_predictions), 1/(np.std(sub_learner_predictions)+0.00001)

class DTH:
    ''''' Time needs to be the first feature in the input data. '''''
    def __init__(self, 
                 epsilon=0.5, 
                 num_sub_learners=2, 
                 min_new_samples_for_base_learner_update=1, 
                 min_new_samples_for_pruining=1, 
                 multi_threading_sub_learners=True, 
                 num_cold_start_samples=10,
                 pruning_disabled=False, 
                 num_pruning_threads=5,
                 max_elimination = 10):
        ### hyper-parameters
        self.base_learner = learning_models.DecissionTree(max_depth=5)
        self.num_sub_learners = num_sub_learners
        self.sub_model = learning_models.DecissionTree(max_depth=2)  # sub-models are used to estimate the uncertainty of the base_learner, they need to be relatively fast or be only a few
        self.epsilon = epsilon
        self.min_new_samples_for_base_learner_update = min_new_samples_for_base_learner_update
        self.min_new_samples_for_pruining = min_new_samples_for_pruining
        self.multi_threading_sub_learners = multi_threading_sub_learners
        self.num_cold_start_samples = num_cold_start_samples
        self.pruning_disabled = pruning_disabled
        self.num_pruning_threads = num_pruning_threads
        self.max_elimination = max_elimination

        #### initialization
        self.method_name = 'DTH'
        self.sub_learners = []
        self.base_learner_is_fitted = False
        self.memory = []
        self.sample_id_counter = 0
        self.current_time = 0
        self.first_time = True
        self.new_samples_count_for_base_learner_update = 0 # number of new samples added to memory since last update
        self.new_samples_count_for_pruining = 0
        for _ in range(self.num_sub_learners):
            self.sub_learners.append(copy.deepcopy(self.sub_model))


    def add_sample(self,X, y, t):
        if len(X) == 1:
            if self.current_time < t:
                    self.current_time = t
            self.memory.append((X, y, t))
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
            
            if len(self.memory) >= self.num_sub_learners:
                
            
                self.new_samples_count_for_pruininge = 0
                # prune the memory
                if not self.pruning_disabled:
                    if self.num_pruning_threads > 1:
                        if self.first_time:
                            print('pruning with concurrent...')
                        self.prune_memory_concurrent()
                        
                    else:
                        if self.first_time:
                            print('pruning without concurrent...')
                        self.prune_memory()
                    # self.prune_memory_concurrent()
                
                    self.first_time = False


    def fit_base_learner(self):
        # fit the base_learner to the memory
        if len(self.memory) == 0:
            print('no samples in memory to fit the base_learner')
        # else:
            # print('fitting the base_learner with ', len(self.memory), ' samples')
        # X, y = self.samples2xy(self.memory)
        # make X and y numpy arrays where X includes the time
        X = np.array([np.append(sample[0],sample[2]) for sample in self.memory])
        y = np.array([sample[1] for sample in self.memory])
        self.base_learner.model.fit(X, y)
        self.base_learner_is_fitted = True


    def prune_memory(self):
        # update sub-learners if memory is large enough
        
        self.update_sub_learners()
        
        # assess the validity of the samples in memory
        for i, (X, y, t) in enumerate(self.memory):
            elimination_count = 0
            # stability_at_sampling_time = self.get_prediction_stability_at(X, t)
            # stability_at_current_time = self.get_prediction_stability_at(X, self.current_time)    
            # prediction_at_sampling_time = self.predict_online_model(X, t)
            # prediction_at_current_time = self.predict_online_model(X, self.current_time)

            prediction_at_current_time, stability_at_current_time = get_prediction_at(X, self.current_time, self.sub_learners)
            prediction_at_sampling_time, stability_at_sampling_time = get_prediction_at(X, t, self.sub_learners)
            

            prediction_at_between_time, stability_at_between_time = get_prediction_at(X, (t+self.current_time)/2, self.sub_learners)

            
            # set alpha as the difference of the two predictions average and the between time prediction
            # alpha = np.abs((prediction_at_sampling_time+prediction_at_current_time)/2)

            # alpha = np.abs(prediction_at_sampling_time*stability_at_sampling_time - prediction_at_current_time*stability_at_current_time)
            # if alpha < self.epsilon:
            if stability_at_current_time > self.epsilon:
                if min(prediction_at_sampling_time, prediction_at_current_time) < prediction_at_between_time < max(prediction_at_sampling_time, prediction_at_current_time):
                    # self.memory.remove(sample)
                    self.memory.pop(i)
                    elimination_count += 1
                    if elimination_count > self.max_elimination:
                        self.update_sub_learners()
                        elimination_count = 0
    
    def prune_memory_concurrent(self):
        K = self.num_pruning_threads
        elimination_count = 0
        self.update_sub_learners()
        chunk_size = len(self.memory) // K + (len(self.memory) % K > 0)
        tasks = []
        with ThreadPoolExecutor(max_workers=K) as executor:
            for i in range(0, len(self.memory), chunk_size):
                end_index = min(i + chunk_size, len(self.memory))
                memory_slice = self.memory[i:end_index]
                tasks.append(executor.submit(worker, memory_slice, i, self.epsilon, self.sub_learners, self.current_time))
        indices_to_remove = []
        for future in as_completed(tasks):
            indices_to_remove.extend(future.result())
            # Remove samples in reverse order to maintain correct indexing
        
        for index in sorted(indices_to_remove, reverse=True):
            self.memory.pop(index)
            elimination_count += 1
            if elimination_count > self.max_elimination:
                self.update_sub_learners()
                elimination_count = 0

    def calculate_alpha(self, X, t):
        stability_at_sampling_time = self.get_prediction_stability_at(X, t)
        stability_at_current_time = self.get_prediction_stability_at(X, self.current_time)
        prediction_at_sampling_time = self.predict_online_model(X, t)
        prediction_at_current_time = self.predict_online_model(X, self.current_time)
        # alpha = np.abs(prediction_at_sampling_time * stability_at_sampling_time - prediction_at_current_time * stability_at_current_time)
        alpha = np.abs(prediction_at_sampling_time - prediction_at_current_time) + stability_at_sampling_time  + stability_at_current_time
        
        return alpha



    def update_sub_learners(self):
        batches = [[] for _ in range(self.num_sub_learners)]
        random.shuffle(self.memory)
        
        for i, sample in enumerate(self.memory):
            batches[i % self.num_sub_learners].append(sample)

        if self.multi_threading_sub_learners:
            # Use ThreadPoolExecutor to manage a pool of threads
            with ThreadPoolExecutor(max_workers=self.num_sub_learners) as executor:
            # with ProcessPoolExecutor(max_workers=self.num_sub_learners) as executor:
                futures = []
                for i, batch in enumerate(batches):
                    # Submit each training task to the thread pool
                    future = executor.submit(self.train_sub_learner, batch, self.sub_learners[i])
                    futures.append(future)

                for future in futures:
                    result = future.result()
        else:
            for i, batch in enumerate(batches):
                self.train_sub_learner(batch, self.sub_learners[i])

    def train_sub_learner(self, batch, sub_learner):
        # X, y = self.samples2xy(batch)
        # make X and y numpy arrays where X includes the time
        X = np.array([np.append(sample[0],sample[2]) for sample in batch])
        y = np.array([sample[1] for sample in batch])

        sub_learner.model.fit(X, y)


    def samples2xy(self, samples):
        X = np.array([sample.X_with_time() for sample in samples])
        y = np.array([sample.y for sample in samples])
        
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
            # X, y = self.samples2xy(self.memory)
            X = np.array([np.append(sample[0],sample[2]) for sample in self.memory])
            y = np.array([sample[1] for sample in self.memory])

            self.base_learner.model.fit(X, y)
            self.base_learner_is_fitted = True
            return self.base_learner.predict(X_time_included)
        else: # raise error 'no model is fitted'
            return 0
    




