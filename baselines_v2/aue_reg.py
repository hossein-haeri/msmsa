import numpy as np
import copy
from sklearn.model_selection import cross_val_score
# from skmultiflow.drift_detection import KSWIN as KolmogorovSmirnovWIN

# from river.drift import KSWIN as KolmogorovSmirnovWIN

class AUE:
    def __init__(self, min_memory_len=10, batch_size=100, mse_r=0.1):
        self.base_learner_is_fitted = False
        self.base_learner = None
        self.memory = []
        self.models_pool = []
        self.batch_size = batch_size
        self.epsilon = 0.0001
        self.k = 10
        self.max_num_models_in_pool = 20
        self.mse_r = mse_r # mse of a random predictor
        # self.change_flag = False
        # self.change_flag_history = []
        self.method_name = 'AUE'
        self.min_memory_len = min_memory_len
        self.hyperparams = {
                    }

    def add_sample(self, X, y):
        self.memory.append((X, y))

    # def detect(self, error):
    #     # self.add_element(error)
    #     self.update(error)
    #     self.change_flag = self.drift_detected
    #     self.update_memory()
    #     return False, self.change_flag

    # def update_memory(self):
    #     if self.change_flag:
    #         self.memory = self.memory[-self.min_memory_len:]

    def get_recent_data(self):
        return self.memory
    
    def reset_detector(self):
        self.reset()
        self.memory = []
        self.change_flag = False
        self.change_flag_history = []
        

    def cross_validation(self, model, memory):
        return -cross_val_score(model.model, [sample[0] for sample in memory], [sample[1] for sample in memory], cv=5, scoring='neg_mean_squared_error').mean()

    def sort_models(self, models_pool):
        # sort the models in the pool based on their weights
        return models_pool.sort(key=lambda x: x.weight, reverse=True) 

    def update_online_model(self, X, y):

        self.add_sample(X, y)

        if len(self.memory) > self.batch_size:
            
            model_ = copy.copy(self.base_learner)
            model_.reset()
            model_.fit(self.memory)
            if len(self.memory) > 1:
                self.base_learner_is_fitted = True

            # Calculate the MSE error via cross validation
            mse_ = self.cross_validation(model_, self.memory)
            model_.weight = 1 / (mse_ + self.epsilon)

            for mdl in self.models_pool:
                mse = self.cross_validation(mdl, self.memory)
                mdl.weight = 1 / (mse + self.epsilon)
                # print(mdl.weight)
            # print('\n')
            self.sort_models(self.models_pool)
            top_k_models = self.models_pool[:self.k]
            top_k_models.append(model_)
            self.models_pool.append(model_)

            for mdl in top_k_models[0:-1]: # exclude the last model
                if mdl.weight > 1 / self.mse_r:
                    mdl.fit(self.memory)
            
            self.memory = []
            self.sort_models(self.models_pool)
            self.models_pool = self.models_pool[0:self.max_num_models_in_pool]
        return None
    
    # def update_online_model(self, X, y):
    #     self.add_sample((X, y))
    #     if self.base_learner_is_fitted:
    #         y_hat = self.predict_online_model(X)
    #     else:
    #         y_hat = 0
    #     error = self.mean_absoulte_error(y, y_hat)
    #     self.detect_(error)
    #     self.base_learner.reset()
    #     self.base_learner.fit(self.memory)
    #     if len(self.memory) > 1:
    #         self.base_learner_is_fitted = True
    #     return None
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))
    
    def predict_online_model(self, X):
        # from self.models_pool predict the output of the ensemble
        y_pred = 0
        for model in self.models_pool:
            y_pred += model.predict(X)
        if len(self.models_pool) == 0:
            return None
        else:
            return y_pred / len(self.models_pool)
    