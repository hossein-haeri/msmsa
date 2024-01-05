import numpy as np

class Naive:
    def __init__(self):
        self.base_learner = None
        self.base_learner_is_fitted = False
        self.memory = []
        self.method_name = 'Naive'
        self.t = 0
        self.hyperparams = {
            }

    def add_sample(self, X, y):
        self.memory.append((X, y))
        
    def detect(self, error):
        self.t += 1
        
    def update_memory(self):
        pass
    
    def reset(self):
        self.t = 0

    def reset_detector(self):
        self.__init__(self)
        

    def get_recent_data(self):
        return self.memory

    # def update_(self, model, error):
    #     self.detect(error)
    #     model.reset()
    #     model.fit(self.memory)
    #     return model, self.t

    def update_online_model(self, X, y):
        self.add_sample(X, y)
        if self.base_learner_is_fitted:
            y_hat = self.predict_online_model(X)
        else:
            y_hat = 0
        error = self.mean_absoulte_error(y, y_hat)
        self.detect(error)
        self.base_learner.reset()
        self.base_learner.fit(self.memory)
        if len(self.memory) > 1:
            self.base_learner_is_fitted = True
        return None
    
    def predict_online_model(self, X):
        return self.base_learner.predict(X)
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))