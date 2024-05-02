import numpy as np
from utility.sample import Memory

class DDM(Memory):
    def __init__(self, alpha_w=2, alpha_d=3, min_memory_len=10):
        super().__init__()
        Memory.__init__(self)
        # self.base_learner = None
        # self.base_learner_is_fitted = False
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        # self.memory = []
        self.error_history = []
        self.warning_flag_history = []
        self.change_flag_history = []
        self.t = 0 # number of timesteps since the last detected change
        self.warning_t = None # specifies the timestep at which warning signal has been alarmed
        self.n_min = 30 # minimum number of timesteps before change detection
        self.p = None # learning model's average error
        self.s = None # standard deviation of the learning model's error    
        self.p_min = None
        self.s_min = None       
        self.change_flag = False
        self.warning_flag = False
        self.method_name = 'DDM'
        self.min_memory_len = min_memory_len

        self.hyperparams = {'alpha_w':alpha_w,
                            'alpha_d': self.alpha_d,
                            'min_memory_len': min_memory_len,
                    }
        
    # def add_sample(self, X, y):
    #     self.memory.append((X, y))
        
    # detect potential changes given a new prediction error
    def detect(self, error):
        
        if self.change_flag is True:
            self.reset()
        
        self.error_history.append(error)
        self.p = np.mean(self.error_history)
        self.s = np.std(self.error_history)
        
        if self.t > self.n_min:
            if self.p_min is None:
                self.p_min = self.p
                self.s_min = self.s

            if self.p + self.s < self.p_min + self.s_min:
                self.p_min = self.p
                self.s_min = self.s
            
        
            ## WARNING ZONE
            if self.warning_flag is False:
                if self.p + self.s > self.p_min + self.alpha_w * self.s_min:
                    
                    self.warning_t = self.t
                    self.warning_flag = True
                else:
                    self.warning_flag = False
#                     self.warning_t = None
                
            ### DETECTION ZONE
            if self.p + self.s > self.p_min + self.alpha_d * self.s_min:
                self.change_flag = True
            
        self.update_memory()

        self.warning_flag_history.append(self.warning_flag)
        self.change_flag_history.append(self.change_flag)
        self.t += 1
        return self.warning_flag, self.change_flag
    
    def update_memory(self):
        if self.change_flag:
            if (self.warning_t is None) or ((self.t - self.warning_t) < self.min_memory_len): ### This means change is detected withough any prior warning
                # self.samples = self.samples[-self.min_memory_len:]
                self.forget_before(self.min_memory_len)
            else:
                # self.samples = self.samples[self.warning_t:]
                if self.change_flag:
                    self.forget_before(self.t-self.warning_t)
    
    
    def reset(self):
        self.error_history = []
        self.change_flag = False
        self.warning_flag = False
        self.p_min = None
        self.s_min = None  
        self.t = 0

    def reset_detector(self):
        self.__init__(self.alpha_w, self.alpha_d)

    # def get_recent_data(self):
    #     return self.memory

    # def get_val_horizon(self):
    #     return len(self.memory)
    # def update_(self, model, error):
    #     self.detect(error)
    #     model.reset()
    #     model.fit(self.memory)
    #     return model, len(self.memory)

    def update_online_model(self, X, y):
        self.add_sample(X, y)
        if self.base_learner_is_fitted:
            y_hat = self.predict_online_model(X)
        else:
            y_hat = 0
        error = self.mean_absoulte_error(y, y_hat)
        self.detect(error)
        # self.
        # self.base_learner.reset()
        # self.base_learner.fit(self.memory)
        self.fit_to_memory()
        if len(self.memory) > 1:
            self.base_learner_is_fitted = True
        return None
    
    def predict_online_model(self, X):
        return self.base_learner.predict(X)
    
    
    def mean_absoulte_error(self, y_true, y_pred):
        return np.mean(np.absolute(y_true - y_pred))
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from mean_regressor import MovMean
    
    ddm = DDM(alpha_w=2, alpha_d=3)
    model = MovMean()

    # create a toy step signal with Gaussian white noise
    noise_variation = 0.5
    Y = np.random.normal(0,noise_variation,1000)
    Y[500:] = np.random.normal(4,noise_variation,500)

    change_points = []
    for i, y in enumerate(Y):
        y_pred = model.predict()
#         error = model.get_error(y, y_pred)
        error = mean_squared_error([y], [y_pred], squared=False)
        ddm.add_sample(y)
        warning_flag, change_flag = ddm.detect(error)
        train_data = ddm.get_recent_data()
        model.fit(train_data)
        if warning_flag:
            print('warning at ' + str(i))
        if change_flag:
            print('change detected at ' + str(i))
            change_points.append(i)
            
    plt.close()
    plt.plot(Y, label='data')
    plt.plot(model.y_pred_history, label='prediction')
    for point in change_points:
        plt.axvline(x=point , color='red',alpha=0.4)
    plt.axvline(x=500 , color='red',linestyle='dashed',alpha=0.5)
    plt.xlabel('timestep')
    plt.ylabel('y')
    plt.legend()
    plt.show()