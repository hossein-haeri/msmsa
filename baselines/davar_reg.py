
import numpy as np
import learning_models
from collections import deque
import copy
import sys

from scipy.ndimage import gaussian_filter

# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
# from sklearn.pipeline import make_pipeline

class DAVAR:
    # def __init__(self, num_param, dim_X, dim_y, lam=10):
    def __init__(self, lam=10, min_memory_len=10):
        self.t = 0
        # self.dim_X = dim_X
        # self.dim_y = dim_y
        # self.num_param = num_param
        self.num_candids = 100
        self.hor_candids = [int(1.115**j)+min_memory_len-1 for j in range(1, self.num_candids+1)]
        # print(self.hor_candids)
        # self.hor_candids = range(min_memory_len,self.num_candids)
        # self.hor_candids = [j for j in range(2,100)]
        # self.hor_candids = list(range(2,100))
        self.num_candids = len(self.hor_candids)
        self.validity_horizon = 1
        self.lam = lam
        # self.stride = 1
        self.memory_size = np.max(self.hor_candids)
        # self.para_memory = np.zeros([self.num_candids, self.lam, self.num_param])
        self.errors = []
        self.memory = []
        self.memory_pointer = 0
        self.models = [deque([]) for _ in range(len(self.hor_candids))]
        self.avars = [None]*len(self.hor_candids)
        self.avars_filtered = [None]*len(self.hor_candids)
        self.method_name = 'DAVAR'
        self.hyperparams = {
                            }

    def add_sample(self,sample):
        # print('added sample: ', sample)
        self.memory.append(sample)
        if len(self.memory) > self.memory_size+1:
            self.memory[-self.memory_size:]
        self.t += 1

 
    def update_(self, model, error):
        # self.errors.append([error])
        for i, tau in enumerate(self.hor_candids):
            # if self.t % max(2,int(tau/8)) == 0 :
            if self.t % tau == 0:
            # if tau > 100:
                # train a new model using last tau samples and append it to the right side of the que
                model.reset()
                model.fit(self.memory[-tau:])
                self.models[i].append(copy.copy(model))

                # get model parameters
                params = [mdl.get_parameters() for mdl in self.models[i]]
                # calculate allan variance (unstability) between models at this horizon

                new_avar = self.get_allan_variance(params)
                

                if (new_avar is not None) and (self.avars[i] is not None):
                    self.avars[i] = 0.0*self.avars[i] + 1.0*new_avar
                if (new_avar is not None) and (self.avars[i] is None):
                    self.avars[i] = new_avar

                # pop the oldest model from left side of the que
                if len(self.models[i]) > self.lam:
                    self.models[i].popleft()
    

        
        # self.filter_values(3)


        self.avars_filtered = self.avars
        if not all(v is None for v in self.avars_filtered): # check for warm start condition
            idx = self.avars_filtered.index(min([v for v in self.avars_filtered if v is not None])) # get the index of the minimum among non-None elements
            self.validity_horizon = min(self.t, self.hor_candids[idx])
            model.reset()
            model.fit(self.memory[-self.validity_horizon:])
            return model, self.validity_horizon

        else: # means we are in cold start period
            self.validity_horizon = self.t
            model.reset()
            model.fit(self.memory[-self.validity_horizon:])
            return model,self.validity_horizon
        
            # return model, min(self.t,self.hor_candids[p])


            p = None
            for i in range(self.num_candids):
                if len(self.models[i])>0: # if there exists a model built at this timescale
                    p = i
            if p is not None: # If we have at least one model build at some horizon cadidate (if multiple, p indicates the index of the largest one)
                model.reset()
                model.fit(self.memory[-self.validity_horizon:])
                return model, min(self.t,self.hor_candids[p])
                # return self.models[p][-1], min(self.t,self.hor_candids[p])
            else: # this means we do not have enough data to a model the model even at the shortest horizon candidates (strict cold start)
                return None, None
            


    def get_allan_variance(self, params):
        params = np.array(params)
        avar = []
        if len(params) < 2:
            return None
        for i in range(len(params)-1):
            avar.append( ((params[i,:] - params[i+1, :])**2).mean() )
        return np.mean(avar)
    
    def get_val_horizon(self):
        return len(self.validity_horizon)


    def filter_values(self, w):
        # print(self.avars)
        for i in range(len(self.avars)):
            windowed = self.avars[max(0,i-w-1) : min(len(self.avars), i+w+1)]
            windowed_no_none = [x for x in windowed if x is not None]
            if len(windowed_no_none) > 0 and self.avars[i] is not None:
                self.avars_filtered[i] = 0.1*np.mean(windowed_no_none) + 0.9*self.avars[i]
        # print(self.avars,'\n')