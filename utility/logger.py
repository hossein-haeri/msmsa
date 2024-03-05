import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        ...
    def logs2df(self, loggers):
        # this method should take a list of loggers and aggrigate them into a dataframe
        dfs = []
        for logger in loggers:
            # create a dataframe from the logger.summary dictionary
            df = pd.DataFrame(logger.summary, index=[0])
            # append the dataframe to the list of dataframes
            dfs.append(df)
        
        # concatenate the list of dataframes into a single dataframe
        self.summary_df = pd.concat(dfs)
        return self.summary_df
    



class Logger:
    def __init__(self):
        # self.summary = pd.DataFrame(columns=[   'dataset_name',
        #                                         'method_name', 
        #                                         'base_learner_name', 
        #                                         'mae', 
        #                                         'average_training_samples', 
        #                                         ])
        self.summary = {}
        self.y = []
        self.y_pred = []
        self.num_train_samples = []
        self.errors = []


    def rescale(y, scaler):
        return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))
    
    def log(self, y, y_pred, num_train_samples=None):
        self.y.append(y)
        self.y_pred.append(y_pred)
        self.num_train_samples.append(num_train_samples)
        self.errors.append(np.abs(y - y_pred))
    
    def finish(self):
        self.summary['mae'] = np.mean(self.errors)
        self.summary['average_training_samples'] = np.mean(self.num_train_samples)

        # assuming synthetic_param is a dictionary, concat it to the summary dictionary
        self.summary = {**self.summary, **self.synthetic_param}

        

        