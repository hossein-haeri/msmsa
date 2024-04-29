import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def plot_loggers(self, loggers):
        logs = self.logs2df(loggers)


        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        # plot MAE for every method and every dataset on sns barplot
        axs[0] = sns.barplot(x="dataset_name", y="mae", data=logs, hue="method_name", errorbar="sd",ax=axs[0])
        axs[0].set_title('Mean Absolute Error')
        # plot validity horizon for every method and every dataset on sns barplot
        axs[1] = sns.barplot(x="dataset_name", y="average_training_samples", data=logs, hue="method_name", errorbar="sd", ax=axs[1])
        axs[1].set_title('Average # of Training Samples')




        # # plot mae vs X
        # fig2, axs2 = plt.subplots(1, 1, figsize=(10, 10))
        # for i, logger in enumerate(loggers):
        #     y = np.array(logger.y).squeeze()
        #     y_pred = np.array(logger.y_pred).squeeze()
        #     X = np.array(logger.X).squeeze()
        #     e = np.abs(y - y_pred)
        #     # print(, logger.y)
        #         # sns.lineplot(x=X, y=e, label=logger.method_name, ax=axs2)
        #         # plt.xlabel('X')
        #         # plt.ylabel('MAE')

        
        #     fig4, axs4 = plt.subplots(1, 1, figsize=(10, 10))
        #     t = np.arange(0, len(y), 1)

        #     contour = axs4.tricontourf(X, t, e, 20, cmap='jet')
        #     axs4.set_xlabel('X')
        #     axs4.set_ylabel('Timestep')
        #     # set colorbar
        #     plt.colorbar(contour, ax=axs4)
        #     
            


        # for logger in loggers:
        #     if logger.method_name == 'MSMSA+':
        #         anchors =np.array(logger.anchors).squeeze()
        #         val_hor = np.array(logger.val_hor).squeeze()

        #         fig3, axs3 = plt.subplots(1, 1, figsize=(10, 10))
        #         for i, logger in enumerate(loggers):
        #             for i in range(len(logger.y)):
        #                 # plot the validity horizon for every anochor point 
        #                 plt.cla()
        #                 print(anchors[0:10], val_hor[i,0:10])
        #                 axs3.plot(anchors, val_hor[i,:], label=logger.method_name)
        #                 # axs3.hlines(i, min(anchors), max(anchors), color='black', alpha=0.5)
        #                 plt.pause(0.5)
 

        plt.show()
    



class Logger:
    def __init__(self):
        # self.summary = pd.DataFrame(columns=[   'dataset_name',
        #                                         'method_name', 
        #                                         'base_learner_name', 
        #                                         'mae', 
        #                                         'average_training_samples', 
        #                                         ])
        self.summary = {}
        self.X = []
        self.y = []
        self.y_pred = []
        self.num_train_samples_list = []
        self.errors = []
        self.memory_history = [] 
        self.method_name = ''
        self.val_hor = []


    def rescale(y, scaler):
        return scaler.inverse_transform(np.asarray(y).reshape(-1, 1))
    
    def log(self, y, y_pred, num_train_samples=None):
        self.y.append(y)
        self.y_pred.append(y_pred)
        self.num_train_samples_list.append(num_train_samples)
        self.errors.append(np.abs(y - y_pred))
        
    
    def finish(self):
        self.summary['mean_memory_size'] = np.mean(self.num_train_samples_list)
        self.summary['MAE'] = np.mean(self.errors)
        # add RMSE to the summary dictionary
        self.summary['RMSE'] = np.sqrt(np.mean(np.square(self.errors)))
        # add MAPE to the summary dictionary
        self.summary['MAPE'] = np.mean(np.abs(np.array(self.errors) / np.array(self.y)))
        # add R-squared to the summary dictionary
        self.summary['R2'] = 1 - np.sum(np.square(self.errors)) / np.sum(np.square(np.array(self.y) - np.mean(self.y)))


        # # assuming synthetic_param is a dictionary, concat it to the summary dictionary
        # if 'Hyper' in self.summary['dataset_name']:
        #     self.summary = {**self.summary, **self.synthetic_param}
        

        

        