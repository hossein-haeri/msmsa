import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

class MovMean:
    def __init__(self):
        self.y_pred_history = []
        self.error_history = []
        self.mean = 0
        
        

    def fit(self, train_data):
        # X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.mean = np.mean(y)
        else:
            self.mean = y[0]
            

    def predict(self, X=None):
        pred = self.mean
        self.y_pred_history.append(pred)
        return pred
    
    def get_parameters(self):
        return [self.mean]

    def reset(self):
        self.y_pred_history = []
        self.error_history = []
        self.mean = 0

class KNN:
    def __init__(self, n_neighbors=10):
        self.y_pred_history = []
        self.error_history = []
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)



    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > self.n_neighbors:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        pred = self.model.predict([X])
        self.y_pred_history.append(pred)
        return pred

    def get_sub_predictions(self, X):
        # tree_predictions =  [tree.predict(X) for tree in self.model.estimators_]
        neighbors_distances, neighbors_indices = self.model.kneighbors(X)
        means = []
        std_devs = []
        for indices in neighbors_indices:
            neighbor_targets = y[indices]
            means.append(np.mean(neighbor_targets))
            std_devs.append(np.std(neighbor_targets))
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
        
    

    def reset(self):
        self.__init__()


class DecissionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.y_pred_history = []
        self.error_history = []


    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        # if X is a single feature vector
        if len(X.shape) == 1:
            pred = self.model.predict([X])
        # if X is a batch of feature vectors
        else:
            pred = self.model.predict(X)
        # self.y_pred_history.append(pred)
        return pred

    def get_parameters(self):
        return self.model.coef_+self.model.intercept_
    
    def reset(self):
        self.__init__(max_depth=self.max_depth)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=5, n_jobs=-1, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs, bootstrap=bootstrap)
        self.y_pred_history = []
        self.error_history = []

    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        # if X is a single feature vector
        if len(X.shape) == 1:
            pred = self.model.predict([X])
        # if X is a batch of feature vectors
        else:
            pred = self.model.predict(X)
        return pred
    
    def get_sub_predictions(self, X):
        tree_predictions =  [tree.predict(X) for tree in self.model.estimators_]
        return np.mean(tree_predictions, axis=0), np.std(tree_predictions, axis=0)

    def reset(self):
        self.__init__(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs)


class Linear:
    def __init__(self, alpha=10):
        self.alpha = alpha
        # self.model = LinearRegression(fit_intercept = True)
        self.model = Ridge(alpha=self.alpha, fit_intercept = True)
        self.y_pred_history = []
        self.error_history = []


    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.model.fit(X, y)

    def predict(self, X):
        # if X is a single feature vector
        if len(X.shape) == 1:
            pred = self.model.predict([X])
        # if X is a batch of feature vectors
        else:
            pred = self.model.predict(X)
        # self.y_pred_history.append(pred)
        return pred

    def get_parameters(self):
        return self.model.coef_+self.model.intercept_
    
    def reset(self):
        self.__init__()



class SVReg:
    def __init__(self):
        # self.model = SVR(C=1.0, epsilon=0.2)
        self.model = SVR(kernel='rbf', C=10, gamma=0.3, epsilon=.1)
        self.y_pred_history = []
        self.error_history = []


    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        pred = self.model.predict([X])
        self.y_pred_history.append(pred)
        return pred

    def get_parameters(self):
        return self.model.coef_+self.model.intercept_
    
    def reset(self):
        self.__init__()



class Polynomial:
    def __init__(self, degree=3, fit_intercept=True):
        self.degree = degree
        self.fit_intercept=fit_intercept
        self.model = Pipeline([         
                                        # ('scaler', StandardScaler()),
                                        ('poly',PolynomialFeatures(degree=self.degree)),
                                        # ('linear_reg', LinearRegression(fit_intercept = fit_intercept))
                                        ('ridge', Ridge(alpha=10, fit_intercept = True))
                                        ])
        self.y_pred_history = []
        self.error_history = []


    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]

        if len(train_data) > 1:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        X = X.reshape(1, -1)
        pred = self.model.predict(X)
        self.y_pred_history.append(pred)
        return pred

    def get_parameters(self):
        # return self.model.coef_
        # print(self.model.named_steps)
        # print(self.model.named_steps['linearregression'].get_parameters())
        return self.model.named_steps['linearregression'].coef_+self.model.named_steps['linearregression'].intercept_

    def reset(self):
        self.degree = self.degree
        self.model = Pipeline([         
                                        # ('scaler', StandardScaler()),
                                        ('poly',PolynomialFeatures(degree=self.degree)),
                                        # ('linear_reg', LinearRegression(fit_intercept = fit_intercept))
                                        ('ridge', Ridge(alpha=10, fit_intercept = True))
                                        ])
        self.y_pred_history = []
        self.error_history = []




class NeuralNet:
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=50, activation='relu', solver='adam', alpha=0.005, max_iter=20):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.hidden_layer_sizes = (n_neurons_per_layer,) * n_hidden_layers  # Creates a tuple with a specified number of neurons per layer, repeated for the number of layers
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, 
                                  activation=self.activation, 
                                  solver=self.solver, 
                                  alpha=self.alpha, 
                                  max_iter=self.max_iter)
        self.y_pred_history = []
        self.error_history = []
        # suppress warnings about convergence
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)


    def fit(self, train_data):
        X = [sample[0] for sample in train_data]
        y = [sample[1] for sample in train_data]
        if len(train_data) > 1:
            self.model.fit(X, y)
        else:
            pass

    def predict(self, X):
        pred = self.model.predict([X])
        # self.y_pred_history.append(pred)
        return pred

    def get_parameters(self):
        parameters = {
            'coefs_': self.model.coefs_,
            'intercepts_': self.model.intercepts_
        }
        return parameters
    
    def reset(self):
        self.__init__(n_hidden_layers=self.n_hidden_layers, n_neurons_per_layer=self.n_neurons_per_layer,
                      activation=self.activation, solver=self.solver, alpha=self.alpha, max_iter=self.max_iter)

# if __name__ == '__main__':
#     model = Linear()
#     a = [1,2]
#     X_train = np.random.uniform(-1,1,[100,2])
#     y_train = [np.dot(a,x) + np.random.normal(0,0.05) for x in X_train]
#     model.fit(X_train,y_train)
#     param = model.get_parameters()
#     # print(param)
