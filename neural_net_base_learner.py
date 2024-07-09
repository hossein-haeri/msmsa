import torch
import torch.nn as nn
import torch.optim as optim

class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1, dropout=0.1, learning_rate=0.01, epochs=10):
        super(RegressionNN, self).__init__()
        # Define the architecture
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Dropout layer with a default rate of 0.5
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        self.model = nn.Sequential(*layers).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mean = 0
        self.std = 1
        self.label_mean = None
        self.label_std = None
        self.epsilon = 1e-4
    
    # define a get_params method to get the model parameters
    def get_params(self):
        params = {'lr': self.learning_rate, 'epochs': self.epochs}
        return params 

    def forward(self, x):
        # use the normalization parameters to normalize the input
        # Normalize the input
        if self.mean is not None and self.std is not None:
           x = (x - self.mean) / (self.std + self.epsilon)
        return self.model(x)
    
    def fit(self, X, y):
        self.reset()
        self.train()  # Set the model to training mode
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # shuffle the data
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]

        # if there is only one sample in the dataset, then do not normalize the data
        if X.size(0) != 1:
            # Normalize the input data and store the mean and std for later use
            self.mean = X.mean(0)
            self.std = X.std(0)
            X = (X - self.mean) / (self.std + self.epsilon)
            
            # Normalize the labels and store the mean and std for later use
            self.label_mean = y.mean(0)
            # if y is a 1D tensor, then y.std(0) will return a scalar tensor
            self.label_std = y.std(0)
            y = (y - self.label_mean) / (self.label_std + self.epsilon)


        for epoch in range(self.epochs):
            y_pred = self.forward(X).squeeze(1)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def predict(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            pred = self.forward(X).detach().cpu().numpy()[0]
            if pred is not None:
                return pred
            else:
                return 0
    
    def reset(self):
        # Reinitialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    def make_uncertain_predictions(self, X, num_samples=20):
        # Function to make predictions using dropout at inference
        self.eval()
        # Temporarily turn on dropout layers during inference
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()

        self.apply(apply_dropout)
        predictions = []
        X = torch.tensor(X, dtype=torch.float32)

        for _ in range(num_samples):
            with torch.no_grad():
                predictions.append(self.forward(X))
        
        predictions = torch.stack(predictions)
        # mean = predictions.mean(0)
        std = predictions.std(0)
        # print(mean.numpy().shape, std.numpy().shape)
        model_pred = self.model(X).squeeze(1).detach().cpu().numpy()
        return model_pred, std.detach().cpu().numpy().squeeze()
        # return mean.numpy().squeeze(), std.numpy().squeeze()

# The following calls are commented out and should be uncommented only after the user's approval:
# model = RegressionNN(input_dim=10, hidden_layers=[100, 50])
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# uncertain_predictions = model.make_uncertain_prediction(X_test)

