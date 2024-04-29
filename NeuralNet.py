import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class NeuralNetWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(NeuralNetWithDropout, self).__init__()
        self.layers = nn.ModuleList()
        # Adding the first layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Adding hidden layers with dropout
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x, apply_dropout=False):
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x) if apply_dropout else F.dropout(x, p=0, training=False)
            else:
                x = F.relu(layer(x))
        # No activation on the final layer for regression
        return x
    
    def predict_with_uncertainty(self, x, n_samples=10):
        self.train()  # Keep the model in training mode to use dropout during prediction
        predictions = [self.forward(x, apply_dropout=True) for _ in range(n_samples)]
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std_dev = predictions.std(dim=0)
        return mean, std_dev
    
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode to disable dropout
        return self.forward(x, apply_dropout=False)
    
    def fit(self, X, y, epochs, learning_rate):
        losses =[]
        self.train()  # Set the model to training mode
        # Loss function for regression
        criterion = nn.MSELoss()
        # Optimizer (using Adam here)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients
            outputs = self.forward(X, apply_dropout=True)  # Forward pass
            loss = criterion(outputs, y)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
            losses.append(loss.item())
        return losses
# Commenting out method calls
# x_train = torch.randn(1000, 10)  # Example training data (100 samples, 10 features each)
# y_train = torch.randint(0, 1, (1000,))  # Example target values for 3 classes

df = pd.read_csv('datasets/bike_hour.csv')
data_X = df[['workingday','mnth','holiday','weathersit','season','atemp','temp','hum','windspeed']].to_numpy()
# data_X = df[['atemp','temp','hum','windspeed']].to_numpy()
data_y = df['cnt'].to_numpy()

scaler_X = StandardScaler()
data_X = scaler_X.fit_transform(data_X)
scaler_y = StandardScaler()
data_y = scaler_y.fit_transform(data_y.reshape(-1, 1))+0.1

# convert to torch tensors
x_train = torch.tensor(data_X, dtype=torch.float32)
y_train = torch.tensor(data_y, dtype=torch.float32)

# print the shape of the tensors
print(x_train.shape, y_train.shape)

# get number of features
num_features = x_train.shape[1]

model = NeuralNetWithDropout(input_size=num_features, hidden_sizes=[50, 50], output_size=1)
losses = model.fit(x_train, y_train, epochs=100, learning_rate=0.02)

# make prediction on the training data with uncertainty
# make random x_test data
x_test = torch.randn(5, num_features)
y_pred, y_std = model.predict_with_uncertainty(x_test, n_samples=10)

print('y_pred: ',y_pred)
print('y_std: ',y_std)


# plot the loss
# import matplotlib.pyplot as plt
# plt.plot(range(len(losses)),losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()