import jax.numpy as jnp
from jax import random, grad, jit
from flax import linen as nn
from flax.training import train_state
import optax

# Define the DNN model
class DNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Define a class similar to the DecisionTree class but for a DNN model
class DNNRegressor:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = DNN()
        self.params = None
        self.rng = random.PRNGKey(0)
        self.optimizer = None
        self.y_pred_history = []
        self.error_history = []

    def create_train_state(self, rng, learning_rate, input_shape):
        params = self.model.init(rng, jnp.ones(input_shape))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    def loss_fn(self, params, inputs, targets):
        preds = self.model.apply({'params': params}, inputs)
        return jnp.mean((preds - targets) ** 2)

    def fit(self, train_data, epochs=1, batch_size=32):
        X = jnp.array([sample[0] for sample in train_data])
        y = jnp.array([sample[1] for sample in train_data]).reshape(-1, 1)

        if self.optimizer is None:
            self.rng, rng_init = random.split(self.rng)
            self.optimizer = self.create_train_state(rng_init, self.learning_rate, X.shape)


        @jit
        def train_step(optimizer, batch):
            def loss_fn(params):
                return self.loss_fn(params, batch[0], batch[1])
            grads = grad(loss_fn)(optimizer.params)
            return optimizer.apply_gradients(grads=grads)

        # for epoch in range(epochs):
        self.optimizer = train_step(self.optimizer, (X, y))

    def predict(self, X):
        X = jnp.array(X).reshape(1, -1)
        pred = self.model.apply({'params': self.optimizer.params}, X)
        self.y_pred_history.append(pred)
        return pred

    def reset(self):
        self.__init__(learning_rate=self.learning_rate)