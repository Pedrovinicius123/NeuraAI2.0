import numpy as np
from scipy.special import expit, softmax

class InputModel:
    def __init__(self, n_neurons_hidden:int, n_neurons_output:int, learning_rate:float, x_:np.ndarray):
        # Setting x_ variable
        self.x_ = x_

        # Other important variables
        self.n_neurons_hidden = n_neurons_hidden
        self.learning_rate = learning_rate

        # Initializing weights and biases
        self.W_input = np.random.randn(self.x_.shape[1], self.n_neurons_hidden)
        self.B_input = np.random.randn(1, self.x_.shape[1])
        
        self.W_output = np.random.randn(self.n_neurons_hidden, self.n_neurons_output)
        self.B_output = np.random.randn(1, self.n_neurons_hidden)

    def sigmoidal_deriv(self, x):
        return expit(x) * (1 - expit(x))

    def forward(self):
        # Forward feeding processing

        self.x1 = self.x_.dot(self.W_input) + self.B_input
        self.activation = expit(self.x1)
        output = self.activation.dot(self.W_output) + self.B_output
        return softmax(output, axis=0)


