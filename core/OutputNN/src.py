import numpy as np
from scipy.special import expit, softmax

class OutputModel:
    def __init__(self, n_neurons_hidden:int, n_neurons_output:int, learning_rate:float, x_:np.ndarray=None, y_:np.ndarray=None):
        # Setting x_ y_ variables
        self.y_ = y_
        self.x_ = x_

        # Other important variables
        self.n_neurons_hidden = n_neurons_hidden
        self.n_neurons_output = n_neurons_output
        self.learning_rate = learning_rate

        # Initializing weights and biases
        self.W_input = np.random.randn(self.x_.shape[1], self.n_neurons_hidden)
        self.B_input = np.ones((1, self.n_neurons_hidden))
        
        self.W_output = np.random.randn(self.n_neurons_hidden, self.n_neurons_output)
        self.B_output = np.ones((1, self.n_neurons_output))

    def forward(self, x:np.array):
        # Forward feeding processing
        self.x_ = x        
        self.x1 = x.dot(self.W_input) + self.B_input
        self.activation = expit(self.x1)
        output = self.activation.dot(self.W_output) + self.B_output

        return softmax(output, axis=0)    
