import numpy as np
import time
from scipy.special import expit

class InputModel:
    def __init__(self, n_neurons_hidden:int, n_neurons_output:int, learning_rate:float, x_:np.ndarray):
        # Setting x_ variable
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
        self.x1 = x.dot(self.W_input) + self.B_input
        self.activation = expit(self.x1)
        self.output = self.activation.dot(self.W_output) + self.B_output

        return expit(self.output)

    def sigmoidal_deriv(self, x):
        return expit(x) * (1-expit(x))
    
    def backpropagation(self, delta:list, W_ant:list, output_inner:list):
        delta = np.array(delta)
        W_ant = np.array(W_ant)
        output_inner = np.array(output_inner)

        self.forward(self.x_)

        delta_prog = delta.dot(W_ant.T)*self.sigmoidal_deriv(output_inner)
        
        dWn = self.learning_rate*(self.activation.T).dot(delta_prog)
        dBn = self.learning_rate*np.sum(delta_prog, axis=0, keepdims=True)

        W_ant = np.copy(self.W_output)
        self.W_output -= dWn
        self.B_output -= dBn

        delta_n = delta_prog.dot(self.W_output.T)*self.sigmoidal_deriv(self.activation)
        dWn = self.learning_rate*(self.x_.T).dot(delta_n)
        dBn = self.learning_rate*np.sum(delta_n, axis=0, keepdims=True)

        W_anterior = np.copy(self.W_input)
        self.W_input -= dWn
        self.B_input -= dBn
