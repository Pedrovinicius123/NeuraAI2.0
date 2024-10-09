import numpy as np
import joblib

from sklearn.datasets import make_moons, make_blobs, load_iris
from scipy.special import expit
from InputNN.src import InputModel
from OutputNN.src import OutputModel


class ModularNN:
    def __init__(self, x_:np.ndarray, y_:np.ndarray, n_input_models:int, learning_rate:float, std_n_neurons_hidden:int, std_output_inner:int, output_shape:int):
        """
        __params__:
            x_: np.ndarray =  x value
            y_: np.ndarray =  y value
            n_input_models: int = number of models for input
            learning_rate: float = learning_rate of the NN model
            std_n_neurons_hidden: int = the standard numbe of hidden neurons from the input NN
            std_output_inner: int = the standard input NN output
            output_shape: int = the output shape

        """

        # Initializing important variables        
        self.n_input_models = n_input_models
        self.learning_rate = learning_rate
        self.std_n_neurons_hidden = std_n_neurons_hidden
        self.std_output_inner = std_output_inner

        # Initialize x y variables
        self.x_ = x_
        self.y_ = y_.reshape(-1, 1)
        self.output_shape = output_shape

        self.inner_W = np.random.randn(std_output_inner, std_output_inner)
        self.inner_B = np.ones((1, std_output_inner))

        #Initialize models
        self.models = []
        for model in range(self.n_input_models):
            self.models.append(InputModel(self.std_n_neurons_hidden, self.std_output_inner, learning_rate=learning_rate, x_=self.x_))
        
        self.output_model = OutputModel(self.std_n_neurons_hidden, self.output_shape, learning_rate=learning_rate, x_=np.random.randn(self.x_.shape[0], self.std_output_inner), y_=self.y_)

    def forward(self, x:np.ndarray):
        # Estabilishing parallel working
        with joblib.Parallel(n_jobs=self.n_input_models) as parallel:
            delayed_funcs = [joblib.delayed(lambda f: f.forward(x))(model) for model in self.models]
            results = parallel(delayed_funcs)

        self.output_inner = np.sum(results, axis=0)
        self.output_model.x_ = self.output_inner.dot(self.inner_W) + self.inner_B
        pred = self.output_model.forward(self.output_model.x_)

        return pred

    def backpropagation(self, softmax:np.ndarray):
        # First iteration
        self.forward()

        delta_n = np.copy(softmax)
        delta_n[self]
        Wn = np.copy(self.output_model.W_output)
        self.output_model.W_output -= lr*
        self.output_model.B_output -= 


if __name__ == "__main__":
    X, y = make_moons(n_samples=400, random_state=3, noise=1.5)
    brain = ModularNN(x_=X, y_=y, n_input_models=5, learning_rate=0.001, std_n_neurons_hidden=10, std_output_inner=4, output_shape=2)
    result = brain.forward(X)
    
    print(result)
