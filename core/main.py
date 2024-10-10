import numpy as np
import joblib

from sklearn.datasets import make_moons, make_blobs, load_iris
from sklearn.model_selection import train_test_split
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
    
    def sigmoidal_deriv(self, x:np.ndarray):
        return expit(x) * (1-expit(x))

    def forward(self, x:np.ndarray):
        # Estabilishing parallel working
        with joblib.Parallel(n_jobs=self.n_input_models) as parallel:
            delayed_funcs = [joblib.delayed(lambda f: f.forward(x))(model) for model in self.models]
            results = parallel(delayed_funcs)

        self.output_inner = np.sum(results, axis=0)
        self.activ = self.output_inner.dot(self.inner_W) + self.inner_B
        self.output_model.x_ = expit(self.activ)
        pred = self.output_model.forward()

        return pred

    def backpropagation(self):
        # First iteration
        softmax = self.forward(self.x_)

        delta_n = np.copy(softmax)
        delta_n[range(self.x_.shape[0]-1), self.y_] -= 1

        dWn = self.learning_rate*(self.output_model.activation.T).dot(delta_n)
        dBn = self.learning_rate*np.sum(delta_n, axis=0, keepdims=True)

        W_ant = np.copy(self.output_model.W_output)
        self.output_model.W_output -= dWn
        self.output_model.B_output -= dBn

        # Second iteration

        delta_prog = delta_n.dot(W_ant.T)*self.sigmoidal_deriv(self.output_model.x1)
        dWn = self.learning_rate*(self.output_model.x_.T).dot(delta_prog)
        dBn = self.learning_rate*np.sum(delta_prog, axis=0, keepdims=True)

        W_ant = np.copy(self.output_model.W_input)
        self.output_model.W_input -= dWn
        self.output_model.B_input -= dBn

        # Third iteration

        delta_n = delta_prog
        delta_prog = delta_n.dot(W_ant.T)*self.sigmoidal_deriv(self.output_model.x_)

        dWn = self.learning_rate*(self.activ.T).dot(delta_prog)
        dBn = self.learning_rate*np.sum(delta_prog, axis=0, keepdims=True)

        W_ant = np.copy(self.inner_W)
        self.inner_W -= dWn
        self.inner_B -= dBn

        # Last iteration
        with joblib.Parallel(n_jobs=self.n_input_models, prefer="threads") as parallel:
            delayed_funcs = [joblib.delayed(lambda x: x.backpropagation(delta_prog.tolist(), W_ant.tolist(), self.output_inner.tolist()))(model) for model in self.models]
            parallel(delayed_funcs)

        return softmax

    def fit(self, epochs:int=100):
        correct = 0

        for epoch in range(epochs):
            outputs = self.backpropagation()
            prediction = np.argmax(outputs, axis=0)
            correct = (prediction == self.y_).sum()
            accuracy = correct/self.y_.shape[0]

            if (epoch-1)%10 == 0:
                print(f"Epoch: {epoch}/{epochs}, Accuracy: {accuracy}")

        return prediction
        


if __name__ == "__main__":
    X, y = make_moons(n_samples=400, random_state=3, noise=1.5)
    
    brain = ModularNN(x_=X, y_=y, n_input_models=5, learning_rate=0.001, std_n_neurons_hidden=10, std_output_inner=4, output_shape=2)
    result = brain.fit()
