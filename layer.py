import numpy as np

class Layer:
    def __init__(self):
        self.X = None
        self.Y = None

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass

# add exception for incorrect size of input X or dE_dY
class Dense(Layer):

    def __init__(self, num_input_nodes, num_output_nodes):
        self.W = np.random.randn(num_output_nodes, num_input_nodes)
        self.b = np.random.randn(num_output_nodes, 1)

    def forward_prop(self, X):
        self.X = X
        return np.dot(self.W, X) + self.b

    def backward_prop(self, dE_dY, learning_rate):
        old_W = self.W.copy()
        self.W -= learning_rate * np.dot(dE_dY, self.X.T)
        self.b -= learning_rate * dE_dY
        return np.dot(old_W.T, dE_dY)

class Activation(Layer):

        def f(self, X):
            return X

        def f_prime(self, X):
            return 1

        def forward_prop(self, X):
            self.X = X
            return self.f(X)

        def backward_prop(self, dE_dY, learning_rate):
            return np.multiply(dE_dY, self.f_prime(self.X))

class ReLU(Activation):

    def __init__(self, leaky_coefficient = 0):
        self.a = leaky_coefficient

    def f(self, X):
        return X * (X > 0) - self.a * X * (X < 0)

    def f_prime(self, X):
        return X > 0 - a * (X < 0)

class Tanh(Activation):

    def f(self, X):
        return np.tanh(X)

    def f_prime(self, X):
        return 1 - self.f(X) ** 2

class Sigmoid(Activation):

    def f(self, X):
        return 1 / (1 + np.exp(-X))

    def f_prime(self, X):
        return self.f(X) * (1 - self.f(X))

class Softmax(Activation):

    def forward_prop(self, X):
        self.Y = np.exp(X) / np.sum(np.exp(X))
        return self.Y

    def backward_prop(self, dE_dY, learning_rate):
        return np.dot((np.identity(np.size(self.Y)) - self.Y.T) * self.Y, dE_dY)