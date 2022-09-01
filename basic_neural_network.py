import layer
import loss

import numpy as np

class BasicNeuralNetwork:

    def __init__(self, layers, loss = loss.MSE()):
        self.layers = layers
        self.loss = loss

    def predict(self, X):
        Y_predicted = X
        for layer in self.layers:
            Y_predicted = layer.forward_prop(Y_predicted)
        return Y_predicted

    def train(self, X_train, Y_train, epochs = 1000, learning_rate = 0.01, print_status = True):
        for epoch in range(epochs):
            error = 0
            for X, Y in zip(X_train, Y_train):

                Y_predicted = self.predict(X)

                error += self.loss.L(Y_predicted, Y)

                dE_dY = self.loss.L_prime(Y_predicted, Y)
                for layer in reversed(self.layers):
                    dE_dY = layer.backward_prop(dE_dY, learning_rate)

            error /= len(X_train)
            if (print_status):
                print(f"epoch: {epoch + 1}/{epochs}, error = {error}")