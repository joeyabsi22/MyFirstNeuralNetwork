import numpy as np

# add exception for predicted vs true different sizes
class Loss:

    def L(self, Y_predicted, Y_true):
        pass

    def L_prime(self, Y_predicted, Y_true):
        pass

class MSE(Loss):

    def L(self, Y_predicted, Y_true):
        return np.mean(np.power(Y_true - Y_predicted, 2))

    def L_prime(self, Y_predicted, Y_true):
        return 2 * (Y_predicted - Y_true) / np.size(Y_true)

class CrossEntropy(Loss):

    def L(self, Y_predicted, Y_true):
        return -1 * np.mean(Y_true * np.log2(Y_predicted))

    def L_prime(self, Y_predicted, Y_true):
        return (Y_predicted - Y_true) / np.size(Y_true)