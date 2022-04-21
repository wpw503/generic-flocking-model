import math

import numpy as np


class MLP:
    def __init__(self, n_in, hidden, n_out, sigmoid_last=True):
        self.n_in = n_in + 1  # add bias
        self.n_out = n_out
        self.hidden = hidden
        self.sigmoid_last = sigmoid_last

        self.layers = []

        if len(hidden) == 0:
            self.layers.append(np.random.randn(self.n_out, self.n_in))
        else:
            self.layers.append(np.random.randn(self.hidden[0], self.n_in))

            for i in range(1, len(hidden)):
                self.layers.append(np.random.randn(self.hidden[i], self.hidden[i - 1]))

            self.layers.append(np.random.randn(self.n_out, self.hidden[-1]))

    def leaky_ReLU(self, x):
        if x > 0:
            return x
        else:
            return x * 0.02

    def sigmoid(self, x):
        try:
            ans = (1 / (1 + math.exp(-x)))
        except OverflowError:
            ans = float('inf')
        return ans

    def feed_forward(self, inputs):
        output = inputs + [1]

        for layer in self.layers[:-1]:
            output = [self.leaky_ReLU(x) for x in np.dot(layer, output)]

        if self.sigmoid_last:
            output = [self.sigmoid(x) for x in np.dot(self.layers[-1], output)]
        else:
            output = [self.leaky_ReLU(x) for x in np.dot(self.layers[-1], output)]

        return output

    def get_weights(self):
        weights = []

        for layer in self.layers:
            weights += list(layer.flatten())

        return weights

    def set_weights(self, weights):
        self.layers = []

        if len(self.hidden) == 0:
            self.layers.append(np.array(weights).reshape((self.n_out, self.n_in)))

        else:
            self.layers.append(np.array(weights[0:(self.hidden[0] * self.n_in)]).reshape((self.hidden[0], self.n_in)))

            prev_index = (self.hidden[0] * self.n_in)

            for i in range(1, len(self.hidden)):
                self.layers.append(
                    np.array(weights[prev_index:prev_index + (self.hidden[i] * self.hidden[i - 1])]).reshape(
                        (self.hidden[i], self.hidden[i - 1])))
                prev_index = prev_index + (self.hidden[i] * self.hidden[i - 1])

            self.layers.append(np.array(weights[prev_index:prev_index + (self.hidden[-1] * self.n_out)]).reshape(
                (self.n_out, self.hidden[-1])))