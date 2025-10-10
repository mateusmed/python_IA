

import numpy as np
import activation_functions as activation_functions

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0.0
        self.delta = 0.0

    def activate(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        self.output = activation_functions.sigmoid(z)
        return self.output

    def update_weights(self, inputs, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.delta * inputs[i]
        self.bias -= learning_rate * self.delta