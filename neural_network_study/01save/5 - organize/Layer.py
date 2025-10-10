
import Neuron as Neuron
import activation_functions as activation_functions


class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.outputs = []
        self.errors = []

    def forward(self, inputs):
        self.outputs = [n.activate(inputs) for n in self.neurons]
        return self.outputs

    def calculate_error(self, expected=None, next_layer=None):

        self.errors = []
        if next_layer is None:  # camada de saída
            for n, exp in zip(self.neurons, expected):
                self.errors.append(n.output - exp)
        else:
            for i, n in enumerate(self.neurons):
                err = sum(next_n.weights[i] * next_n.delta for next_n in next_layer.neurons)
                self.errors.append(err)

    def calculate_deltas(self):
        for i, n in enumerate(self.neurons):
            n.delta = self.errors[i] * activation_functions.sigmoid_derivative(n.output)

