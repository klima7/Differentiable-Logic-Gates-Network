import numpy as np

from gate import Gate


class Layer:

    def __init__(self, size, input_size, connections_rate):
        self.size = size
        self.input_size = input_size
        self.connections_rate = connections_rate
        self.gates = [Gate(input_size, connections_rate) for _ in range(size)]

    def propagate_boolean(self, input):
        return np.array([gate.propagate_boolean(input) for gate in self.gates])

    def propagate_real(self, input):
        return np.array([gate.propagate_real(input) for gate in self.gates])

    def backpropagate(self, gradient, learning_rate):
        return np.array([gate.backpropagate(gradient, learning_rate) for gate in self.gates])
