from gate import Gate


class Layer:

    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.gates = [Gate(input_size) for _ in range(size)]

    def propagate_boolean(self, input):
        return ...

    def propagate_real(self, input):
        return ...

    def backpropagate(self, gradients, learning_rate):
        return ...
