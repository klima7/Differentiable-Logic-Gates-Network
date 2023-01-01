import numpy as np
from tqdm import tqdm

from layer import Layer


class Network:

    def __init__(self, input_size, layers_sizes):
        self.input_size = input_size
        self.layers = self.__create_layers(input_size, layers_sizes)

    def fit(self, inputs, outputs, epochs=1, learning_rate=0.01):
        for epoch_no in range(epochs):
            self.__learn_epoch(inputs, outputs, learning_rate, epoch_no)

    def predict_boolean(self, inputs):
        outputs = [self.__propagate_boolean(input) for input in inputs]
        return np.array(outputs)

    def predict_real(self, inputs):
        outputs = [self.__propagate_real(input) for input in inputs]
        return np.array(outputs)

    def __propagate_boolean(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.propagate_boolean(current_input)
        return current_input

    def __propagate_real(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.propagate_real(current_input)
        return current_input

    def __learn_epoch(self, inputs, outputs, learning_rate, epoch_no):
        shuffled_inputs, shuffled_outputs = self.__shuffle(inputs, outputs)
        iterator = tqdm(zip(shuffled_inputs, shuffled_outputs), total=len(inputs), desc=f'Epoch {epoch_no:.3f}')
        for input, output in iterator:
            self.__learn_single(input, output, learning_rate)

    def __learn_single(self, input, output, learning_rate):
        prediction = self.__propagate_real(input)
        gradients = self.__get_loss_deriv(prediction, output)
        self.__backpropagate(gradients, learning_rate)

    def __backpropagate(self, gradients, learning_rate):
        for layer in reversed(list(self.layers)):
            gradients = layer.backpropagate(gradients, learning_rate)
        return gradients

    @staticmethod
    def __shuffle(inputs, outputs):
        shuffled_indexes = np.arange(len(inputs))
        np.random.shuffle(shuffled_indexes)
        shuffled_inputs = inputs[shuffled_indexes]
        shuffled_outputs = outputs[shuffled_indexes]
        return shuffled_inputs, shuffled_outputs

    @staticmethod
    def __create_layers(input_size, layers_sizes):
        layers = []
        all_sizes = [input_size, *layers_sizes]
        for input_size, gates_count in zip(all_sizes, all_sizes[1:]):
            layer = Layer(size=gates_count, input_size=input_size)
            layers.append(layer)
        return layers

    @staticmethod
    def __get_loss_deriv(prediction, target):
        return 2 * (prediction - target) / prediction.size