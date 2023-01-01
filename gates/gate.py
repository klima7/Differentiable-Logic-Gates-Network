import numpy as np

import funs


class Gate:

    FUNCTIONS = [
        funs.TrueFun(), funs.TrueFun(neg=True),
        funs.AFun(), funs.AFun(neg=True),
        funs.AndFun(), funs.AndFun(neg=True),
        funs.OrFun(), funs.OrFun(neg=True),
        funs.XorFun(), funs.XorFun(neg=True),
        funs.ImpABFun(), funs.ImpABFun(neg=True),
    ]

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.zeros((len(self.FUNCTIONS), self.input_size, self.input_size))

    def propagate_boolean(self, input):
        ...

    def propagate_real(self, input):
        ...

    def backpropagate(self, gradient, learning_rate):
        next_gradient = self.__get_next_gradient(gradient)
        self.__update_weights(gradient, learning_rate)
        return next_gradient

    def __get_next_gradient(self, gradient):
        return ...

    def __update_weights(self, gradient, learning_rate):
        ...
