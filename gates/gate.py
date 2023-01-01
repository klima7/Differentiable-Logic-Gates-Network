import numpy as np

import funs


class Gate:

    FUNCTIONS = [
        funs.TrueFun(), funs.TrueFun(neg=True),
        funs.IdentityFun(), funs.IdentityFun(neg=True),
        funs.AndFun(), funs.AndFun(neg=True),
        funs.OrFun(), funs.OrFun(neg=True),
        funs.XorFun(), funs.XorFun(neg=True),
        funs.ImpABFun(), funs.ImpABFun(neg=True),
        funs.ImpBAFun(), funs.ImpBAFun(neg=True),
    ]

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = self.__create_weights()

    def propagate_real(self, input):
        soft_weights = self.__softmax(self.weights)

    def backpropagate(self, gradient, learning_rate):
        next_gradient = self.__get_next_gradient(gradient)
        self.__update_weights(gradient, learning_rate)
        return next_gradient

    def __get_next_gradient(self, gradient):
        return ...

    def __update_weights(self, gradient, learning_rate):
        ...

    @staticmethod
    def __softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        y = e / s if s != 0 else np.zeros_like(e)
        return y
