import random
from itertools import combinations

import numpy as np

from . import funs


class Gate:

    FUNCTIONS = [funs.TrueFun, funs.IdentityFun, funs.AndFun, funs.OrFun, funs.XorFun, funs.ImpABFun, funs.ImpBAFun]

    def __init__(self, input_size, connections_rate):
        self.input_size = input_size
        self.connections = connections_rate

        self.funs = self.__create_desired_functions_variants()
        self.weights = np.random.randn(len(self.funs))

        self.__input = None
        self.__funs_results = None

    def propagate_boolean(self, input):
        max_weight_idx = np.argmax(self.weights)
        fun = self.funs[max_weight_idx]
        return fun.boolean(input)

    def propagate_real(self, input):
        funs_results = np.array([fun.real(input) for fun in self.funs])
        soft_weights = self.__softmax(self.weights)
        self.__input = input
        self.__funs_results = funs_results
        return soft_weights @ funs_results

    def backpropagate(self, gradient, learning_rate):
        next_gradient = self.__get_next_gradient(gradient)
        self.__update_weights(gradient, learning_rate)
        return next_gradient

    def __get_next_gradient(self, gradient):
        derivs = np.column_stack([fun.deriv(self.__input) for fun in self.funs])
        derivs = derivs * self.__softmax(self.weights)
        derivs = np.sum(derivs, axis=1)
        return gradient * derivs

    def __update_weights(self, gradient, learning_rate):
        self.weights -= gradient * self.__funs_results * learning_rate

    @staticmethod
    def __softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        y = e / s if s != 0 else np.zeros_like(e)
        return y

    def __create_desired_functions_variants(self):
        all_variants = self.__create_all_functions_variants()
        desired_count = self.connections if isinstance(self.connections, int) else int(self.connections * len(all_variants))
        return random.choices(all_variants, k=desired_count)

    def __create_all_functions_variants(self):
        all_variants = []
        for func_cls in self.FUNCTIONS:
            func_variants = self.__create_function_variants(func_cls)
            all_variants.extend(func_variants)
        return all_variants

    def __create_function_variants(self, func_cls):
        vars_combinations = list(combinations(range(self.input_size), func_cls.ARGS_COUNT))
        funcs = [func_cls(vars_combination, neg) for vars_combination in vars_combinations for neg in [True, False]]
        return funcs
