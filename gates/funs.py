from abc import ABC, abstractmethod

import numpy as np


class LogicFun(ABC):
    ARGS_COUNT = None

    def __init__(self, var_indexes, neg=False):
        self.var_indexes = var_indexes
        self.neg = neg

    @property
    def idx0(self):
        return self.var_indexes[0]

    @property
    def idx1(self):
        return self.var_indexes[1]

    def boolean(self, inputs):
        value = self._boolean(inputs)
        if self.neg:
            value = not value
        return value

    def real(self, inputs):
        value = self._real(inputs)
        if self.neg:
            value = 1 - value
        return value

    def deriv(self, inputs):
        value = self._deriv(inputs)
        if self.neg:
            value = -value
        return value

    @abstractmethod
    def _boolean(self, inputs):
        pass

    @abstractmethod
    def _real(self, inputs):
        pass

    @abstractmethod
    def _deriv(self, inputs):
        pass


class TrueFun(LogicFun):
    ARGS_COUNT = 0

    def _boolean(self, inputs):
        return True

    def _real(self, inputs):
        return 1

    def _deriv(self, inputs):
        return np.zeros_like(inputs)


class IdentityFun(LogicFun):
    ARGS_COUNT = 1

    def _boolean(self, inputs):
        return inputs[self.idx0]

    def _real(self, inputs):
        return inputs[self.idx0]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = 1
        return deriv


class AndFun(LogicFun):
    ARGS_COUNT = 2

    def _boolean(self, inputs):
        return inputs[self.idx0] and inputs[self.idx1]

    def _real(self, inputs):
        return inputs[self.idx0] * inputs[self.idx1]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = inputs[self.idx1]
        deriv[self.idx1] = inputs[self.idx0]
        return deriv


class OrFun(LogicFun):
    ARGS_COUNT = 2

    def _boolean(self, inputs):
        return inputs[self.idx0] or inputs[self.idx1]

    def _real(self, inputs):
        return inputs[self.idx0] + inputs[self.idx1] - inputs[self.idx0]*inputs[self.idx1]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = 1 - inputs[self.idx1]
        deriv[self.idx1] = 1 - inputs[self.idx0]
        return deriv


class XorFun(LogicFun):
    ARGS_COUNT = 2

    def _boolean(self, inputs):
        return inputs[self.idx0] != inputs[self.idx1]

    def _real(self, inputs):
        return inputs[self.idx0] + inputs[self.idx1] - 2*inputs[self.idx0]*inputs[self.idx1]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = 1 - 2*inputs[self.idx1]
        deriv[self.idx1] = 1 - 2*inputs[self.idx0]
        return deriv


class ImpABFun(LogicFun):
    ARGS_COUNT = 2

    def _boolean(self, inputs):
        return not(inputs[self.idx0]) or inputs[self.idx1]

    def _real(self, inputs):
        return 1 - inputs[self.idx0] + inputs[self.idx0] * inputs[self.idx1]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = -1 + inputs[self.idx1]
        deriv[self.idx1] = inputs[self.idx0]
        return deriv


class ImpBAFun(LogicFun):
    ARGS_COUNT = 2

    def _boolean(self, inputs):
        return not(inputs[self.idx1]) or inputs[self.idx0]

    def _real(self, inputs):
        return 1 - inputs[self.idx1] + inputs[self.idx0] * inputs[self.idx1]

    def _deriv(self, inputs):
        deriv = np.zeros_like(inputs)
        deriv[self.idx0] = inputs[self.idx1]
        deriv[self.idx1] = -1 + inputs[self.idx0]
        return deriv
