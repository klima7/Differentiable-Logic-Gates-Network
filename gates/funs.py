from abc import ABC, abstractmethod

import numpy as np


class Fun(ABC):

    def __init__(self, neg=False):
        self.neg = neg

    def boolean(self, a, b):
        value = self._boolean(a, b)
        if self.neg:
            value = ~value
        return value

    def real(self, a, b):
        value = self._real(a, b)
        if self.neg:
            value = 1 - value
        return value

    def deriv(self, a, b):
        value = self._deriv(a, b)
        if self.neg:
            value = -value
        return value

    @abstractmethod
    def _boolean(self, a, b):
        pass

    @abstractmethod
    def _real(self, a, b):
        pass

    @abstractmethod
    def _deriv(self, a, b):
        pass


class TrueFun(Fun):

    def _boolean(self, a, b):
        return np.ones_like(a)

    def _real(self, a, b):
        return np.ones_like(a)

    def _deriv(self, a, b):
        return np.zeros_like(a), np.zeros_like(a)


class AFun(Fun):

    def _boolean(self, a, b):
        return np.array(a)

    def _real(self, a, b):
        return np.array(a)

    def _deriv(self, a, b):
        return np.ones_like(a), np.zeros_like(a)


class AndFun(Fun):

    def _boolean(self, a, b):
        return np.logical_and(a, b)

    def _real(self, a, b):
        return a * b

    def _deriv(self, a, b):
        return np.array(b), np.array(a)


class OrFun(Fun):

    def _boolean(self, a, b):
        return np.logical_or(a, b)

    def _real(self, a, b):
        return a + b - a*b

    def _deriv(self, a, b):
        deriv_a = 1 - b
        deriv_b = 1 - a
        return deriv_a, deriv_b


class XorFun(Fun):

    def _boolean(self, a, b):
        return np.logical_xor(a, b)

    def _real(self, a, b):
        return a + b - 2*a*b

    def _deriv(self, a, b):
        deriv_a = 1 - 2*b
        deriv_b = 1 - 2*a
        return deriv_a, deriv_b


class ImpABFun(Fun):

    def _boolean(self, a, b):
        return np.logical_or(~a, b)

    def _real(self, a, b):
        return 1 - a + a * b

    def _deriv(self, a, b):
        deriv_a = -1 + b
        deriv_b = np.array(a)
        return deriv_a, deriv_b
