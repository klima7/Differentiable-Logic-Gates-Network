from abc import ABC, abstractmethod

import numpy as np


class LogicFun(ABC):

    def __init__(self, neg=False):
        self.neg = neg

    def boolean(self, *args):
        value = self._boolean(*args)
        if self.neg:
            value = ~value
        return value

    def real(self, *args):
        value = self._real(*args)
        if self.neg:
            value = 1 - value
        return value

    def deriv(self, *args):
        value = self._deriv(*args)
        if self.neg:
            value = -value
        return value

    @abstractmethod
    def _boolean(self, *args):
        pass

    @abstractmethod
    def _real(self, *args):
        pass

    @abstractmethod
    def _deriv(self, *args):
        pass


class NoArgLogicFun(LogicFun):

    @abstractmethod
    def _boolean(self):
        pass

    @abstractmethod
    def _real(self):
        pass

    def _deriv(self, *args):
        return None


class OneArgLogicFun(LogicFun):

    @abstractmethod
    def _boolean(self, a):
        pass

    @abstractmethod
    def _real(self, a):
        pass

    @abstractmethod
    def _deriv(self, a):
        pass


class TwoArgLogicFun(LogicFun):

    @abstractmethod
    def _boolean(self, a, b):
        pass

    @abstractmethod
    def _real(self, a, b):
        pass

    @abstractmethod
    def _deriv(self, a, b):
        pass


class TrueFun(NoArgLogicFun):

    def _boolean(self):
        return True

    def _real(self):
        return 1


class IdentityFun(OneArgLogicFun):

    def _boolean(self, a):
        return np.array(a)

    def _real(self, a):
        return np.array(a)

    def _deriv(self, a):
        return np.ones_like(a)


class AndFun(TwoArgLogicFun):

    def _boolean(self, a, b):
        return np.logical_and(a, b)

    def _real(self, a, b):
        return a * b

    def _deriv(self, a, b):
        return np.array(b), np.array(a)


class OrFun(TwoArgLogicFun):

    def _boolean(self, a, b):
        return np.logical_or(a, b)

    def _real(self, a, b):
        return a + b - a*b

    def _deriv(self, a, b):
        deriv_a = 1 - b
        deriv_b = 1 - a
        return deriv_a, deriv_b


class XorFun(TwoArgLogicFun):

    def _boolean(self, a, b):
        return np.logical_xor(a, b)

    def _real(self, a, b):
        return a + b - 2*a*b

    def _deriv(self, a, b):
        deriv_a = 1 - 2*b
        deriv_b = 1 - 2*a
        return deriv_a, deriv_b


class ImpABFun(TwoArgLogicFun):

    def _boolean(self, a, b):
        return np.logical_or(~a, b)

    def _real(self, a, b):
        return 1 - a + a * b

    def _deriv(self, a, b):
        deriv_a = -1 + b
        deriv_b = np.array(a)
        return deriv_a, deriv_b


class ImpBAFun(TwoArgLogicFun):

    def _boolean(self, a, b):
        return np.logical_or(~b, a)

    def _real(self, a, b):
        return 1 - b + a * b

    def _deriv(self, a, b):
        deriv_a = np.array(b)
        deriv_b = -1 + a
        return deriv_a, deriv_b
