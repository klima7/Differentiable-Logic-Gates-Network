from abc import ABC, abstractmethod


class Fun(ABC):

    def __init__(self, neg=False):
        self.neg = neg

    def boolean(self, a, b):
        value = self._boolean(a, b)
        if self.neg:
            value = not value
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
        return True

    def _real(self, a, b):
        return 1

    def _deriv(self, a, b):
        return 0, 0


class AFun(Fun):

    def _boolean(self, a, b):
        return a

    def _real(self, a, b):
        return a

    def _deriv(self, a, b):
        return 1, 0


class BFun(Fun):

    def _boolean(self, a, b):
        return b

    def _real(self, a, b):
        return b

    def _deriv(self, a, b):
        return 0, 1


class AndFun(Fun):

    def _boolean(self, a, b):
        return a and b

    def _real(self, a, b):
        return a * b

    def _deriv(self, a, b):
        return b, a


class OrFun(Fun):

    def _boolean(self, a, b):
        return a or b

    def _real(self, a, b):
        return a + b - a*b

    def _deriv(self, a, b):
        deriv_a = 1 - b
        deriv_b = 1 - a
        return deriv_a, deriv_b


class XorFun(Fun):

    def _boolean(self, a, b):
        return a != b

    def _real(self, a, b):
        return a + b - 2*a*b

    def _deriv(self, a, b):
        deriv_a = 1 - 2*b
        deriv_b = 1 - 2*a
        return deriv_a, deriv_b


class ImpABFun(Fun):

    def _boolean(self, a, b):
        return not a or b

    def _real(self, a, b):
        return 1 - a + a * b

    def _deriv(self, a, b):
        deriv_a = -1 + b
        deriv_b = a
        return deriv_a, deriv_b


class ImpBAFun(Fun):

    def _boolean(self, a, b):
        return not b or a

    def _real(self, a, b):
        return 1 - b + a * b

    def _deriv(self, a, b):
        deriv_a = b
        deriv_b = -1 + a
        return deriv_a, deriv_b


FUNCTIONS = [
    TrueFun(), TrueFun(neg=True),
    AFun(), AFun(neg=True),
    BFun(), BFun(neg=True),
    AndFun(), AndFun(neg=True),
    OrFun(), OrFun(neg=True),
    XorFun(), XorFun(neg=True),
    ImpABFun(), ImpABFun(neg=True),
    ImpBAFun(), ImpBAFun(neg=True),
]
