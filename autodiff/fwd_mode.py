import math

import torch


class BaseFwdNumber:
    def __init__(self, value, derivative=0.):
        self._val = float(value)
        self._dot = float(derivative)

    def __repr__(self):
        return f"FwdNum({self._val:.4f}, {self._dot:.4f})"


class FwdNumber(BaseFwdNumber):

    def __truediv__(self, other: BaseFwdNumber):
        """
        v = a / b
        dv = (da.b - a.db) / (b^2)
        """
        return FwdNumber(
            self._val / other._val,
            (self._dot * other._val - other._dot * self._val) / (other._val * other._val)
        )

    def __mul__(self, other: BaseFwdNumber):
        """
        v = a * b
        dv = da.b + a.db
        """
        return FwdNumber(
            self._val * other._val,
            self._dot * other._val + self._val * other._dot
        )

    def __add__(self, other: BaseFwdNumber):
        """
        v = a + b
        dv = da + db
        """
        return FwdNumber(
            self._val + other._val,
            self._dot + other._dot
        )

    def __sub__(self, other: BaseFwdNumber):
        """
        v = a - b
        dv = da - db
        """
        return FwdNumber(
            self._val - other._val,
            self._dot - other._dot
        )

    def exp(self):
        """
        v = exp(a)
        dv = exp(a).da
        """
        return FwdNumber(
            math.exp(self._val),
            math.exp(self._val) * self._dot
        )

    def sin(self):
        """
        v = sin(a)
        dv = cos(a).da
        """
        return FwdNumber(
            math.sin(self._val),
            math.cos(self._val) * self._dot
        )


def exp(num: FwdNumber):
    return num.exp()


def sin(num: FwdNumber):
    return num.sin()


class TensorNumber:
    def __call__(self, num): return torch.Tensor([num])
