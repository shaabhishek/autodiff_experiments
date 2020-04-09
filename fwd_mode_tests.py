import torch

from fwd_mode import FwdNumber, exp, sin, TensorNumber


def f1(x1, x2, scalar_constructor):
    out = (x1 + x2 / scalar_constructor(4)).exp().sin()
    return out


def f2(x1, x2, scalar_constructor):
    a = x1 / x2
    b = exp(x2)
    out = (sin(a) + a - b) * (a - b)
    return out


def test_with_pytorch(f):
    x1, x2 = torch.Tensor([1.]).requires_grad_(), torch.Tensor([1.])
    out = f(x1, x2, TensorNumber())
    out.backward()
    print(out.data, x1.grad)


def test_with_fwdnumber(f):
    x1, x2 = FwdNumber(1, 1), FwdNumber(1)
    out = f(x1, x2, FwdNumber)
    print(out)