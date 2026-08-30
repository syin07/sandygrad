import pytest
import torch

from sandygrad.scalar import Scalar

class TestForward:
    def test_add(self):
        assert (Scalar(2.0) + Scalar(3.0)).data == 5.0

    def test_sub(self):
        assert (Scalar(3.0) - 2).data == 1.0
        assert (3 - Scalar(2.0)).data == 1.0

    def test_mul(self):
        assert (Scalar(2.0) * Scalar(3.0)).data == 6.0
        assert (2 * Scalar(3.0)).data == 6.0
        assert (Scalar(2.0) * 3).data == 6.0

    def test_div(self):
        assert (Scalar(4.0) / Scalar(2.0)).data == 2.0
        assert (4 / Scalar(2.0)).data == 2.0
        assert (Scalar(4.0) / 2).data == 2.0

    @pytest.mark.parametrize("base,exp,expected", [
        (2.0, 3, 8.0),
        (3.0, 0, 1.0),
        (2.0, -1, 0.5),
    ])
    def test_pow(self, base, exp, expected):
        a = Scalar(base)
        assert (a ** exp).data == pytest.approx(expected)

def test_gradients():
    a = Scalar(5.0)
    b = 3 * a + 5 + a
    c = (b ** 2).ReLU()
    d = b + c + a - 5

    d.backward()
    sa, sd = a, d

    a = torch.Tensor([5.0]).double()
    a.requires_grad = True
    b = 3 * a + 5 + a
    c = (b ** 2).relu()
    d = b + c + a - 5

    d.backward()
    ta, td = a, d

    # forward pass
    assert sd.data == td.data.item()
    # backward pass
    assert sa.grad == ta.grad.item()


