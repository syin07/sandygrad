from sandytorch.scalar import Scalar



def test_add():
    assert (Scalar(2.0) + Scalar(3.0)).data == 5.0