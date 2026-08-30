class Scalar:

    def __init__(self, data: float, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __str__(self):
        return f"Scalar({self.data}, {self.grad})"

    def __repr__(self):
        return f"Scalar(data={self.data},\ngrad={self.grad},_op={self._op})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(float(other))

        out = Scalar(
            data=self.data + other.data,
            _prev=(self, other),
            _op='+',
        )

        def _backward():
            self.grad += 1
            other.grad += 1

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(float(other))

        out = Scalar(
            data=self.data * other.data,
            _prev=(self, other),
            _op='*',
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Scalar(
            data= -self.data,
            _prev=(self,),
            _op='-',
        )

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        """Custom power method. Other object is a regular python int."""
        if not isinstance(other, int):
            raise TypeError("Power must be a python integer")

        out = Scalar(
            data=self.data**other,
            _prev=(self,Scalar(float(2))),
            _op='**',
        )

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return -self + other

    def __rtruediv__(self, other):
        return (self**-1) * other

    def ReLU(self):
        out = Scalar(
            data=max(float(0), self.data),
            _prev=(self,),
            _op='ReLU',
        )

        def _backward():
            local_grad = 1 if out.data > 0 else 0
            self.grad += local_grad * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0

        # traverse computational graph in topological order
        # makes sure each node's prerequisite nodes have been propagated to avoid partial propagation
        topo = []
        vis = set()
        def topological_sort(node):
            if node not in vis:
                vis.add(node)
                for child in node._prev:
                    topological_sort(child)
                topo.append(node)

        topological_sort(self)
        for node in reversed(topo):
            node._backward()


