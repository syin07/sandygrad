from sandygrad.scalar import Scalar

import random

class Module:

    def zero_grad(self):
        zeros = 0
        for p in self.params():
            p.grad = 0.0

    def params(self):
        return []


class Neuron(Module):

    def __init__(self, nweights):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(nweights)]
        self.bias = Scalar(0)

    def __call__(self, x):
        """Returns the output of this single neuron."""
        return sum([xi * wi for xi, wi in zip(x, self.weights)]) + self.bias

    def __repr__(self):
        return f"Neuron({len(self.weights)})"

    def params(self):
        return self.weights + [self.bias]

class Layer(Module):

    def __init__(self, nin, nneurons, nonlinear=True):
        self.neurons = [Neuron(nin) for _ in range(nneurons)]
        self.nonlinear = nonlinear

    def __call__(self, x):
        """Returns a list of outputs for each neuron in the layer."""
        out = [neuron(x).ReLU() if self.nonlinear else neuron(x) for neuron in self.neurons]
        return out[0] if len(out)==1 else out

    def __repr__(self):
        return f"Layer({[n for n in self.neurons]})"

    def params(self):
        out = []
        for neuron in self.neurons:
            out += neuron.params()
        return out

class MLP(Module):

    def __init__(self, nin, layers):
        net = [nin] + layers
        self.layers = [Layer(net[i], net[i+1], nonlinear=i!=len(layers)-1) for i in range(len(layers))]

    def __call__(self, x):
        self.zeros = 0
        out = x

        for layer in self.layers:
            cur = layer(out)

            # for debugging dead neurons
            cnt = cur if isinstance(cur, list) else [cur]
            for val in cnt:
                self.zeros += 1 if val.data == 0.0 else 0

            out = cur

        return cur
        
    def __repr__(self):
        return f"MLP({[l for l in self.layers]})"

    def reset_dead_count(self):
        """Reset self.zeros. Used when debugging dead neurons."""
        self.zeros = 0

    def params(self):
        out = []
        for layer in self.layers:
            out += layer.params()
        return out

    
