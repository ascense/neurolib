import numpy
import theano
import six
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class _BaseConnection(object):
    def __init__(self, in_layer, out_layer, name=None, init_W=None, init_b=None):
        """Create a connection between layers of the neural net

        name defines the name of the connection

        init_W is a function init_W(out_dim, in_dim) used to initialize the
         weight matrix

        init_b is a function init_b(out_dim) used to initialize the
         bias vector
        """
        self.name = name if name else "{}-{}".format(in_layer.name, out_layer.name)
        self.in_layer = in_layer
        self.out_layer = out_layer

        if not init_W:
            init_W = numpy.random.randn
        if not init_b:
            init_b = numpy.zeros

        self._W = theano.shared(
            init_W(in_layer.dim, out_layer.dim),
            name="W{}".format(self.name)
        )
        self._b = theano.shared(
            init_b(out_layer.dim),
            name="b{}".format(self.name)
        )
        self._z = None

    def derivates_for(self, cost):
        """Create derivate calculations for connection

        returns tuple(dW, db), where W is weights, b is bias,
         and d is derivate with regard to the value of 'cost'
        """
        dW = theano.tensor.grad(cost, self._W)
        db = theano.tensor.grad(cost, self._b)

        return (dW, db)

    @property
    def weights(self):
        return self._W

    @property
    def biases(self):
        return self._b

    @property
    def output_term(self):
        return self._z

    @abstractmethod
    def build(self, in_):
        raise NotImplementedError("Not implemented")

class Connection(_BaseConnection):
    """Full connection with weights and bias

    Equation: output = input.dot(W) + b
    """

    def build(self, in_):
        self._z = in_.dot(self._W) + self._b

        return self._z

