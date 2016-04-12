import numpy
import theano
import six
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class _BaseLayer(object):
    def __init__(self, dim, name):
        self._dim = dim
        self._name = name

        self._inputs = list()
        self._outputs = list()

        # activation term
        self._a = None

    def add_input(self, connection):
        self._inputs.append(connection)

    @property
    def inputs(self):
        return self._inputs

    def add_output(self, connection):
        self._outputs.append(connection)

    @property
    def outputs(self):
        return self._outputs

    @property
    def dim(self):
        return self._dim

    @property
    def name(self):
        return self._name

    @property
    def output_term(self):
        return self._a

    def __len__(self):
        return self._dim

    @abstractmethod
    def activate(self, in_):
        raise NotImplementedError("Not implemented")

    def build(self, inputs):
        self._a = self.activate(reduce(lambda x, y: x+y, inputs))
        return self._a


class LinearLayer(_BaseLayer):
    def activate(self, in_):
        return in_


class TanhLayer(_BaseLayer):
    def activate(self, in_):
        return theano.tensor.tanh(in_)


class SigmoidLayer(_BaseLayer):
    def activate(self, in_):
        return theano.tensor.sigmoid(in_)


class SoftmaxLayer(_BaseLayer):
    def activate(self, in_):
        return theano.tensor.nnet.softmax(in_)

