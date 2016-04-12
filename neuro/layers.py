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
    def build(self, in_):
        raise NotImplementedError("Not implemented")


class LinearLayer(_BaseLayer):
    def build(self, in_):
        self._a = in_[0]
        return self._a


class TanhLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.tanh(in_[0])
        return self._a


class SigmoidLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.sigmoid(in_[0])
        return self._a


class SoftmaxLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.nnet.softmax(in_[0])
        return self._a

