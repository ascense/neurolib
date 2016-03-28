import numpy
import theano
import six
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class _BaseLayer(object):
    def __init__(self, dim, name):
        self._dim = dim
        self._name = name

        # activation term
        self._a = None

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
        self._a = in_
        return self._a


class TanhLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.tanh(in_)
        return self._a


class SigmoidLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.sigmoid(in_)
        return self._a


class SoftmaxLayer(_BaseLayer):
    def build(self, in_):
        self._a = theano.tensor.nnet.softmax(in_)
        return self._a

