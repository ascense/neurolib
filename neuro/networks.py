import theano
import six
from abc import ABCMeta, abstractmethod

from .connections import Connection
from .layers import SoftmaxLayer
from .loss import CrossEntropyLoss


class NeuralNetError(Exception):
    pass


@six.add_metaclass(ABCMeta)
class _BaseNetwork(object):
    DEFAULT_LOSS_FUNCTION = {
        SoftmaxLayer: CrossEntropyLoss
    }

    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__

        self._in_layer = None
        self._out_layer = None

        self._layers = list()
        self._connections = dict()
        self._loss_function = None

        self._in_term = None
        self._train_in_term = None

    def set_input_layer(self, layer):
        self._in_layer = layer
        self.add_layer(layer)

    def set_output_layer(self, layer):
        self._out_layer = layer
        self.add_layer(layer)

    def add_layer(self, layer):
        self._layers.append(layer)
        self._connections[layer] = list()

    def add_connection(self, connection):
        if (connection.in_layer not in self._layers
                or connection.out_layer not in self._layers):
            raise NeuralNetError(
                "Both layers in connection must belong to the network"
            )

        self._connections[connection.in_layer].append(connection)

    def create_connection(self, in_name, out_name, *args, **kwargs):
        """Create a new connection between layers in network"""
        conn = Connection(self[in_name], self[out_name], *args, **kwargs)
        self.add_connection(conn)

    def set_loss_function(self, loss_func):
        self._loss_function = loss_func

    @property
    def input_term(self):
        return self._in_term

    @property
    def training_input_term(self):
        return self._train_in_term

    @property
    def output_term(self):
        return self._out_layer.output_term

    def layers(self):
        for layer in self._layers:
            yield layer

    def connections(self):
        for connections in self._connections.itervalues():
            for connection in connections:
                yield connection

    def __getitem__(self, k):
        for layer in self._layers:
            if layer.name == k:
                return layer

        for connection in self.connections():
            if connection.name == k:
                return connection

    def _connect(self, layer, in_):
        out = layer.build(in_)
        for connection in self._connections[layer]:
            self._connect(connection.out_layer, connection.build(out))

    def _build_loss_function(self):
        if not self._loss_function:
            if self._out_layer in self.DEFAULT_LOSS_FUNCTION:
                self._loss_function = self.DEFAULT_LOSS_FUNCTION[self._out_layer]()
            else:
                raise NeuralNetError("No loss function defined")

        self._loss_function.build(self)

    def _set_input_term(self, term):
        self._in_term = term

    def _set_training_input_term(self, term):
        self._train_in_term = term

    def build(self):
        """Prepare the network for use"""
        if not self._in_layer or not self._out_layer:
            raise NeuralNetError(
                "Input and output layers must be defined"
            )

        self._connect(self._in_layer, self._in_term)
        self._build_loss_function()

    def calculate_loss(self, training_in, training_out):
        """Calculate the current loss"""
        return self._loss_function.calculate(training_in, training_out)

    def gradient_step(self, training_in, training_out):
        """Do one step of training for the net"""
        return self._loss_function.gradient_step(training_in, training_out)

    @abstractmethod
    def activate(self, in_):
        """Calculate neural net output for the given input"""
        raise NotImplementedError("Not implemented")


class FeedForwardNetwork(_BaseNetwork):
    def __init__(self, name=None):
        super(FeedForwardNetwork, self).__init__(name)

        self._f_fw_prop = None
        self._f_gradient_step = None

        self._set_input_term(theano.tensor.matrix('X'))
        self._set_training_input_term(theano.tensor.lvector('y'))

    def build(self):
        super(FeedForwardNetwork, self).build()

        self._f_fw_prop = theano.function([self.input_term], self.output_term)

    def activate(self, in_):
        return self._f_fw_prop(in_)

