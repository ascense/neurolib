import theano

from .connections import Connection
from .layers import SoftmaxLayer
from .loss import CrossEntropyLoss


class NeuralNetError(Exception):
    pass


class FeedForwardNetwork(object):
    """Feed Forward neural network implementation

    Usage example:

    >>> net = neuro.FeedForwardNetwork()
    >>> net.set_input_layer(neuro.LinearLayer(2, 'in'))
    >>> net.add_layer(neuro.TanhLayer(3, 'hidden'))
    >>> net.set_output_layer(neuro.SoftmaxLayer(2, 'out'))
    >>> net.create_connection('in', 'hidden')
    >>> net.create_connection('hidden', 'out')
    >>> net.build()
    """
    DEFAULT_LOSS_FUNCTION = {
        SoftmaxLayer: CrossEntropyLoss
    }

    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__

        self._in_layer = None
        self._out_layer = None

        self._layers = list()
        self._connections = list()
        self._loss_function = None

        self._in_term = None
        self._train_in_term = None

        self._f_fw_prop = None

    def set_input_layer(self, layer):
        if self._in_layer:
            raise NeuralNetError("Input layer already defined")
        self._in_layer = layer
        self.add_layer(layer)

    def set_output_layer(self, layer):
        if self._out_layer:
            raise NeuralNetError("Output layer already defined")
        self._out_layer = layer
        self.add_layer(layer)

    def add_layer(self, layer):
        self._layers.append(layer)

    def add_connection(self, connection):
        if (connection.in_layer not in self._layers
                or connection.out_layer not in self._layers):
            raise NeuralNetError(
                "Both layers in connection must belong to the network"
            )

        self._connections.append(connection)
        connection.in_layer.add_output(connection)
        connection.out_layer.add_input(connection)

    def create_connection(self, in_name, out_name, *args, **kwargs):
        """Create a new connection between layers in network"""
        conn = Connection(self[in_name], self[out_name], *args, **kwargs)
        self.add_connection(conn)

    def set_loss_function(self, loss_func):
        if self._loss_function:
            raise NeuralNetError("Loss function already defined")
        self._loss_function = loss_func

    def set_learning_rate(self, learning_rate):
        if not self._loss_function:
            raise NeuralNetError("No loss function defined")
        self._loss_function.learning_rate = learning_rate

    @property
    def input_term(self):
        return self._in_term

    @input_term.setter
    def input_term(self, term):
        if self._in_term:
            raise NeuralNetError("Input term already defined")
        self._in_term = term

    @property
    def training_input_term(self):
        return self._train_in_term

    @training_input_term.setter
    def training_input_term(self, term):
        if self._train_in_term:
            raise NeuralNetError("Training input term already defined")
        self._train_in_term = term

    @property
    def output_term(self):
        return self._out_layer.output_term

    def layers(self):
        return iter(self._layers)

    def connections(self):
        return iter(self._connections)

    def __getitem__(self, k):
        for layer in self._layers:
            if layer.name == k:
                return layer

        for connection in self._connections:
            if connection.name == k:
                return connection

    def _build_inputs(self):
        if not self.input_term:
            self.input_term = theano.tensor.matrix('X')
        if not self.training_input_term:
            self.training_input_term = theano.tensor.lvector('y')

    def _build_connections(self, layer, in_):
        out = layer.build(in_)
        for connection in layer.outputs:
            self._build_connections(
                connection.out_layer,
                connection.build(out)
            )
        return out

    def _build_loss_function(self):
        if not self._loss_function:
            if self._out_layer in self.DEFAULT_LOSS_FUNCTION:
                self._loss_function = self.DEFAULT_LOSS_FUNCTION[self._out_layer]()
            else:
                raise NeuralNetError("No loss function defined")

        self._loss_function.build(self)

    def build(self):
        """Prepare the network for use"""
        if not self._in_layer or not self._out_layer:
            raise NeuralNetError(
                "Input and output layers must be defined"
            )

        self._build_inputs()
        self._build_connections(self._in_layer, self._in_term)
        self._build_loss_function()

        self._f_fw_prop = theano.function([self.input_term], self.output_term)

    def calculate_loss(self, training_in, training_out):
        """Calculate the current loss"""
        return self._loss_function.calculate(training_in, training_out)

    def gradient_step(self, training_in, training_out):
        """Do one step of training for the net"""
        return self._loss_function.gradient_step(training_in, training_out)

    def activate(self, in_):
        """Calculate neural net output for the given input"""
        return self._f_fw_prop(in_)

