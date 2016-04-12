import theano
import logging
import six
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class _BaseLoss(object):
    def __init__(self, learning_rate):
        self._epsilon = learning_rate

    @property
    def learning_rate(self):
        return self._epsilon

    @learning_rate.setter
    def learning_rate(self, value):
        logging.info("Setting learning rate to %f", value)
        self._epsilon = value

    @abstractmethod
    def build(self, net):
        """Initialize loss function for neural net 'net'"""
        NotImplementedError("Not implemented")

    @abstractmethod
    def gradient_step(self, training_in, training_out):
        """Do one step of training for the net"""
        NotImplementedError("Not implemented")

    @abstractmethod
    def calculate(self, training_in, training_out):
        """Calculate the current loss"""
        NotImplementedError("Not implemented")


class CrossEntropyLoss(_BaseLoss):
    def __init__(self, epsilon=0.01, lambda_=0.01, num_examples=None):
        self._epsilon = epsilon
        self._lambda = lambda_
        self._num_examples = num_examples

        self._f_loss = None
        self._f_gradient_step = None

    def _build_loss_reg(self, net):
        """Build regularization term for the loss function"""
        if not self._num_examples:
            logging.info("Not building regularization term,"
                         " as training data size is unknown")
            return None

        loss_sums = None
        for connection in net.connections():
            sum_ = theano.tensor.sum(theano.tensor.sqr(connection.weights))
            if loss_sums is None:
                loss_sums = sum_
            else:
                loss_sums += sum_

        loss_reg = 1.0/self._num_examples * self._lambda/2.0 * (loss_sums)

        return loss_reg

    def _build_loss(self, net, reg_term):
        """Build the loss function

        returns the loss term
        """
        loss = theano.tensor.nnet.categorical_crossentropy(
            net.output_term,
            net.training_input_term
        ).mean()
        if reg_term:
            logging.debug("Using loss regularization term")
            loss += reg_term

        self._f_loss = theano.function(
            [net.input_term, net.training_input_term],
            loss
        )

        return loss

    def _build_gradient_step(self, net, loss_term):
        gradient_updates = list()
        for connection in net.connections():
            (d_weight, d_bias) = connection.derivates_for(loss_term)
            gradient_updates.append(
                (connection.weights,
                 connection.weights - self._epsilon * d_weight)
            )
            gradient_updates.append(
                (connection.biases,
                 connection.biases - self._epsilon * d_bias)
            )

        self._f_gradient_step = theano.function(
            [net.input_term, net.training_input_term],
            updates=gradient_updates
        )

    def build(self, net):
        logging.debug("Building %s for network '%s'",
                      self.__class__.__name__, net.name)
        reg_term = self._build_loss_reg(net)
        loss_term = self._build_loss(net, reg_term)
        self._build_gradient_step(net, loss_term)

    def gradient_step(self, training_in, training_out):
        """Do one step of gradient descent"""
        return self._f_gradient_step(training_in, training_out)

    def calculate(self, training_in, training_out):
        """Calculate loss

        Depending on the size of the training data set, this can
         be a very costly operation.
        """
        return self._f_loss(training_in, training_out)

