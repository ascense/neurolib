#!/usr/bin/env python
import numpy
import sklearn.datasets
import logging

import theano.printing

import neuro


numpy.random.seed(0)
train_X, train_y = sklearn.datasets.make_moons(200, noise=0.20)

logging.basicConfig(level=logging.DEBUG)

def build_model(dim_hidden=3):
    net = neuro.FeedForwardNetwork()

    net.set_input_layer(neuro.LinearLayer(2, 'in'))
    net.add_layer(neuro.TanhLayer(dim_hidden, 'hidden'))
    net.set_output_layer(neuro.SoftmaxLayer(2, 'out'))

    def init_weights(in_dim, out_dim):
        return numpy.random.randn(in_dim, out_dim) / numpy.sqrt(in_dim)

    net.create_connection('in', 'hidden', name='1', init_W=init_weights)
    net.create_connection('hidden', 'out', name='2', init_W=init_weights)

    net.set_loss_function(
        neuro.CrossEntropyLoss(epsilon=0.1, lambda_=0.01, num_examples=len(train_X))
    )

    net.build()

    return net

def train(net, num_passes=20000, print_loss=False):
    for i in xrange(num_passes):
        net.gradient_step(train_X, train_y)

        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" % (i, net.calculate_loss(train_X, train_y))

net = build_model()
train(net, print_loss=True)

