"""Training functions for iteratively training neural networks."""
import logging


def stochasticGradientDescent(net,
                              training_set, training_labels,
                              learning_rate=0.01,
                              epochs=100, evaluate_after=5):
    """Train a neural net using stochastic gradient descent

    net - neural net to be trained
    training_set - training data
    training_labels - correct outputs for the training data
    learning_rate - initial learning rate for SGD
    epochs - number of times to iterate through the training data
    evaluate_after - evaluate loss after this many epochs
    """
    losses = list()
    net.set_learning_rate(learning_rate)

    for epoch in xrange(epochs):
        if (epoch % evaluate_after) == 0:
            losses.append(net.calculate_loss(training_set, training_labels))
            logging.info("Loss at epoch %d: %f", epoch, losses[-1])

            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                logging.info("Adjusting learning rate")
                net.set_learning_rate(learning_rate)

        for i in xrange(len(training_set)):
            net.gradient_step(training_set[i], training_labels[i])

    return losses

