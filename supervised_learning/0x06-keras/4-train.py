#!/usr/bin/env python3
""" train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ train a model using keras"""

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle)
    return history
