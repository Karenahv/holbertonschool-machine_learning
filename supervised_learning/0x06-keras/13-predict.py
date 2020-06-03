#!/usr/bin/env python3
"""predict """

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predict using a nerural network"""
    result = network.predict(data, verbose=verbose)
    return result
