#!/usr/bin/env python3
"""save models weights"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """save models weights"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ load model weights"""
    network.load_weights(filename)
    return None
