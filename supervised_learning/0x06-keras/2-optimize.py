#!/usr/bin/env python3
""" Optimization of a Keras Model"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Optimization of a Keras Model"""
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss=K.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return None
