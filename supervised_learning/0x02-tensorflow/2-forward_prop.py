#!/usr/bin/env python3
""" creates the forward propagation graph for the neural network"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network"""
    sess = tf.Session()
    init = tf.global_variables_initializer()
    for layer in range(len(layer_sizes)):
        y = create_layer(x, layer_sizes[layer], activations[layer])
        x = y
    return x
