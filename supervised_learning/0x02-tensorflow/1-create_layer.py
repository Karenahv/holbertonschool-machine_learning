#!/usr/bin/env python3
""" returns the tensor output of the layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """ return the tensor output of the layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            name='layer',
                            kernel_initializer=init,
                            activation=activation)
    layer_tensor = layer(prev)
    
    return layer_tensor
