#!usr/bin/env python3
"""creates tensorflow layer with l2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates tensorflow layer with l2 regularization"""
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg_weights = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=initialize,
                            kernel_regularizer=reg_weights)
    return layer(prev)
