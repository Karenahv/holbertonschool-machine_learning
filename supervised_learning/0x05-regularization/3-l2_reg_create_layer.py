#!usr/bin/env python3
"""creates tensorflow layer with l2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates tensorflow layer with l2 regularization"""
    reg_weights = tf.contrib.layers.l2_regularizer(lambtha)
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initialize,
                            kernel_regularizer=reg_weights)
    return layer(prev)
