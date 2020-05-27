#!/usr/bin/env python3
""" creates a layer using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer using dropout"""
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initialize)
    drop = tf.layers.Dropout(keep_prob)
    return drop(layer(prev))
