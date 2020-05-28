#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    """
    k_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    k_reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=k_ini, kernel_regularizer=k_reg)
    return layer(prev)
