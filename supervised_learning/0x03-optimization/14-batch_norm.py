#!/usr/bin/env python3
"""batch norm tensorflow """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch norm tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    hidden_l = tf.layers.Dense(units=n, kernel_initializer=init,
                               activation=None)
    m, s = tf.nn.moments(hidden_l(prev), axes=[0])

    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)
    epsilon = tf.constant(1e-8)
    z = hidden_l(prev)
    z_norm = tf.nn.batch_normalization(x=z, mean=m, variance=s,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)
    if activation is None:
        return z_norm
    return activation(z_norm)
