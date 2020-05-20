#!/usr/bin/env python3
"""trainning op RMSProp with Tensorflow"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """trainning op RMSProp with Tensorflow"""
    return (tf.train.RMSPropOptimizer(learning_rate=alpha,
                                      decay=beta2,
                                      epsilon=epsilon).minimize(loss))
