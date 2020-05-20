#!/usr/bin/env python3
""" train with momentum"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """train with momentum"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
