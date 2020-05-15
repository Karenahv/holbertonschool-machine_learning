#!/usr/bin/env python3
""" returns two placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """ create two placeholders and
    return for the neural netwoek"""
    x = tf.placeholder(tf.float32, [None, nx], name='x')
    y = tf.placeholder(tf.float32, [None, classes], name='y')
    return (x, y)
