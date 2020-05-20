#!/usr/bin/env python3
""" learning rate decay operation in tensorflow """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate,
                        global_step,
                        decay_step):
    """ learning rate decay operation in tensorflow """
    alpha_new = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
    return alpha_new
