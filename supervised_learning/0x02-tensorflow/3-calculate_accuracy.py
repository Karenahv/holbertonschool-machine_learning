#!/usr/bin/env python3
"""calculates the accuracy of the prediction"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates accuracy of the prediction"""
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
