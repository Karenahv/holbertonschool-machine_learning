#!usr/bin/env python3
"""converts a label vector into a one hot matriz"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one hot matriz"""
    return K.utils.to_categorical(labels, classes)
