#!/usr/bin/env python3
""" converts numeric label into hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """convert to hot matrix"""
    if not isinstance(classes, int):
        return None
    if not isinstance(Y, np.ndarray):
        return None
    if Y.size is 0:
        return None
    if classes < Y.max() + 1:
        return None
    return np.squeeze(np.eye(classes)[Y.reshape(-1)]).T
