#!/usr/bin/env python3
""" hot matrix to labels vector"""

import numpy as np


def one_hot_decode(one_hot):
    """ hot matrix to labels vector"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.size is 0:
        return None
    return np.argmax(one_hot, axis=0)
