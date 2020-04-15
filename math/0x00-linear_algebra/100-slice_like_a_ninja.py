#!/usr/bin/env python3
"""slices a matrix """


def np_slice(matrix, axes={}):
    """slice like a ninja"""
    for key in axes:
        axe=key
    values = axes[axe]
    value1 = values[0]
    value2 = values[1]
    if axe == 1:
        mat = matrix[:, value1:value2]
        return mat
    return None
