#!/usr/bin/env python3
"""slices a matrix """


def np_slice(matrix, axes={}):
    """Slice like a ninja"""
    result = matrix[:]
    final_slice = []
    for axe in range(matrix.ndim):
        value = axes.get(axe, None)
        if value is not None:
            final_slice.append(slice(*value))
        else:
            final_slice.append(slice(value))

    return result[tuple(final_slice)]
