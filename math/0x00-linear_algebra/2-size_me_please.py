#!/usr/bin/env python3
""" shape of a matrix"""


def size_matrix(matrix, size):
    """size of each list"""
    if isinstance(matrix, list):
        size += [len(matrix)]
        if (len(matrix) > 0):
            return size_matrix(matrix[0], size)
        else:
            return [0]
    else:
        return size


def matrix_shape(matrix):
    """ return shape of a matrix"""
    size = []
    return size_matrix(matrix, size)
