#!/usr/bin/env python3
""" Definiteness """
import numpy as np


def definiteness(matrix):
    """ calculates the definiteness of a matrix:
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    w, v = np.linalg.eig(matrix)

    pos, neg, zer = 0, 0, 0
    for e_val in w:
        if e_val > 0:
            pos += 1
        elif e_val < 0:
            neg += 1
        else:
            zer += 1

    # classify
    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    if pos and not neg and zer:
        return 'Positive semi-definite'
    if not pos and neg and not zer:
        return 'Negative definite'
    if not pos and neg and zer:
        return 'Negative semi-definite'
    return 'Indefinite'
