#!/usr/bin/env python3
"""determines if a markov
 chain is absorbing"""

import numpy as np


def absorbing(P):
    """

    :param P:is a is a square 2D numpy.ndarray
     of shape (n, n) representing the transition matrix
    :return: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n = P.shape[0]
    if n != P.shape[1]:
        return False
    if (P == np.eye(P.shape[0])).all():
        return True
    flag = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if i == j and P[i][j] == 1:
                state_absorbing = i
                flag = 1
                break
    if flag == 0:
        return False
    count = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if i == j and ((i + 1) < len(P)) and ((j + 1) < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True
