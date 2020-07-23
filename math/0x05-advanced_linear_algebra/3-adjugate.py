#!/usr/bin/env python3
"""matriz adjugate"""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """calculate adjugate matrix"""
    if (type(matrix) != list or len(matrix) == 0 or
            not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    lm = len(matrix)
    if lm == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all([len(n) == lm for n in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if lm == 1 and len(matrix[0]) == 1:
        return [[1]]
    if lm == 1 and len(matrix[0]) == 1:
        return [[1]]
    cofactor_matrix = cofactor(matrix)
    lcm = len(cofactor_matrix)
    adjugate_matrix = []
    for i in range(lcm):
        temp_matrix = []
        for j in range(len(cofactor_matrix[0])):
            temp_matrix.append(cofactor_matrix[j][i])
        adjugate_matrix.append(temp_matrix)
    return adjugate_matrix
