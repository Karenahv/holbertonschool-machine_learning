#!/usr/bin/env python3
"""concatenates two matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrix"""
    if (len(mat1[0]) == len(mat2[0]) and axis == 0):
        result = []
        result += [elem.copy() for elem in mat1]
        result += [elem.copy() for elem in mat2]
        return result
    elif (len(mat1) == len(mat2) and axis == 1):
        res = [mat1[i] + mat2[i] for i in range(len(mat1))]
        return res
    else:
        return None
