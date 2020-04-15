#!/usr/bin/env python3
""" add matrix"""


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


def add_matrices(mat1, mat2):
    """ """
    size1 = matrix_shape(mat1)
    size2 = matrix_shape(mat2)
    if size1 == size2:
        if len(size1) == 1:
            result = [mat1[i] + mat2[i] for i in range(len(mat1))]
            return result
        elif len(size1) == 2:
            result = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                      for i in range(len(mat1))]
            return result
        else:
            new = []
            for i in range(len(mat1)):
                result = add_matrices(mat1[i], mat2[i])
                new.append(result)
            return new
    else:
        return None
