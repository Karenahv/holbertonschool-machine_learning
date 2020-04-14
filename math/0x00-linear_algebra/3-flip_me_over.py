#!/usr/bin/env python3
""" transpose a matrix"""


def matrix_transpose(matrix):
    """transpose a matrix"""
    result = [[matrix[j][i] for j in range(len(matrix))]
              for i in range(len(matrix[0]))]
    return result
