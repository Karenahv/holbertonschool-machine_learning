#!/usr/bin/env python3
"""add, substraction, multiplication, division matrix """


import numpy as np


def np_elementwise(mat1, mat2):
    """add, substraction, mult, div"""
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    return (add, sub, mul, div)
