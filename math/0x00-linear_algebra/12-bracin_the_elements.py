#!/usr/bin/env python3
"""add, substraction, multiplication, division matrix """


import numpy as np


def np_elementwise(mat1, mat2):
    return (np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2))
