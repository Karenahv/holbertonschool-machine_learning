#!/usr/bin/env python3
"""mat mul """


def mat_mul(mat1, mat2):
    """ mat mult"""
    if len(mat1[0]) == len(mat2):
        res = [[0 for col in range(len(mat2[0]))] for row in range(len(mat1))]
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat2)):
                    # resulted matrix
                    res[i][j] += mat1[i][k] * mat2[k][j]
        return res
