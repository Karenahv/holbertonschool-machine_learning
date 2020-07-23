#!/usr/bin/env python3
"""Determinant of a matrix"""


def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied
        :return: A copy of the given matrix
    """
    # Section 1: Get matrix dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 2: Create a new matrix of zeros
    MC = zeros_matrix(rows, cols)

    # Section 3: Copy values of M into the copy
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def determinant(matrix):
    """determinant of a matrix"""
    if (type(matrix) != list or len(matrix) == 0 or
            not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    lm = len(matrix)
    if lm == 1 and len(matrix[0]) == 0:
        return 1
    if not all([len(n) == lm for n in matrix]):
        raise ValueError("matrix must be a square matrix")
    if lm == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # Section 1: store indices in list for flexible row referencing
    indices = list(range(len(matrix)))

    # Section 2: when at 2x2 submatrices recursive calls end
    if len(matrix) == 2 and len(matrix[0]) == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return val

    # Section 3: define submatrix for focus column and call this function
    total = 0
    for fc in indices:  # for each focus column, find the submatrix ...
        As = copy_matrix(matrix)  # make a copy, and ...
        As = As[1:]  # ... remove the first row
        height = len(As)

        for i in range(height):  # for each remaining row of submatrix ...
            As[i] = As[i][0:fc] + As[i][fc+1:]  # zero focus column elements

        sign = (-1) ** (fc % 2)  # alternate signs for submatrix multiplier
        sub_det = determinant(As)  # pass submatrix recursively
        total += sign * matrix[0][fc] * sub_det
        # total all returns from recursion

    return total


def minor(matrix):
    """Minor matrix"""
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
    minor_matrix = []
    for i in range(lm):
        sub_matrix = []
        for j in range(lm):
            temp_matrix = [[matrix[m][n] for n in range(lm)
                            if (n != j and m != i)]
                           for m in range(lm)]
            # delete empty nodes
            temp_matrix = [m for m in temp_matrix if len(m) == lm - 1]

            sub_matrix.append(determinant(temp_matrix))
        minor_matrix.append(sub_matrix)
    return minor_matrix


def cofactor(matrix):
    """Cofactor matrix"""
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
    minor_matrix = minor(matrix)
    lmm = len(minor_matrix)
    cofactor_matrix = []
    flag = 0
    for i in range(lmm):
        temp_matrix = []
        for j in range(len(minor_matrix[0])):
            if flag:
                minor_matrix[i][j] = minor_matrix[i][j] * -1
                flag = 0
            else:
                flag = 1
            temp_matrix.append(minor_matrix[i][j])
        if i % 2 == 0:
            flag = 1
        cofactor_matrix.append(temp_matrix)
    return cofactor_matrix
