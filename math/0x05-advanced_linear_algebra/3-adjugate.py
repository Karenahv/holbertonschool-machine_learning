#!/usr/bin/env python3
"""matriz adjugate"""

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
    # Section 1: Establish n parameter and copy A
    AM = copy_matrix(matrix)

    # Section 2: Row manipulate A into an upper triangle matrix
    for fd in range(lm):  # fd stands for focus diagonal
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18  # Cheating by adding zero + ~zero
        for i in range(fd + 1, lm):  # skip row with fd in it.
            crScaler = AM[i][fd] / AM[fd][fd]  # cr stands for "current row".
            for j in range(lm):
                # cr - crScaler * fdRow, one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]

    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(lm):
        product *= AM[i][i]  # ... product of diagonals is determinant

    return int(product)


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
