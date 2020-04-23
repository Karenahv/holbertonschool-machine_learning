#!/usr/bin/env python3
"""summation"""


def summation_i_squared(n):
    """summation i """
    i = 1
    sum = 0
    for i in range(n + 1):
        sum = sum + i**2
    return sum
