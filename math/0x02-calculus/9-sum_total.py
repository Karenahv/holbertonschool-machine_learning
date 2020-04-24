#!/usr/bin/env python3
"""summation"""


def summation_i_squared(n):
    """summation i """
    if n is None or n < 1:
        return None
    res = list(map(lambda x: x**2, list(range(1, n+1))))
    res = sum(res)
    return res
