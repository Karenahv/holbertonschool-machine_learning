#!/usr/bin/env python3
"""derivate polinomial"""


def poly_derivative(poly):
    """poli derivate"""
    arr = []
    if poly is None:
        return None
    if(len(poly) == 1):
        return [0]
    for i in range(1, len(poly)):
        arr.append(i*poly[i])
    return arr
