#!/usr/bin/env python3
"""derivate polinomial"""


def poly_derivative(poly):
    """poli derivate"""
    arr = []
    if type(poly) != list:
        return None
    if poly is None or len(poly) == 0 or poly == []:
        return None
    if(len(poly) == 1):
        return [0]
    for elem in poly:
        if(type(elem) is not int and type(elem) is not float):
            return None
    if type(poly) is list:
        for i in range(1, len(poly)):
            arr.append(i*poly[i])
        return arr
    else:
        return None
