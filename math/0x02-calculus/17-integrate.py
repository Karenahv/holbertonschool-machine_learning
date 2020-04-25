#!/usr/bin/env python3
"""derivate polinomial"""


def poly_integral(poly, C=0):
    """poli derivate"""
    arr = [0]
    arr2 = [0]
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
        for item in poly:
            arr2.append(item)
        for i in range(1, len(arr2)):
            arr.append(arr2[i]/i)
        return arr
    else:
        return None
