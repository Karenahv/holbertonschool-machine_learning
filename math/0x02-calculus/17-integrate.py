#!/usr/bin/env python3
"""poli integral"""


def poly_integral(poly, C=0):
    """poli integral"""
    arr = [C]
    arr2 = [0]
    if type(poly) != list:
        return None
    if poly is None or len(poly) == 0 or poly == []:
        return None
    if(type(C) is not int):
        return None
    if(poly == [0]):
        return [C]
    for elem in poly:
        if(type(elem) is not int and type(elem) is not float):
            return None
    if type(poly) is list:
        for item in poly:
            arr2.append(item)
        for i in range(1, len(arr2)):
            aux = arr2[i]/i
            if (aux).is_integer():
                arr.append(int(arr2[i]/i))
            else:
                arr.append(arr2[i]/i)
        return arr
    else:
        return None
