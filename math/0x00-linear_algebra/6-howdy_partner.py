#!/usr/bin/env python3
"""concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """concatenates two arrays """
    result = []
    for item in arr1:
        result.append(item)
    for item in arr2:
        result.append(item)
    return result
