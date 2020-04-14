#!/usr/bin/env python3
"""function that adds two arrays"""


def add_arrays(arr1, arr2):
    """add two arrays"""
    if (len(arr1) == len(arr2)):
        result = []
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
    return None
