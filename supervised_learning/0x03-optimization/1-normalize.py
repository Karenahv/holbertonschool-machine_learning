#!/usr/bin/env python3
"""standardizes a matrix"""

import numpy as np


def normalize(X, m, s):
    """standardizes a matrix"""
    normalize_matrix = (X - m) / s
    return normalize_matrix
