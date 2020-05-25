#!/usr/bin/env python3
"""sensivity"""

import numpy as np


def sensitivity(confusion):
    """returns sensivity of each class"""
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TPR = TP/(TP+FN)
    return TPR
