#!/usr/bin/env python3
"""precision"""

import numpy as np


def precision(confusion):
    """returns precision"""
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    PPV = TP/(TP+FP)
    return PPV
