#!/usr/bin/env python3
"""specificity"""

import numpy as np


def specificity(confusion):
    """returns specificity"""
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN/(TN+FP)
    return TNR
