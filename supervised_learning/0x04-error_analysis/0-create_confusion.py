#!/usr/bin/env python3
"""creates confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates confusion matrix"""
    confusion = np.matmul(labels.T, logits)
    return confusion
