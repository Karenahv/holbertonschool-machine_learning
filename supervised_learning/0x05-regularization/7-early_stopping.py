#!/usr/bin/env python3
"""determinates if I should stop the gradient descent early"""


def early_stopping(cost, opt_cost, threshold,
                   patience, count):
    """determinates if I should stop the gradient descent early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        return True, count
    return False, count
