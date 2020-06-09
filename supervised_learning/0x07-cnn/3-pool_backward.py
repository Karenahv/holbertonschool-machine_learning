#!/usr/bin/env python3
"""performs back propagation over a pooling
 layer of a neural network"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape,
                  stride=(1, 1), mode='max'):
    """performs back propagation over
    a pooling layer of a neural network"""
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c = dA.shape[3]
    _, h_prev, w_prev, _ = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c):
                    pool = A_prev[i, j*sh:(kh+j*sh), k*sw:(kw+k*sw), l]
                    dA_val = dA[i, j, k, l]
                    if mode == 'max':
                        zero_mask = np.zeros(kernel_shape)
                        maxi = np.amax(pool)
                        np.place(zero_mask, pool == maxi, 1)
                        dA_prev[i, j*sh:(kh+j*sh),
                                k*sw:(kw+k*sw), l] += zero_mask * dA_val
                    if mode == 'avg':
                        avg = dA_val/kh/kw
                        one_mask = np.ones(kernel_shape)
                        dA_prev[i, j*sh:(kh+j*sh),
                                k*sw:(kw+k*sw), l] += one_mask * avg
    return dA_prev
