#!/usr/bin/env python3
"""performs forward propagation
over a pooling layer of a neural network"""

import numpy as np


def pool_forward(A_prev, kernel_shape,
                 stride=(1, 1), mode='max'):
    """performs forward propagation over
    a pooling layer of a neural network"""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    h_final = int(((h_prev-kh)/sh) + 1)
    w_final = int(((w_prev-kw)/sw) + 1)

    array = np.zeros((m, h_final, w_final, c_prev))
    img = np.arange(m)
    for j in range(h_final):
        for i in range(w_final):
            if mode == 'max':
                array[img, j, i] = (np.max(A_prev[img,
                                                  j*sh:(kh+(j*sh)),
                                                  i*sw:(kw+(i*sw))],
                                           axis=(1, 2)))
            if mode == 'avg':
                array[img, j, i] = (np.mean(A_prev[img,
                                                   j*sh:(kh+(j*sh)),
                                                   i*sw:(kw+(i*sw))],
                                            axis=(1, 2)))
    return array
