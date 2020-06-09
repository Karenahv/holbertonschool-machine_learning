#!/usr/bin/env python3
"""performs forward propagation
 over a convolutional layer of a neural
 network"""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """performs forward propagation
    over a convolutional layer of a neural
    network"""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]
    ph = 0
    pw = 0
    sh = stride[0]
    sw = stride[1]
    h_final = int(((h_prev+2*ph-kh)/sh) + 1)
    w_final = int(((w_prev+2*pw-kw)/sw) + 1)

    if padding == 'same':
        if kh % 2 == 0:
            ph = int(((h_prev)*sh+kh-h_prev)/2)
            h_final = int(((h_prev+2*ph-kh)/sh))
        else:
            ph = int(((h_prev-1)*sh+kh-h_prev)/2)
            h_final = int(((h_prev+2*ph-kh)/sh)+1)
        if kw % 2 == 0:
            pw = int(((w_prev)*sw+kw-w_prev)/2)
            w_final = int(((w_prev+2*pw-kw)/sw))
        else:
            pw = int(((w_prev-1)*sw+kw-w_prev)/2)
            w_final = int(((w_prev+2*pw-kw)/sw)+1)
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    array_conv = np.zeros((m, h_final, w_final, c_new))
    img = np.arange(m)
    for j in range(h_final):
        for i in range(w_final):
            for k in range(c_new):
                tmp = A_prev[img, j*sh:(kh+(j*sh)), i*sw:(kw+(i*sw))]
                array_conv[img, j, i, k] = (np.sum(tmp *
                                                   W[:, :, :, k],
                                                   axis=(1, 2, 3)))
                array_conv[img, j, i, k] = (activation(array_conv[img, j,
                                                                  i, k] +
                                                       b[0, 0, 0, k]))
    return array_conv
