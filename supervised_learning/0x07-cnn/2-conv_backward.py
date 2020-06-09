#!/usr/bin/env python3
"""performs back propagation over a
 convolutional layer of a neural network"""

import numpy as np


def conv_backward(dZ, A_prev, W, b,
                  padding="same",
                  stride=(1, 1)):
    """performs back propagation over
    a convolutional layer of a
    neural network"""
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]
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
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = np.ceil(((sh*h_prev)-sh+kh-h_prev)/2)
        ph = int(ph)
        pw = np.ceil(((sw*w_prev)-sw+kw-w_prev)/2)
        pw = int(pw)
    A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)
    dW = np.zeros_like(W)
    dx = np.zeros_like(A_prev)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    tmp_W = W[:, :, :, l]
                    tmp_dz = dZ[i, j, k, l]
                    dx[i, j*sh:j*sh+kh, k*sw:k*sw+kw, :] += tmp_dz * tmp_W

                    tmp_A_prev = A_prev[i, j*sh:j*sh+kh, k*sw:k*sw+kw, :]
                    dW[:, :, :, l] += tmp_A_prev * tmp_dz

    dx = dx[:, ph:dx.shape[1]-ph, pw:dx.shape[2]-pw, :]

    return dx, dW, db
