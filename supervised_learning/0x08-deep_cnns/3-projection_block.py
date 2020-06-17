#!/usr/bin/env python3
"""builds an projection block as described in Deep
 Residual Learning for Image Recognition (2015)"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds an projection block as described in Deep
    Residual Learning for Image Recognition (2015)"""
    kernel_init = K.initializers.he_normal(seed=None)
    x = K.layers.Conv2D(filters[0],
                        (1, 1),
                        padding='same',
                        strides=s,
                        kernel_initializer=kernel_init)(A_prev)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters[1], (3, 3),
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters[2],
                        (1, 1),
                        padding='same',
                        kernel_initializer=kernel_init
                        )(x)
    x = K.layers.BatchNormalization()(x)
    x1 = K.layers.Conv2D(filters[2],
                         (1, 1),
                         padding='same',
                         strides=s,
                         kernel_initializer=kernel_init)(A_prev)
    norm_x1 = K.layers.BatchNormalization()(x1)
    adds = K.layers.add([x, norm_x1])
    output = K.layers.Activation('relu')(adds)

    return output
