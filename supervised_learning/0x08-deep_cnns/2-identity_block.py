#!/usr/bin/env python3
"""builds an identity block as described in Deep
 Residual Learning for Image Recognition (2015)"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ builds an inception block as described in
    Going Deeper with Convolutions (2014)"""
    kernel_init = K.initializers.he_normal(seed=None)
    x = K.layers.Conv2D(filters[0],
                        (1, 1),
                        padding='same',
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
    x = K.layers.add([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
