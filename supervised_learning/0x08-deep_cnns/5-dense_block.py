#!/usr/bin/env python3
"""builds a dense block as
 described in Densely Connected
 Convolutional Networks"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block as
    described in Densely Connected
    Convolutional Networks"""

    kernel_init = K.initializers.he_normal(seed=None)
    for i in range(layers):
        x = K.layers.BatchNormalization()(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(filters=4*growth_rate,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=kernel_init)(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=3,
                            padding='same',
                            kernel_initializer=kernel_init)(x)
        X = K.layers.concatenate([X, x])
        nb_filters += growth_rate
    return x, nb_filters
