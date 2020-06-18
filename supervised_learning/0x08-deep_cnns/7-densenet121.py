#!/usr/bin/env python3
""" hat builds the DenseNet-121 architecture
as described in Densely Connected
Convolutional Networks"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """hat builds the DenseNet-121 architecture as
    described in Densely Connected
    Convolutional Networks:"""
    kernel_init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=2*growth_rate,
                        kernel_size=7,
                        padding='same',
                        strides=2,
                        kernel_initializer=kernel_init)(x)

    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)

    x, nb_filters = dense_block(x, 2*growth_rate, growth_rate, 6)
    for layer in layers:
        x, nb_filters = transition_layer(x,
                                         nb_filters,
                                         compression)
        x, nb_filters = dense_block(x,
                                    nb_filters,
                                    growth_rate,
                                    layer)
    x = K.layers.AveragePooling2D(pool_size=7,
                                  strides=None,
                                  padding='same')(x)

    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_regularizer=K.regularizers.l2())(x)

    model = K.models.Model(inputs=X, outputs=output)

    return model
