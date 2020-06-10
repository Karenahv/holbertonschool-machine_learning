#!/usr/bin/env python3
"""LeNet-5 architecture using tensorflow keras"""


import tensorflow.keras as K


def lenet5(x):
    """LeNet-5 architecture using tensorflow keras"""

    init = K.initializers.he_normal(seed=None)
    # Step 1: Convolution
    layer_convo1 = K.layers.Conv2D(filters=6, kernel_size=5,
                                   activation='relu',
                                   padding="same",
                                   kernel_initializer=init)(x)
    # Step 2: Max Pooling Layer
    layer_poolmax = K.layers.MaxPool2D(pool_size=(2, 2),
                                       strides=(2, 2))(layer_convo1)
    # step 3: Convolution
    layer_convo2 = K.layers.Conv2D(filters=16, kernel_size=5,
                                   activation='relu',
                                   padding="valid",
                                   kernel_initializer=init)(layer_poolmax)
    # step 4: Max pooling 2
    layer_poolmax2 = K.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=(2, 2))(layer_convo2)
    layer_flat = K.layers.Flatten()(layer_poolmax2)
    # step 5: Fully connected layer 120 nodes
    layer_fc = K.layers.Dense(units=120,
                              activation='relu',
                              kernel_initializer=init)(layer_flat)
    # step 6: Fully connected layer 84 nodes
    layer_fc2 = K.layers.Dense(units=84,
                               activation='relu',
                               kernel_initializer=init)(layer_fc)

    # step 7: Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=init)(layer_fc2)

    model = K.models.Model(inputs=x, outputs=output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
