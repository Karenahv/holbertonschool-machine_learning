#!/usr/bin/env python3
""" train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ train a model using keras"""
    callback = None
    if early_stopping and validation_data:
        callback = [K.callbacks.EarlyStopping(patience=patience)]
    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose,
                          validation_data=validation_data,
                          callbacks=callback)
    return history
