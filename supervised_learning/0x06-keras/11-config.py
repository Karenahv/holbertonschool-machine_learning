#!/usr/bin/env python3
"""saves a model’s configuration"""


import tensorflow.keras as K


def save_config(network, filename):
    """ saves a  model's configuration"""
    json_string = network.to_json()
    with open(filename, "w") as f:
        f.write(json_string)
    return None


def load_config(filename):
    """ loads a model with specific configuration"""
    with open(filename, "r") as f:
        network = f.read()
    fresh_model = K.models.model_from_json(network)
    return fresh_model
