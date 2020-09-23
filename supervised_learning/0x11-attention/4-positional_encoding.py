#!usr/bin/env/python3
"""positional encoding"""

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    """
    :param pos:
    :param i:
    :param d_model:
    :return:
    """
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """
    :param max_seq_len: is an integer representing
    the maximum sequence length
    :param dm:is the model depth
    :return: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors
    You can use import numpy as np
    """
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)

    # sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding.reshape(max_seq_len, dm)
