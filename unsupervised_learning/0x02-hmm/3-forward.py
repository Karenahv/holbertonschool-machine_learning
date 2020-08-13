#!/usr/bin/env python3
"""Performs forward algorithm for a hmm"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """

    :param Observation:is a numpy.ndarray of shape (T,)
     that contains the index of the observation
    :param Emission:is a numpy.ndarray of shape (N, M)
     containing the emission probability of
      a specific observation given a hidden state
    :param Transition: is a 2D numpy.ndarray of shape
     (N, N) containing the transition probabilities
    :param Initial: a numpy.ndarray of shape (N, 1)
     containing the probability of starting in a particular hidden state
    :return: P, F, or None, None on failure
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    N1, N2 = Transition.shape
    N3 = Initial.shape[0]

    if ((len(Observation.shape)) != 1) or (type(Observation) != np.ndarray):
        return (None, None)
    if ((len(Emission.shape)) != 2) or (type(Emission) != np.ndarray):
        return (None, None)
    if ((len(Transition.shape)) != 2) or (N != N1) or (N != N2):
        return (None, None)
    if (N1 != N2) or (type(Transition) != np.ndarray):
        return (None, None)
    prob = np.ones((1, N1))
    if not (np.isclose((np.sum(Transition, axis=1)), prob)).all():
        return (None, None)
    if ((len(Initial.shape)) != 2) or (type(Initial) != np.ndarray):
        return (None, None)
    if (N != N3):
        return (None, None)

    F = np.zeros((N, T))

    # initialize all states from Pi
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    # for each t, for each state, sum(all prev-state * transition * ob)
    for t in range(1, len(Observation)):
        F[:, t] = (F[:, t - 1].dot(Transition[:, :])) * \
                  Emission[:, Observation[t]]

    P = np.sum(F[:, -1])

    return (P, F)
