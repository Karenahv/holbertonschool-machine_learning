#!/usr/bin/env python3
"""The Viretbi Algorithm"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Function that calculates the most likely sequence of hidden
        states for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains
            the index of the observation
        T is the number of observations
        Emission is a numpy.ndarray of shape
         (N, M) containing the emission probability
            of a specific observation given a hidden state
            Emission[i, j] is the probability
             of observing j given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations
        Transition is a 2D numpy.ndarray
         of shape (N, N) containing the transition
            probabilities
            Transition[i, j] is the
             probability of transitioning
              from the hidden state i to j
        Initial a numpy.ndarray of
         shape (N, 1) containing the probability of starting in a
            particular hidden state
    Returns: path, P, or None, None on failure
        path is the a list of length T
         containing the most likely sequence of hidden states
        P is the probability of
         obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None
    y = Observation
    A = Transition
    B = Emission
    Pi = Initial

    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi.T * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    P = np.amax(T1[:, y - 1], axis=0)
    P = np.amin(P)
    path = list(x)
    return path, P
