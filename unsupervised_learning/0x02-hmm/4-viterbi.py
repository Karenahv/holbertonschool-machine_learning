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

    V = np.zeros((N, T))
    B = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    # for each t, for each state, choose the
    # biggest from all prev-state * transition * ob,
    # remember the best prev
    for t in range(1,  T):
        for j in range(N):
            V[j, t], B[j, t] = max([(p, s)
                                    for s, p in enumerate(V[:, t - 1]
                                                          * Transition[:, j]
                                                          * Emission[j, Observation[t]])])
    P = np.amax(V, axis=0)
    P = np.amin(P)
    return V, P
