#!/usr/bin/env python3
""" P affinities of a data set"""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    calculates the symmetric P affinities of a data set:
        - X is a numpy.ndarray of shape (n, d) containing the dataset
            to be transformed by t-SNE
            - n is the number of data points
            - d is the number of dimensions in each point
        - perplexity is the perplexity that all Gaussian
            distributions should have
        - tol is the maximum tolerance allowed (inclusive) for the
            difference in Shannon entropy from perplexity for all
            Gaussian distributions
        Returns: P, a numpy.ndarray of shape (n, n) containing the
        symmetric P affinities
    """
    (n, d) = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        betamin = -np.inf;
        betamax = np.inf;
        # Array with distances between each point
        Di = np.append(D[i, :i], D[i, i + 1:])
        Hi, Pi = HP(Di, betas[i])
        diff = Hi - H
        # binary search
        while np.abs(diff) > tol:
            if diff > 0:
                betamin = betas[i].copy();
                if betamax == np.inf or betamax == -np.inf:
                    betas[i] = betas[i] * 2;
                else:
                    betas[i] = (betas[i] + betamax) / 2;
            else:
                betamax = betas[i].copy();
                if betamin == np.inf or betamin == -np.inf:
                    betas[i] = betas[i] / 2;
                else:
                    betas[i] = (betas[i] + betamin) / 2;

                # Recompute the values
            Hi, Pi = HP(Di, betas[i])
            diff = Hi - H
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi;
    # simmetric
    # http://www.jmlr.org/papers/volume9/vandermaaten08a
    return P

