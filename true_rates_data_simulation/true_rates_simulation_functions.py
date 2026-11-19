# -*- coding: utf-8 -*-

"""

Functions to simulate true rates over time.

"""


#### IMPORTS ####

import numpy as np
from scipy.special import expit

from ctmc_modules.functions import gen_to_lams


#### CONSTANT RATES ####


def simulate_constant_rates(*, mu0, K):
    """ Simulate true_states and data where rates remain constant. """
    return [ mu0.reshape(1, -1) for _ in range(K) ]


#### SIGMOID EVOLUTION ####


def sigmoid(x, a, b, m, s):
    """ Sigmoid function. Note: result = a when x = 0. """
    # expit(u) == 1 / (1 + np.exp(-u))
    result = a + (b - a) * expit(s * (x - m))
    result -= a + (b - a) * expit(-s * m)
    result += a
    return result


def simulate_sigmoid_growth(*, K, mu0=[1, 1], max_growth=[7, 7], s=15):
    """ Simulate the true transition rates which undergo sigmoid growth. """
    L = len(mu0)
    return [
        np.array([
            sigmoid(k, mu0[i], mu0[i] + max_growth[i], K/2, s/K)
            for i in range(L)
        ]).reshape(1, -1)
        for k in range(K)
    ]


#### PAPER SS1: SIGMOID & ABRUPT CHANGE COMBINED ####


def sudden_jump_function(x, a, b, m):
    """Piecewise function."""
    return a if x < m else b


def simulate_sigmoid_and_abrupt_rates(*,
        K,
        a1=2, b1=4, s=15, # Sigmoid rate L_1_2
        a2=1, b2=20 # Abrupt change rate L_2_1
    ):
    return [
        np.array([
            sigmoid(k, a1, b1, K/2, s/K),
            sudden_jump_function(k, a2, b2, K/2)
        ]).reshape(1, -1)
        for k in range(K)
    ]


#### PAPER SS2: EHRENFEST MODEL ####


def simulate_ehrenfest_rates(
        get_a_to_b_rate, get_b_to_a_rate, M: int, K: int
    ):
    
    n = M + 1
    true_rates = []
    
    for k in range(K):
        
        A_k = np.zeros((n, n))
        
        for i in range(n): # i = 0, 1, ..., M
            if i >= 1:
                A_k[i, i-1] = i * get_a_to_b_rate(k, K)
            if i < M:
                A_k[i, i+1] = (M-i) * get_b_to_a_rate(k, K)
        
        true_rates.append(
            gen_to_lams(A_k).reshape(1, -1)
        )
    
    return true_rates


def simulate_ehrenfest_rates_v2(
        get_a_to_b_rate, get_b_to_a_rate, M: int, K: int, omega
    ):
    
    n = M + 1
    true_rates = []
    
    for k in range(K):
        
        A_k = np.zeros((n, n))
        
        for i in range(n): # i = 0, 1, ..., M
            if i >= 1:
                A_k[i, i-1] = i * get_a_to_b_rate(k, K, omega[0])
            if i < M:
                A_k[i, i+1] = (M-i) * get_b_to_a_rate(k, K, omega[1])
        
        true_rates.append(
            gen_to_lams(A_k).reshape(1, -1)
        )
    
    return true_rates

