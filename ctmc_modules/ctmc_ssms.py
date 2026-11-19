# -*- coding: utf-8 -*-

"""

CTMC state-space models (SSMs) as Python objects (classes).

"""


#### IMPORTS ####

import numpy as np

from particles import augmented_state_space_models as augssm
from particles import distributions as dists

from ctmc_modules.functions import (
    get_gamma_params_from_mean_var,
    gen_to_lams,
    gen_pos_to_lams_idx,
    lams_idx_to_gen_pos,
    get_default_rw_initial_config,
    compute_transition_prob_matrix,
    get_cat_dist
)



## CTMC SSM CLASS ##


class CTMC(augssm.AugmentedStateSpaceModel):
    r"""
        Augmented SSM for CTMC generator paths

        ----- Parameters -----
        n: number of states in CTMC.
        J: number of random walkers.
        delta_t: real time between observations.
        C: transition variance multiplier.
        rates_of_interest: a set of (p, q) pairs that indicate the transition
          rates we wish to infer/track. p, q = 1, ..., n and p != q. The other
          transition rates we assume we already know, and can get their values
          via get_true_lam_at_k (see below).
        get_true_lam_at_k: a function that takes as input *(p, q, k) and
          returns the true value of the transition rate from states p to q
          at time-step k. If None (default), assumes rates not of interest
          are just 0.
        mu0: Gamma dist mean parameters for PX0 in (n, n) ndarray.
        var0: Gamma dist variance parameters for PX0 in (n, n) ndarray.
        y_init: initial configuration of RWs y_0 (set to None or don't pass 
          into initialisation if using the default). ndarray of shape (J,).
        reg_term: a small constant added to beta in PX to ensure numerical
          stability. Stops beta from being too close to 0. Note that the mean
          of a Gamma distribution is alpha / beta.
        k_verbose: if True and if t % px_verbose_int == 0, prints the value
          of k in PY.
        k_verbose_interval: interval between k-verbose prints.
        
        ----- Notes -----
        - Tracks the list of rates/lams rather than generator A directly.
        - lams: ndarray of shape (n*(n-1),). I.e. rates in a 1D array, ordered
          as 12, ..., 1n, 21, 23, ..., 2n, ..., n1, ..., n(n-2), n(n-1).
        - Default y_init (y_0) corresponds with random walkers spread as evenly
          as possible around CTMC.
        - y_k (data[k]): ndarray of shape (1, J), by convention (see PY).
        - CTMC states are 0 to n-1 in the code itself, however. This is
          because dists.Categorical returns values between 0 and n-1, and also
          these are used to index ndarrays (which assume 0-based indexing).
    """
    
    def __init__(self, *,
            n: int,
            J: int,
            delta_t: float,
            C: float,
            rates_of_interest: set = None,
            get_true_lam_at_k = None,
            mu0,
            var0,
            y_init = None,
            reg_term: float = 1e-6,
            k_verbose: bool = False,
            k_verbose_interval: int = 20
        ):
        
        assert n >= 2 and J >= 1 and delta_t > 0
        self.n = n
        self.max_num_lams = self.n * (self.n - 1)
        self.J = J
        self.delta_t = delta_t
        
        if C <= 0:
            raise ValueError(
                f"C must be positive. Currently, its value is {C}."
            )
        self.C = C
        
        if rates_of_interest is not None:
            self.rates_of_interest = sorted(list(rates_of_interest))
            assert all(
                1 <= p <= self.n and 1 <= q <= self.n and p != q
                for p, q in self.rates_of_interest
            )
            self.num_lams = len(rates_of_interest)
            self.lam_indices_of_interest = [
                gen_pos_to_lams_idx(p, q, self.n)
                for p, q in self.rates_of_interest
            ]
            self.lam_indices_of_interest_set = set(
                self.lam_indices_of_interest
            )
            self.lam_idx_to_j = {
                lam_idx: j
                for j, lam_idx in enumerate(self.lam_indices_of_interest)
            }
        else:
            self.rates_of_interest = None
            self.num_lams = self.max_num_lams
            self.lam_indices_of_interest = None
            self.lam_indices_of_interest_set = None
            self.lam_idx_to_j = None
        
        self.get_true_lam_at_k = get_true_lam_at_k
        if self.get_true_lam_at_k is None:
            self.get_true_lam_at_k = lambda p, q, t: 0
        
        ## Store and reshape mu0 and var0, if required ##
        self.mu0 = mu0
        self.var0 = var0
        if len(self.mu0.shape) > 1: # If currently shape (n, n)
            self.mu0 = gen_to_lams(self.mu0)
        if len(self.var0.shape) > 1: # If currently shape (n, n)
            self.var0 = gen_to_lams(self.var0)
        
        ## Extract elements of interest in mu0 and var0, if applicable ##
        if (
                self.rates_of_interest is not None and
                self.mu0.shape[0] == self.n * (self.n - 1)
            ):
            self.mu0 = self.mu0[self.lam_indices_of_interest]
        elif self.rates_of_interest is not None:
            assert self.mu0.shape[0] == self.num_lams
        
        if (
                self.rates_of_interest is not None and
                self.var0.shape[0] == self.n * (self.n - 1)
            ):
            self.var0 = self.var0[self.lam_indices_of_interest]
        elif self.rates_of_interest is not None:
            assert self.var0.shape[0] == self.num_lams
        
        ## Compute a0 and b0 based on mu0 and var0 ##
        self.a0, self.b0 = get_gamma_params_from_mean_var(mu0, var0)
        assert (
            self.a0.shape[0] == self.num_lams and
            self.a0.shape    == self.b0.shape
        )
        
        ## Get default y_init if not specified ##
        if y_init is None:
            self.y_init = get_default_rw_initial_config(self.n, self.J)
        else:
            self.y_init = y_init
        
        self.reg_term = reg_term
        
        self.k_verbose = k_verbose
        self.k_verbose_interval = k_verbose_interval
    
    
    def PX0(self): # A_1
        return dists.IndepProd(
            *[dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        )
    
    
    def PX(self, t, xp): # A_k | A_{k-1}, k >= 2
        
        delta_t_times_C = self.delta_t * self.C
        
        alpha = xp ** 2 / delta_t_times_C
        beta  = (
            xp / delta_t_times_C
            + self.reg_term # Forces beta != 0
        )
        
        lams_dists = [
            dists.Gamma(alpha[:, l], beta[:, l])
            for l in range(alpha.shape[1])
        ]
        
        return dists.IndepProd(*lams_dists)
    
    
    def PY(self, t, xp, x, datap=None):
        
        if self.k_verbose and t % self.k_verbose_interval == 0:
            print("k:", t+1)
        
        if datap is None: # t == 0
            datap = self.y_init
        else: # t >= 1, datap == data[t-1]
            # datap.shape == (1, J) => datap.shape == (J,)
            datap = datap.reshape(-1)
        
        ## If required, remake x in order to create P_mat ##
        if self.rates_of_interest is not None:
            N = x.shape[0] # Number of particles
            x = np.stack([
                x[:, self.lam_idx_to_j[i]]
                if i in self.lam_indices_of_interest_set
                else np.full(
                    N,
                    self.get_true_lam_at_k(
                        *lams_idx_to_gen_pos(i, self.n), t
                    )
                )
                for i in range(self.max_num_lams)
            ], axis=1)
        
        P_k = np.stack(
            [
                compute_transition_prob_matrix(
                    x[i],
                    self.n,
                    self.delta_t
                )
                for i in range(x.shape[0])
            ],
            axis=0
        )
        
        return dists.IndepProd(
            *[get_cat_dist(P_k, y_i) for y_i in datap]
        )



