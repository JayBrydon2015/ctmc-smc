# -*- coding: utf-8 -*-

"""

Functions and helper functions.

"""


#### IMPORTS ####

import numpy as np
import xarray as xr
import pandas as pd
from scipy.linalg import expm # Matrix exponential

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments
from particles import distributions as dists


#### FUNCTIONS ####


def get_k_series(K: int):
    """ Returns ndarray [1, 2, ..., K] of shape (K,). """
    return np.arange(1, K+1)


def get_time_series(delta_t, k_series):
    """ Time here refers to time since first observation at t_1. """
    return delta_t * (k_series - 1)


def get_gamma_params_from_mean_var(mean, var):
    """ Compute the Gamma distribution parameters, alpha
        and beta, from the mean and variance.
    """
    return mean ** 2 / var, mean / var


def gen_to_lams(gen):
    """ Convert generator to lams.
        
        Flattens an ndarray of shape (n, n) and removes the diagonal elements.
        
        lams: list of transition rates ordered as
        12, 13, ..., 1n, 21, 23, ..., 2n, ..., n1, ..., n(n-1).
    """
    n = gen.shape[0]
    lams = np.array([])
    for m in range(n):
      lams = np.append(lams, gen[m,0:m])
      lams = np.append(lams, gen[m,m+1:])
    return lams


def lams_to_gen(lams):
    """ Convert lams to generator.
        
        lams: list of transition rates ordered as
        12, 13, ..., 1n, 21, 23, ..., 2n, ..., n1, ..., n(n-1).
    """
    l = len(lams)
    n = int((1 + np.sqrt(1 + 4*l))/2)
    gen = np.zeros((n,n))
    for m in range(n):
      lams_m = lams[m*(n-1):(m+1)*(n-1)]
      gen[m, 0:m] = lams_m[0:m]
      gen[m, m+1:] = lams_m[m:]
      gen[m,m] = -np.sum(lams_m)
    return gen


def lams_idx_to_gen_pos(idx, n):
    """ Returns the (p, q) position of the rate given its index
        in the lams array and the number of CMTC states n.
    """
    i = idx // (n-1)
    j = idx %  (n-1)
    if i <= j:
        j += 1
    return i + 1, j + 1


def gen_pos_to_lams_idx(p: int, q: int, n: int):
    """ Given p, q, and n, return the index in the flattened array
        
        L_1_2, L_1_3, ..., L_1_n,
        L_2_1, L_2_3, ..., L_2_n,
        ...
        L_n_1, ..., L_n_(n-1).
    """
    
    if not (1 <= p <= n and 1 <= q <= n):
        raise ValueError(
            f"p and q must both be between 1 and {n}. "
            f"Got p={p}, q={q}."
        )
    
    if p == q:
        raise ValueError(
            "p and q must be different because self-transition "
            "rates are not included."
        )
    
    lam_idx = (p - 1) * (n - 1)
    if q < p:
        lam_idx += q - 1
    else:
        lam_idx += q - 2
    
    return lam_idx


def get_p_q_from_lam_col_name(lam_col: str):
    """ Given L_p_q, returns p and q as integers. """
    p, q = lam_col.split("_")[1:]
    return int(p), int(q)


def get_rates_of_interest_ehrenfest(M: int):
    """ Returns a list of (p, q) pairs indicating the rates of interest
        in an ehrenfest CTMC.
    """
    rates_of_interest = []
    for i in range(1, M+1):
        rates_of_interest.append((i, i+1))
        rates_of_interest.append((i+1, i))
    rates_of_interest.sort()
    return rates_of_interest


def get_default_rw_initial_config(n: int, J: int):
    """ The random walkers are initialised approximately uniformly around
        the continuous-time Markov chain.
    """
    return np.sort(
        np.array([i % n for i in range(J)])
    )


def get_rw_all_state_p_initial_config(*, J: int, p: int = 1, n: int = None):
    """ The random walkers are initialised to all start in state p. """
    if n is not None and not 1 <= p <= n:
        raise ValueError("p must be between 1 and n (inclusive).")
    return np.array([(p - 1) for _ in range(J)])


def compute_transition_prob_matrix(lams, n: int, delta_t: float):
    """ Computes the matrix exponential of delta_t * lams_to_gen(lams). """
    return expm(delta_t * lams_to_gen(lams))


def get_cat_dist(P_k, y_i):
    """ Returns the categorical distribution for a particular
        random walker (and for every particle).
    """
    return dists.Categorical(P_k[:, y_i])


def weighted_quantile(values, weights, quantiles):
    """ Calculates weighted quantiles using linear interpolation.
    
        The weighted empirical cumulative distribution function is constructed
        from the values and weights, and the requested quantiles are obtained
        by linearly interpolating its inverse.
    
        Inputs
        ------
        values: (N,)
            Values for which to calculate the weighted quantiles.
        weights: (N,)
            Corresponding non-negative weights.
        quantiles: array-like in [0, 1]
            Quantiles to calculate.
    """
    
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)


def run_particle_filter(*,
        ctmc_ssm,
        data,
        N,
        k_series,
        qs = None,
        also_return_pf_boot: bool = False,
    ):
    
    """
    Runs the bootstrap particle filter and stores the results
    into an xarray.Dataset.
        
    If quantiles is None, calculate and store the quantiles 5%, 50%, and
    95% of the filtering ensembles in the Dataset.
        
    Inputs
    ------
    ctmc_ssm: CTMC state-space model as a Python object.
    data: a list of length K+1 containing the data/observations.
    N: number of particles in the particle filter.
    k_series: [1, 2, ..., K-1, K], a list, NumPy 1D ndarray, etc.
    qs: NumPy ndarray of shape (Q,) containing Q quantiles to compute, each
        between 0 and 1. Set to None if using default (see below).
    also_return_pf_boot: True if also returning pf_boot; False otherwise.
    """
    
    if qs is None:
        qs = np.array([0.05, 0.5, 0.95])
    
    
    #### Run the bootstrap particle filter ####
    
    fk_boot = augssm.AugmentedBootstrap(ssm=ctmc_ssm, data=data)
    pf_boot = particles.SMC(
        fk=fk_boot, N=N,
        resampling='stratified', 
        store_history=True, collect=[Moments()]
    )
    print()
    print("Beginning the bootstrap particle filter.")
    pf_boot.run()
    print("Bootstrap particle filter finished.")
    print()
    print()
    
    
    #### Store lambda particles and weights in an xarray.Dataset ####
    
    if ctmc_ssm.rates_of_interest is None:
        lams_gen_positions = [
            lams_idx_to_gen_pos(i, ctmc_ssm.n)
            for i in range(ctmc_ssm.num_lams)
        ]
    else:
        lams_gen_positions = ctmc_ssm.rates_of_interest
    
    lam_names = [f"L_{p}_{q}" for p, q in lams_gen_positions]
    
    ds_boot = xr.Dataset({
        
        'X': xr.DataArray(
            np.stack([pf_boot.hist.X[k-1] for k in k_series]),
            dims=("k", "particle", "lam"),
            coords={
                "k": k_series,
                "lam": lam_names,
            },
            name="Bootstrap PF Particles"
        ),
        
        'W': xr.DataArray(
            np.stack([pf_boot.hist.wgts[k-1].W for k in k_series]),
            dims=("k", "weight"),
            coords={
                "k": k_series
            },
            name="Bootstrap PF Weights"
        ),
        
        'ESS': xr.DataArray(
            pf_boot.summaries.ESSs,
            dims="k",
            coords={
                "k": k_series
            },
            name="Bootstrap PF ESSs"
        )
    })
    
    
    #### Calculate quantiles and add into ds_boot ####
    
    ds_boot["X_quantiles"] = xr.apply_ufunc(
        weighted_quantile,
        ds_boot["X"],
        ds_boot["W"],
        input_core_dims=[["particle"], ["weight"]],
        output_core_dims=[["quantile"]],
        vectorize=True,
        kwargs={"quantiles": qs},
        dask="parallelized",
        output_dtypes=[float],
    ).assign_coords(quantile=qs).rename("Bootstrap PF Quantiles")
    
    return ds_boot, pf_boot if also_return_pf_boot else ds_boot


def run_particle_smoother(
        pf, M: int, k_series, quantiles=None, lam_coords=None
    ):
    """
    Runs the O(M) MCMC backward sampling particle smoother.
    Returns the smooth trajectories and quantiles.
    """
    
    if quantiles is None:
        quantiles = np.array([0.05, 0.50, 0.95])
    
    print("Beginning particle smoother.")
    smooth_trajectories = pf.hist.backward_sampling_mcmc(M)
    print("Particle smoother finished.")
    print()
    
    smooth_quantiles = np.array([
        np.quantile(x, quantiles, axis=0)
        for x in smooth_trajectories
    ]).transpose(0, 2, 1)
    
    # Return as xr.DataArray if lam_coords provided
    if lam_coords is not None:
        smooth_quantiles = xr.DataArray(
            smooth_quantiles,
            dims=("k", "lam", "quantile"),
            coords={
                "k": k_series,
                "lam": lam_coords,
                "quantile": quantiles
            },
            name="Smoothing quantiles"
        )
    
    return smooth_trajectories, smooth_quantiles


def make_true_rates_dataframe(*,
        true_rates: list,
        n: int,
        lams_gen_positions: list = None
    ):
    
    """
    Creates a pandas data frame storing the true transition rates. The columns
    of the data frame refer to each of the transition rates and column 'L_p_q'
    is for the transition rate from state p to state q.
    
    Inputs
    ------
    true_rates: a list containing the true transition rates at each time.
    n: number of states in the CTMC.
    lams_gen_positions: a list containing (p, q) pairs indicating particular
        transition rates of interest. If None, is computed automatically
        (all possible n*(n-1) rates are of interest).
    """
    
    if lams_gen_positions is None:
        num_rates = n * (n - 1)
        lams_gen_positions = [
            lams_idx_to_gen_pos(i, n)
            for i in range(num_rates)
        ]
    
    lams_gen_positions.sort()
    
    true_rates_df = pd.DataFrame(
        np.stack([true_rates_k.reshape(-1) for true_rates_k in true_rates]),
        columns=[f"L_{p}_{q}" for p, q in lams_gen_positions]
    )
    
    return true_rates_df


