# -*- coding: utf-8 -*-

"""

The Ehrenfest state-space models used in the second simulation study.

"""


#### IMPORTS ####

import numpy as np
import xarray as xr

import particles
from particles import augmented_state_space_models as augssm
from particles.collectors import Moments
from particles import distributions as dists

from ctmc_modules.functions import (
    weighted_quantile,
    get_gamma_params_from_mean_var,
    lams_idx_to_gen_pos,
    get_default_rw_initial_config,
    compute_transition_prob_matrix,
    get_cat_dist,
)   



#### EHRENFEST FUNCTIONS ####


def run_particle_filter_bv(*,
        ehrenfest_ssm,
        data,
        N,
        k_series,
        qs = None,
        also_return_pf_boot: bool = False,
    ):
    
    """
    Runs the bootstrap particle filter and stores the results into an
    xarray.Dataset. For the Ehrenfest SSM that tracks the container
    volumes.
        
    If quantiles is None, calculate and store the quantiles 5%, 50%, and
    95% of the filtering ensembles in the Dataset.
    """
    
    if qs is None:
        qs = np.array([0.05, 0.5, 0.95]) # 90% interval & median
    
    
    #### Run the bootstrap particle filter ####
    
    fk_boot = augssm.AugmentedBootstrap(ssm=ehrenfest_ssm, data=data)
    pf_boot = particles.SMC(
        fk=fk_boot, N=N,
        resampling='stratified', 
        store_history=True, collect=[Moments()]
    )
    print("Beginning the bootstrap particle filter.")
    pf_boot.run()
    print("Bootstrap particle filter finished.")
    print()
    
    
    #### Store particles, weights, etc in an xarray.Dataset ####
    
    ds_boot = xr.Dataset({
        
        'X': xr.DataArray(
            np.stack([pf_boot.hist.X[k-1] for k in k_series]),
            dims=("k", "particle", "volume"),
            coords={
                "k": k_series,
                "volume": ["V_A", "V_B"],
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


def run_particle_smoother_bv(
        pf, M: int, quantiles=None
    ):
    """
    Runs the O(N_smooth) MCMC backward sampling particle smoother.
    Returns the smooth trajectories and quantiles. For the Ehrenfest
    SSM that tracks the container volumes.
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
    
    
    smooth_quantiles = xr.DataArray(
        smooth_quantiles,
        dims=("k", "volume", "quantile"),
        coords={
            "k": np.arange(smooth_quantiles.shape[0]),
            "volume": ["V_A", "V_B"],
            "quantile": quantiles
        },
        name="Smoothing quantiles"
    )
    
    return smooth_trajectories, smooth_quantiles



#### EHRENFEST SSM ####


class EhrenfestSSM_BV(augssm.AugmentedStateSpaceModel):
    r"""
        Ehrenfest state-space model - inferring the box volumes
        
        An extension of the continuous-time Markov chain state-space model
        for inferring the container volumes of an Ehrenfest model setup.
        
        INPUTS
        ------
        L: number of total gas molecules in the containers.
        J: number of random walkers.
        delta_t: real time between observations.
        C: parameter for scaling the transition variance.
        omega: ndarray of shape (2,), used in the calculation of the
            transition rates.
        mu0: ndarray of shape (2,), the means of the prior distributions of
            both containers' volumes.
        var0: ndarray of shape (2,), similar to above but the variance.
        y_init: initial configuration of RWs (set to None or don't pass 
            into initialisation if using the default).
        px_reg_term: a small constant added to beta in Option 2 in PX to help
            with numberical stability. Stops beta from being too close to 0.
            Note that the mean of a Gamma distribution is alpha / beta.
        px_verbose: if True and if t % px_verbose_int == 0, prints the value
            of t in PX.
        px_verbose_int: will print the verbose every amount of these steps.
    """
    
    def __init__(self, *,
            L: int,
            J: int,
            delta_t: float,
            C: float,
            omega: float,
            mu0,
            var0,
            y_init = None,
            reg_term: float = 1e-6,
            px_verbose: bool = False,
            px_verbose_int: int = 20
        ):
        
        assert (
            L >= 1 and J >= 1 and delta_t > 0 and C > 0 and
            reg_term > 0 and px_verbose_int > 0
        )
        
        assert (
            mu0.shape == (2,) and var0.shape == (2,) and
            omega.shape == (2,)
        )
        
        self.L = L
        self.n = self.L + 1 # Number of CTMC states
        self.delta_t = delta_t
        self.J = J
        self.C = C
        self.omega = omega
        
        self.mu0 = mu0
        self.var0 = var0
        self.a0, self.b0 = get_gamma_params_from_mean_var(
            self.mu0, self.var0
        )
        
        if y_init is None:
            self.y_init = get_default_rw_initial_config(self.n, self.J)
        else:
            self.y_init = y_init
        
        self.max_num_lams = self.n * (self.n - 1)
        
        self.all_gen_positions = [
            lams_idx_to_gen_pos(idx, self.n)
            for idx in range(self.max_num_lams)
        ]
        
        self.reg_term = reg_term
        
        self.px_verbose = px_verbose
        self.px_verbose_int = px_verbose_int
    
    
    def PX0(self):
        return dists.IndepProd(
            *[dists.Gamma(a, b) for a, b in zip(self.a0, self.b0)]
        )
    
    
    def PX(self, t, xp):
        
        if self.px_verbose and t % self.px_verbose_int == 0:
            print("t:", t)
        
        delta_t_times_C = self.delta_t * self.C
        
        # Note: transition variance == delta_t * C
        alpha = xp ** 2 / delta_t_times_C
        beta  = xp / delta_t_times_C + self.reg_term # Forces beta != 0
        
        return dists.IndepProd(*[
            dists.Gamma(alpha[:, i], beta[:, i])
            for i in range(alpha.shape[1])
        ])
    
    
    def PY(self, t, xp, x, datap=None):
        
        if datap is None:
            datap = self.y_init
        else:
            # (1, J) => (J,)
            datap = datap.reshape(-1)
        
        N = x.shape[0] # Number of PF particles
        
        nu_k = self.omega / x
        
        lams_k = np.stack([
            (p - 1) * nu_k[:, 0] if (p - 1 == q) else
            (self.L + 1 - p) * nu_k[:, 1] if (p + 1 == q) else
            np.full(N, 0)
            for p, q in self.all_gen_positions
        ], axis=1)
        
        P_mat = np.stack([
            compute_transition_prob_matrix(
                lams_k[i],
                self.n,
                self.delta_t
            )
            for i in range(N)
        ], axis=0)
        
        return dists.IndepProd(
            *[get_cat_dist(P_mat, y_i) for y_i in datap]
        )

