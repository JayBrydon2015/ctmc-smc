# -*- coding: utf-8 -*-

"""

Functions for generating simple plots.

"""


#### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

from plotting_modules.plotting_helper_functions import (
    get_latex_rate_symbol
)

from ctmc_modules.functions import get_p_q_from_lam_col_name


#### SIMPLE PLOTTING FUNCTIONS ####


def plot_true_rates(true_rates_df, time_series):
    """ Generates a simple plot of the true transition rates. """
    for col in true_rates_df.columns:
        plt.plot(
            time_series, true_rates_df[col],
            label=get_latex_rate_symbol(
                *get_p_q_from_lam_col_name(col)
            )[0]
        )
    plt.xlabel("time")
    plt.ylabel("Value")
    plt.title("True rates over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_data(*, data: list, time_series, n: int, J: int, J_max: int = 10):
    """ Generates a simple plot of the data (if J not too large). """
    if J <= J_max:
        
        # Add 1 because the data is originally between 0 and n-1
        data_plot = np.vstack(data) + 1
        
        fig, axes = plt.subplots(
            nrows=J,
            ncols=1,
            sharex=True,
            figsize=(8, 0.4 * n * J)
        )
        fig.suptitle("Random walker states over time", fontsize=14)
        
        # Ensure axes is always iterable (important if J == 1)
        if J == 1:
            axes = [axes]
        
        for j in range(J):
            axes[j].plot(time_series, data_plot[:, j])
            # axes[j].set_ylabel(f"RW {j+1}")
            axes[j].grid(True)
        
        axes[-1].set_xlabel("time")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Too many random walkers to plot: {J} random walkers.")
        print()


def plot_ess(ess):
    """ Plots filtering ESS at each time. """
    plt.plot(ess)
    plt.xlabel("time")
    plt.ylabel("ESS")
    plt.title("Filtering ESS")
    plt.tight_layout()
    plt.show()


def plot_central_90_bandplots(ds_filt, true_rates_df, time_series):
    """ Plots the central 90% band plots of the (filtering) ensembles. """
    for lam in true_rates_df.columns:
        
        p, q = get_p_q_from_lam_col_name(lam)
        latex_rate_symbol, _ = get_latex_rate_symbol(p, q)
        
        median = ds_filt["X_quantiles"].sel(lam=lam, quantile=0.5)
        lq = ds_filt["X_quantiles"].sel(lam=lam, quantile=0.05)
        uq = ds_filt["X_quantiles"].sel(lam=lam, quantile=0.95)
        
        plt.plot(
            time_series, true_rates_df[lam].values,
            label=f"True {latex_rate_symbol}",
            color='red', alpha=0.7
        )
        
        plt.plot(
            time_series, median, color="green",
                label="PF median", alpha=0.7
        )
        
        plt.fill_between(
            time_series, y1=lq, y2=uq, color="green", alpha=0.3
        )
        
        plt.xlabel("time")
        plt.ylabel(f"Value of {lam}")
        
        plt.title(
            f"Boot PF central 90% band plot: {latex_rate_symbol}"
        )
        
        plt.legend()
        
        plt.show()


def plot_central_90_bandplots_fs(*,
        filt_quantiles, smooth_quantiles, true_rates_df, time_series
    ):
    """
    Plots the central 90% band plots of the filtering and smoothing 5%,
    50% (median), and 95% quantiles.
    """
    for lam_idx, lam in enumerate(true_rates_df.columns):
        
        p, q = get_p_q_from_lam_col_name(lam)
        latex_rate_symbol, _ = get_latex_rate_symbol(p, q)
        
        # Filtering quantiles
        filter_median = filt_quantiles.sel(lam=lam, quantile=0.5)
        filter_lq = filt_quantiles.sel(lam=lam, quantile=0.05)
        filter_uq = filt_quantiles.sel(lam=lam, quantile=0.95)
        
        # Smoothing quantiles
        smooth_lq = smooth_quantiles[:, 0, lam_idx]
        smooth_median = smooth_quantiles[:, 1, lam_idx]
        smooth_uq = smooth_quantiles[:, 2, lam_idx]
        
        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # --------------------
        # Filtering
        # --------------------
        axes[0].plot(
            time_series,
            true_rates_df[lam].values,
            label=f"True {latex_rate_symbol}",
            color="red",
            alpha=0.7
        )
        
        axes[0].plot(
            time_series,
            filter_median,
            color="green",
            label="PF median",
            alpha=0.7
        )
        
        axes[0].fill_between(
            time_series,
            y1=filter_lq,
            y2=filter_uq,
            color="green",
            alpha=0.3
        )
        
        axes[0].set_xlabel("time")
        # axes[0].set_ylabel(f"Value of {lam}")
        axes[0].set_title(f"Filtering: {latex_rate_symbol}")
        axes[0].legend()
        
        # --------------------
        # Smoothing
        # --------------------
        axes[1].plot(
            time_series,
            true_rates_df[lam].values,
            label=f"True {latex_rate_symbol}",
            color="red",
            alpha=0.7
        )
        
        axes[1].plot(
            time_series,
            smooth_median,
            color="blue",
            label="PS median",
            alpha=0.7
        )
        
        axes[1].fill_between(
            time_series,
            y1=smooth_lq,
            y2=smooth_uq,
            color="blue",
            alpha=0.3
        )
        
        axes[1].set_xlabel("time")
        # axes[1].set_ylabel(f"Value of {lam}")
        axes[1].set_title(f"Smoothing: {latex_rate_symbol}")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


