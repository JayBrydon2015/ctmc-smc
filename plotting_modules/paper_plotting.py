# -*- coding: utf-8 -*-

"""

Plot generation for the paper.

"""


#### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

from plotting_modules.plotting_config import (
    TEXTWIDTH_IN, FONT_SIZE, TICK_FONT_SIZE
)

from plotting_modules.plotting_helper_functions import (
    get_latex_rate_symbol, apply_font_sizes
)

from ctmc_modules.functions import get_p_q_from_lam_col_name



#### Simulation Study #1 Plotting Functions ####


def plot_true_rates_ss1(true_rates_df, time_series):
    """Plot true rates of the first simulation study."""
    
    lw=2
    alpha=1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2))
    
    ## Left plot: L_1_2 ##
    
    left_col = true_rates_df.columns[0]
    
    p, q = get_p_q_from_lam_col_name(left_col)
    
    cur_color = "royalblue"
    
    ax1.plot(
        time_series,
        true_rates_df[left_col],
        label=get_latex_rate_symbol(
            p, q
        )[0],
        color=cur_color,
        lw=lw,
        alpha=alpha
    )
    
    ax1.set_xlabel("time")
    ax1.set_ylim(0, 5.5)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    # ax1.set_ylabel("Rate")
    ax1.legend()
    
    ## Right plot: L_2_1 ##
    
    right_col = true_rates_df.columns[1]
    
    p, q = [int(i) for i in right_col.split("_")[1:]]
    
    cur_color = "orange"
    
    ax2.plot(
        time_series,
        true_rates_df[right_col],
        label=get_latex_rate_symbol(
            p, q
        )[0],
        color=cur_color,
        lw=lw,
        alpha=alpha
    )
    
    ax2.set_xlabel("time")
    ax2.set_ylim(0, 5.5)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.legend()
    
    ## Adjust and show overall plot ##
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    # plt.tight_layout()
    plt.show()



#### Simulation Study #2 (Ehrenfest Model) Plotting Functions ####


def plot_volumes_and_true_rates(
        *, true_rates_df, k_series, time_series, K, V_A, V_B
    ):
    """Plot theoretical rates alongside the true rates."""
    
    lw = 2
    alpha = 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2))

    # Left: theoretical rates
    V_A_points = V_A(k_series, K)
    V_B_points = V_B(k_series, K)

    ax1.plot(
        time_series,
        V_A_points,
        label="Box A",
        color="orange",
        lw=lw,
        ls="--",
        alpha=alpha
    )
    ax1.plot(
        time_series,
        V_B_points,
        label="Box B",
        color="royalblue",
        lw=lw,
        ls=":",
        alpha=alpha
    )

    ax1.set_xlabel("time")
    ax1.set_ylabel("Volume")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Box volumes")
    ax1.legend()

    # Right: true rates
    for col in true_rates_df.columns:
        
        p, q = [int(i) for i in col.split("_")[1:]]
        
        if p > q:
            cur_color = "orange"
            ls="--"
        else:
            cur_color = "royalblue"
            ls=":"
        
        ax2.plot(
            time_series,
            true_rates_df[col],
            label=get_latex_rate_symbol(
                p, q
            )[0],
            color=cur_color,
            lw=lw,
            ls=ls,
            alpha=alpha
        )
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("Rate")
    ax2.set_title("True transition rates")
    # ax2.legend()
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    # plt.tight_layout()
    plt.show()


def plot_central_90_bandplots_stacked_fs_bv(*,
        filt_quantiles, smooth_quantiles, true_volumes, time_series,
        wspace=0.15
    ):
    """
    Plots the central 90% band plots of the filtering and smoothing 5%,
    50% (median), and 95% quantiles. Used for the box volumes.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4))
    
    true_volumes = np.vstack(true_volumes)
    
    for i in range(2):
        
        if i == 0:
            vol = "V_A"
            vol_title = r"$V^{(A)}(t)$"
        else:
            vol = "V_B"
            vol_title = r"$V^{(B)}(t)$"
        
        # Filtering quantiles
        filter_lq =     filt_quantiles.sel(volume=vol, quantile=0.05)
        filter_median = filt_quantiles.sel(volume=vol, quantile=0.5)
        filter_uq =     filt_quantiles.sel(volume=vol, quantile=0.95)
        
        # Smoothing quantiles
        smooth_lq =     smooth_quantiles.sel(volume=vol, quantile=0.05)
        smooth_median = smooth_quantiles.sel(volume=vol, quantile=0.5)
        smooth_uq =     smooth_quantiles.sel(volume=vol, quantile=0.95)
        
        # --------------------
        # Filtering
        # --------------------
        axes[i, 0].plot(
            time_series,
            true_volumes[:, i],
            label=f"True {vol}",
            color="red",
            alpha=0.7
        )
        
        axes[i, 0].plot(
            time_series,
            filter_median,
            color="green",
            label="PF median",
            alpha=0.7
        )
        
        axes[i, 0].fill_between(
            time_series,
            y1=filter_lq,
            y2=filter_uq,
            color="green",
            alpha=0.3
        )
        
        axes[i, 0].set_title(f"Filtering: {vol_title}")
        # axes[i, 0].legend()
        
        # --------------------
        # Smoothing
        # --------------------
        axes[i, 1].plot(
            time_series,
            true_volumes[:, i],
            label=f"True {vol}",
            color="red",
            alpha=0.7
        )
        
        axes[i, 1].plot(
            time_series,
            smooth_median,
            color="blue",
            label="PS median",
            alpha=0.7
        )
        
        axes[i, 1].fill_between(
            time_series,
            y1=smooth_lq,
            y2=smooth_uq,
            color="blue",
            alpha=0.3
        )
        
        axes[i, 1].set_title(f"Smoothing: {vol_title}")
        # axes[i, 1].legend()
    
    for ax in axes[-1]:
        ax.set_xlabel("time")
    
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate")
    
    for ax in axes.flatten():
        ax.set_ylim(-0.1, 1.7)
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    # plt.tight_layout()
    plt.subplots_adjust(
        # left=0.1,    # Left margin
        # right=0.9,   # Right margin
        # top=0.9,     # Top margin
        # bottom=0.1,  # Bottom margin
        wspace=wspace, # Horizontal spacing between subplots (width fraction)
        hspace=0.4     # Vertical spacing between subplots (height fraction)
    )
    plt.show()



#### General plotting functions ####


def plot_central_90_bandplots_stacked_fs(*,
        filt_quantiles, smooth_quantiles, true_rates_df, time_series,
        wspace=0.15
    ):
    """
    Plots the central 90% band plots of the filtering and smoothing 5%,
    50% (median), and 95% quantiles.
    """
    L = len(true_rates_df.columns)
    
    fig, axes = plt.subplots(L, 2, figsize=(TEXTWIDTH_IN, 2*L))
    
    for lam_idx, lam in enumerate(true_rates_df.columns):
        
        p, q = lam.split("_")[1:]
        p, q = int(p), int(q)
        latex_rate_symbol, _ = get_latex_rate_symbol(p, q)
        
        # Filtering quantiles
        filter_median = filt_quantiles.sel(lam=lam, quantile=0.5)
        filter_lq = filt_quantiles.sel(lam=lam, quantile=0.05)
        filter_uq = filt_quantiles.sel(lam=lam, quantile=0.95)
        
        # Smoothing quantiles
        smooth_lq = smooth_quantiles[:, lam_idx, 0]
        smooth_median = smooth_quantiles[:, lam_idx, 1]
        smooth_uq = smooth_quantiles[:, lam_idx, 2]
        
        # --------------------
        # Filtering
        # --------------------
        axes[lam_idx, 0].plot(
            time_series,
            true_rates_df[lam].values,
            label=f"True {latex_rate_symbol}",
            color="red",
            alpha=0.7
        )
        
        axes[lam_idx, 0].plot(
            time_series,
            filter_median,
            color="green",
            label="PF median",
            alpha=0.7
        )
        
        axes[lam_idx, 0].fill_between(
            time_series,
            y1=filter_lq,
            y2=filter_uq,
            color="green",
            alpha=0.3
        )
        
        axes[lam_idx, 0].set_title(f"Filtering: {latex_rate_symbol}")
        # axes[lam_idx, 0].legend()
        
        # --------------------
        # Smoothing
        # --------------------
        axes[lam_idx, 1].plot(
            time_series,
            true_rates_df[lam].values,
            label=f"True {latex_rate_symbol}",
            color="red",
            alpha=0.7
        )
        
        axes[lam_idx, 1].plot(
            time_series,
            smooth_median,
            color="blue",
            label="PS median",
            alpha=0.7
        )
        
        axes[lam_idx, 1].fill_between(
            time_series,
            y1=smooth_lq,
            y2=smooth_uq,
            color="blue",
            alpha=0.3
        )
        
        axes[lam_idx, 1].set_title(f"Smoothing: {latex_rate_symbol}")
        # axes[lam_idx, 1].legend()
    
    for ax in axes[-1]:
        ax.set_xlabel("time")
    
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate")
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    # plt.tight_layout()
    plt.subplots_adjust(
        # left=0.1,    # Left margin
        # right=0.9,   # Right margin
        # top=0.9,     # Top margin
        # bottom=0.1,  # Bottom margin
        wspace=wspace, # Horizontal spacing between subplots (width fraction)
        hspace=0.4     # Vertical spacing between subplots (height fraction)
    )
    plt.show()


def plot_ess(ess, time_series, N, fig_width_scaler=1):
    """Plots filtering ESS over time."""
    from math import log10
    
    Y = int(log10(N))
    X = int(N / (10**Y))
    
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * fig_width_scaler, 2))
    
    ax.plot(time_series, ess, color="red")
    ax.set_xlabel("time")
    ax.set_ylabel("ESS")
    
    padding = 0.3 * 10**Y  # 5% of one tick interval
    ax.set_ylim(-padding, (X + 0.2) * 10**Y)
    ax.set_yticks(np.arange(0, X + 1, 2) * 10**Y)
    ax.set_yticklabels(range(0, X + 1, 2))
    
    ax.text(
        0, 1.02, rf"$\times 10^{{{Y}}}$",
        transform=ax.transAxes,
        ha="left", va="bottom"
    )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    plt.show()


def plot_state_proportions(
        data, time_series, n
    ):
    """Plot the state proportions over time."""
    
    counts = np.array([
        np.bincount(data_k.reshape(-1), minlength=n)
        for data_k in data
    ])
    
    proportions = (
        counts /
        np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
    )
    
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.2))
    
    labels = [f"{value + 1}" for value in range(n)]
    
    ax.stackplot(
        time_series,
        proportions.T,
        labels=labels,
        alpha=0.7,
        edgecolor="white",
        linewidth=1
    )
    
    ax.set_xlabel("time")
    ax.set_ylabel("Proportion")
    
    t_max = time_series[-1]
    ax.set_xlim(0, t_max)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    
    ax.legend(
        ncols=n,
        title="State",
        loc="lower right",
        framealpha=0.5
    )
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    # fig.tight_layout()
    plt.show()


def plot_ess_and_state_proportions(ess, data, time_series, N, n):
    """Plots filtering ESS and state proportions side-by-side."""
    from math import log10

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH_IN, 2.2),
        sharex=True
    )

    # --- ESS plot ---
    Y = int(log10(N)) - 1
    X = int(N / (10**Y))

    ax1.plot(time_series, ess, color="red")
    ax1.set_xlabel("time")
    ax1.set_ylabel("ESS")

    padding = 0.3 * 10**Y
    ax1.set_ylim(-padding, (X + 0.2) * 10**Y)
    ax1.set_yticks(np.arange(0, X + 1, 2) * 10**Y)
    ax1.set_yticklabels(range(0, X + 1, 2))

    ax1.text(
        0, 1.02, rf"$\times 10^{{{Y}}}$",
        transform=ax1.transAxes,
        ha="left", va="bottom"
    )

    # --- State proportions plot ---
    counts = np.array([
        np.bincount(data_k.reshape(-1), minlength=n)
        for data_k in data
    ])

    proportions = (
        counts /
        np.array([len(data_k.reshape(-1)) for data_k in data])[:, None]
    )

    labels = [f"{value + 1}" for value in range(n)]

    ax2.stackplot(
        time_series,
        proportions.T,
        labels=labels,
        alpha=0.7,
        edgecolor="white",
        linewidth=1
    )

    ax2.set_xlabel("time")
    ax2.set_ylabel("Proportion")

    ax2.set_xlim(0, time_series[-1])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.0, 0.5, 1.0])

    ax2.legend(
        ncols=n,
        title="State",
        loc="upper left",
        framealpha=0.5
    )

    # Apply consistent font sizes
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)

    fig.tight_layout()
    plt.show()


def plot_data(data, time_series, n, fig_width_scaler=1):
    """Plots data over time."""
    
    data = np.vstack(data) + 1
    
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * fig_width_scaler, 2))
    
    ax.plot(time_series, data, color="blue")
    ax.set_xlabel("time")
    ax.set_ylabel("State")
    
    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)
    plt.show()


def plot_ess_and_data(ess, data, time_series, N, n):
    """Plots filtering ESS and state proportions side-by-side."""
    from math import log10

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH_IN, 2.2),
        sharex=True
    )

    ## ESS plot ##
    
    Y = int(log10(N))
    X = int(N / (10**Y))

    ax1.plot(time_series, ess, color="red")
    ax1.set_xlabel("time")
    ax1.set_ylabel("ESS")

    padding = 0.3 * 10**Y
    ax1.set_ylim(-padding, (X + 0.2) * 10**Y)
    ax1.set_yticks(np.arange(0, X + 1, 2) * 10**Y)
    ax1.set_yticklabels(range(0, X + 1, 2))

    ax1.text(
        0, 1.02, rf"$\times 10^{{{Y}}}$",
        transform=ax1.transAxes,
        ha="left", va="bottom"
    )

    ## Data plot ##
    
    data = np.vstack(data) + 1

    ax2.plot(
        time_series,
        data,
        alpha=0.7,
        linewidth=1
    )

    ax2.set_xlabel("time")
    ax2.set_ylabel("State")

    ax2.set_yticks(np.arange(1, n+1))

    apply_font_sizes(fig, FONT_SIZE, TICK_FONT_SIZE)

    fig.tight_layout()
    plt.show()

