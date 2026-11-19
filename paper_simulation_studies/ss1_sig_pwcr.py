# -*- coding: utf-8 -*-

"""

Simulation Study #1: Two-State CTMC

Particle filtering and smoothing for inferring the Sigmoid & PWC transition
rates.

"""


#### IMPORTS ####

import numpy as np

from true_rates_data_simulation.true_rates_simulation_functions import (
    simulate_sigmoid_and_abrupt_rates
)
from true_rates_data_simulation.data_simulation_functions import (
    simulate_data
)

from ctmc_modules.functions import (
    get_k_series, get_time_series,
    run_particle_filter,
    run_particle_smoother,
    make_true_rates_dataframe
)
from ctmc_modules.ctmc_ssms import CTMC

from plotting_modules.paper_plotting import (
    plot_true_rates_ss1,
    plot_central_90_bandplots_stacked_fs,
    plot_ess,
    plot_state_proportions,
    plot_ess_and_state_proportions
)


#### CONSTANTS ####

PLOT_EXTRAS = False


#### Define parameters used in each run ####

n = 2

# Number of particles & smoothing trajectories in every run
N_GLOB = 500000

# t_K - t_1 == 3 <==> (K - 1) * delta_t == 3
run_params_dict = {
    "K":       [101,    101   ],
    "delta_t": [0.03,   0.03  ],
    "C":       [0.5,    2     ],
    "J":       [1000,   1000  ],
    "N":       [N_GLOB, N_GLOB],
    "plot_sp": [True,   False ],
}

num_runs = len(run_params_dict["K"])
assert all(len(value) == num_runs for value in run_params_dict.values())


#### Plot true rates ####

K = 301
delta_t = 0.01
time_series = get_time_series(delta_t, get_k_series(K))

true_rates = simulate_sigmoid_and_abrupt_rates(K=K, b2=5)
true_rates_df = make_true_rates_dataframe(true_rates=true_rates, n=n)

plot_true_rates_ss1(true_rates_df, time_series)


#### Generate results and plots of each run ####

for run_idx in range(num_runs):
    
    print()
    print(f"Executing Run {run_idx}")
    print()
    
    
    ### Simulate true rates ###
    
    K = run_params_dict["K"][run_idx]
    true_rates = simulate_sigmoid_and_abrupt_rates(K=K, b2=5)
    
    
    ### Define CTMC SSM and its parameters ###
    
    delta_t = run_params_dict["delta_t"][run_idx]
    C = run_params_dict["C"][run_idx]
    J = run_params_dict["J"][run_idx]
    
    mu0  = true_rates[0].reshape(-1) + np.array([-0.4, 0.3])
    var0 = np.array([1, 1])
    
    ctmc_ssm = CTMC(
        mu0=mu0, var0=var0, n=n, delta_t=delta_t, J=J, C=C,
        k_verbose=True, k_verbose_interval=10
    )
    
    y_init = ctmc_ssm.y_init
    
    
    ### Simulate data ###
    
    data = simulate_data(
        true_rates=true_rates, n=n, J=J, delta_t=delta_t, y_init=y_init
    )
    
    
    ### Store true transition rates in pandas dataframe ###
    
    true_rates_df = make_true_rates_dataframe(true_rates=true_rates, n=n)
    
    
    ### Particle filtering, smoothing, and other parameters ###
    
    N = run_params_dict["N"][run_idx]
    k_series = get_k_series(K)
    time_series = get_time_series(delta_t, k_series)
    
    
    ### Run bootstrap particle filter ###
    
    ds_boot, pf_boot = run_particle_filter(
        ctmc_ssm=ctmc_ssm,
        data=data,
        N=N,
        k_series=k_series,
        also_return_pf_boot=True,
    )
    
    
    ### Run particle smoother ###
    
    smooth_trajectories, smooth_quantiles = run_particle_smoother(
        pf=pf_boot, M=N, k_series=k_series
    )
    
    
    ### Plot filtering and smoothing quantiles ###
    
    plot_central_90_bandplots_stacked_fs(
        filt_quantiles=ds_boot["X_quantiles"],
        smooth_quantiles=smooth_quantiles,
        true_rates_df=true_rates_df,
        time_series=time_series
    )
    
    
    ### Plot ESS ###
    
    plot_ess(ds_boot["ESS"], time_series, N, fig_width_scaler=0.5)
    
    
    #### Plot state proportions ####
    
    if run_params_dict["plot_sp"][run_idx]:
        print(f"Plotting state proportions of Run {run_idx}.")
        plot_state_proportions(data, time_series, n)
    
    
    #### Plot ESS and state proportions side-by-side ####
    
    if False:
        plot_ess_and_state_proportions(
            ess=ds_boot["ESS"],
            data=data,
            time_series=time_series,
            N=N,
            n=n
        )
    
    print()
    print()
    print()


