# -*- coding: utf-8 -*-

"""

Simulation Study #2: Ehrenfest Model

Particle filtering and smoothing for inferring the box volumes over time.

"""


#### IMPORTS ####

import numpy as np

from true_rates_data_simulation.true_rates_simulation_functions import (
    simulate_ehrenfest_rates_v2
)
from true_rates_data_simulation.data_simulation_functions import (
    simulate_data
)

from ctmc_modules.functions import (
    get_k_series, get_time_series,
    make_true_rates_dataframe,
    get_rw_all_state_p_initial_config,
)

from paper_simulation_studies.ehrenfest_functions_ssm import (
    EhrenfestSSM_BV,
    run_particle_filter_bv,
    run_particle_smoother_bv,
)

from plotting_modules.paper_plotting import (
    plot_volumes_and_true_rates,
    plot_central_90_bandplots_stacked_fs_bv,
    plot_ess,
    plot_state_proportions,
    plot_ess_and_state_proportions
)


#### CONSTANTS ####

PLOT_EXTRAS = False


#### RATE FUNCTIONS ####

def V_A(k, K):
    return np.ones_like(K) * (
        (k <= K / 3) +
        (k > 2 * K / 3) * 0.5 +
        (k > K / 3) * (k <= 2 * K / 3) * (1.5 - 1.5 * k / K)
    )

def V_B(k, K):
    return np.ones_like(k)

def get_a_to_b_rate(k, K, omega):
    return omega / V_A(k, K)

def get_b_to_a_rate(k, K, omega):
    return omega / V_B(k, K)

def get_true_volumes(K, V_A, V_B):
    return [
        np.array([V_A(k, K), V_B(k, K)]).reshape(1, -1)
        for k in range(K)
    ]


#### Define parameters used in each run ####

omega = np.array([1, 1])
L = 4
n = L + 1

# Number of particles & smoothing trajectories in every run
N_GLOB = 50000

# t_K - t_1 == 3 <==> (K - 1) * delta_t == 3
run_params_dict = {
    "K":       [301,    301   ],
    "delta_t": [0.01,   0.01  ],
    "C":       [0.1,    0.1   ],
    "J":       [1,      1000  ],
    "N":       [N_GLOB, N_GLOB],
}

num_runs = len(run_params_dict["K"])
assert all(len(value) == num_runs for value in run_params_dict.values())


#### Plot true rates and box volumes ####

K = 301
delta_t = 0.01
k_series = get_k_series(K)
time_series = get_time_series(delta_t, k_series)

true_volumes = get_true_volumes(K, V_A, V_B)

true_rates = simulate_ehrenfest_rates_v2(
    get_a_to_b_rate, get_b_to_a_rate, L, K, omega
)

true_rates_df = make_true_rates_dataframe(true_rates=true_rates, n=n)

plot_volumes_and_true_rates(
    true_rates_df=true_rates_df,
    k_series=k_series,
    time_series=time_series,
    K=K, V_A=V_A, V_B=V_B
)


#### Generate results and plots of each run ####

for run_idx in range(num_runs):
    
    print()
    print(f"Executing Run {run_idx}")
    print()
    
    
    ### CTMC SSM Parameters ###
    
    delta_t = run_params_dict["delta_t"][run_idx]
    C = run_params_dict["C"][run_idx]
    J = run_params_dict["J"][run_idx]
    
    
    ### Particle filtering, smoothing, and other parameters ###
    
    N = run_params_dict["N"][run_idx]
    K = run_params_dict["K"][run_idx]
    k_series = get_k_series(K)
    time_series = get_time_series(delta_t, k_series)
    
    
    ### Simulate true volumes and rates ###
    
    true_volumes = get_true_volumes(K, V_A, V_B)
    
    true_rates = simulate_ehrenfest_rates_v2(
        get_a_to_b_rate, get_b_to_a_rate, L, K, omega
    )
    
    
    ### Define Ehrenfest SSM ###
    
    mu0  = true_volumes[0].reshape(-1)
    var0 = np.array([0.25, 0.25])
     
    y_init = get_rw_all_state_p_initial_config(J=J, p=n, n=n)
    
    ehrenfest_ssm = EhrenfestSSM_BV(
        L=L, delta_t=delta_t, J=J, C=C,
        mu0=mu0, var0=var0, omega=omega,
        y_init=y_init,
        px_verbose=True, px_verbose_int=20, 
    )
    
    y_init = ehrenfest_ssm.y_init
    
    
    ### Simulate data ###
    
    data = simulate_data(
        true_rates=true_rates, n=n, J=J, delta_t=delta_t, y_init=y_init
    )
    
    
    ### Run bootstrap particle filter ###
    
    ds_boot, pf_boot = run_particle_filter_bv(
        ehrenfest_ssm=ehrenfest_ssm,
        data=data,
        N=N,
        k_series=k_series,
        also_return_pf_boot=True,
    )
    
    
    ### Run particle smoother ###
    
    smooth_trajectories, smooth_quantiles = run_particle_smoother_bv(
        pf_boot, M=N
    )
    
    
    ### Plot filtering and smoothing quantiles ###
        
    plot_central_90_bandplots_stacked_fs_bv(
        filt_quantiles=ds_boot["X_quantiles"],
        smooth_quantiles=smooth_quantiles,
        true_volumes=true_volumes,
        time_series=time_series,
        wspace=0.2
    )
    
    
    ### Plot ESS ###
    
    plot_ess(ds_boot["ESS"], time_series, N)
    
    
    ### Plot state proportions ###
    
    plot_state_proportions(data, time_series, n)
    
    
    ### Plot ESS and state proportions side-by-side ###
    
    if False:
        plot_ess_and_state_proportions(
            ess=ds_boot["ESS"],
            data=data,
            time_series=time_series,
            N=N,
            n=n
        )               
    
