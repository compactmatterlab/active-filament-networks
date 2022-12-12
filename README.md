# active-filament-networks
Advection-Diffusion Models to Describe Active Filament Networks

# Adv_Diff_Simulation_2022-12-09
This file runs the simulation.
Inputs:
- Line 17: Case ID and iteration ID (these are strings used for saving and loading data)
- Line 18: Number of kinesin motors per microtubule and number of myosin motors per actin filament
- Line 19: Gamma_crosslink for microtubules (in (pN*ms/nm)/fil) and gamma_crosslink for actin filaments (in (pN*ms/nm)/fil)
- Line 20: Volume fraction of microtubules, volume fraction of actin filaments, number of columns, and number of rows
- Line 21: Total simulation time (in ms) and simulation time step (in ms)

# Adv_Diff_Analysis_2022-12-09
This file performs spatial analysis and plots the results.
Inputs:
- Line 8: Case ID, iteration ID, and case description
- Line 9: Number of annuli (for spatial distribution analysis)
