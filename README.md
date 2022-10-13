# active-filament-networks
Advection-Diffusion Models to Describe Active Filament Networks

# Adv_Diff_Sim_2022-10-11
This file runs the simulation.
Inputs:
 - Line 16: Case ID and Iteration ID (these are strings used for saving and loading data)
 - Line 17: Number of kinesin motors per microtubule and number of myosin motors per actin filament
 - Line 18: Gamma_crosslink for microtubules (in (pN*ms/nm)/fil) and gamma_crosslink for actin filaments (in (pN*ms/nm)/fil)
 - Line 19: Volume fraction of microtubules, volume fraction of actin filaments, number of iterations, and timestep (in ms)
 - Line 20: Number of columns, number of rows, and number of annuli (for spatial distribution analysis)

# Adv_Diff_Plot_2022-10-11
This file plots the results.
Inputs:
 - Line 9: Case ID and Iteration ID(s)
