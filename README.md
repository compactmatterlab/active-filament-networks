# active-filament-networks
Computational Models to Simulate Active Filament Networks

# Adv_Diff_Model
This file runs the simulation with the advection diffusion model.
Inputs:
- Line 17: Case ID and iteration ID (these are strings used for saving and loading data)
- Line 18: Number of kinesin motors per microtubule and number of myosin motors per actin filament
- Line 19: Gamma_crosslink for microtubules (in (pN*ms/nm)/fil) and gamma_crosslink for actin filaments (in (pN*ms/nm)/fil)
- Line 20: Volume fraction of microtubules, volume fraction of actin filaments, number of columns, and number of rows
- Line 21: Total simulation time (in ms) and simulation time step (in ms)
- Line 22: Optional boundary parameter for minimally damped filament movement

# Energy_Work_Model
This file runs the simulation with the energy work balance model.
Inputs:
- Line 14: Case ID and iteration ID (these are strings used for saving and loading data)
- Line 15: Number of kinesin motors per microtubule and number of myosin motors per actin filament
- Line 16: Crosslink bond energy per microtubule bond (in kBT) and crosslink bond energy per actin filament bond (in kBT)
- Line 17: Activation energy per microtubule (in kBT) and activation energy per actin filament (in kBT)
- Line 18: Volume fraction of microtubules, volume fraction of actin filaments, number of columns, and number of rows
- Line 19: Number of iterations and timestep per iteration (in ms)

# Adv_Diff_Analysis_2022-12-09
This file performs spatial analysis and plots the results.
Inputs:
- Line 8: Case ID, iteration ID, and case description
- Line 9: Number of annuli (for spatial distribution analysis)
