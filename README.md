# active-filament-networks
Computational Models to Simulate Active Filament Networks

# Adv_Diff_Model
This file runs the simulation with the advection diffusion model.
The inputs are:
- Line 17: Case ID and iteration ID (these are strings used for saving and loading data)
- Line 18: Number of kinesin motors per microtubule and number of myosin motors per actin filament
- Line 19: Gamma_crosslink for microtubules (in (pN * ms / nm) / fil) and gamma_crosslink for actin filaments (in (pN * ms / nm) / fil)
- Line 20: Volume fraction of microtubules, volume fraction of actin filaments, number of columns, and number of rows
- Line 21: Total simulation time (in ms) and simulation time step (in ms)
- Line 22: Optional boundary parameter for minimally damped filament movement

# Energy_Work_Model
This file runs the simulation with the energy work balance model.
The inputs are:
- Line 14: Case ID and iteration ID (these are strings used for saving and loading data)
- Line 15: Number of kinesin motors per microtubule and number of myosin motors per actin filament
- Line 16: Crosslink bond energy per microtubule bond (in kB * T) and crosslink bond energy per actin filament bond (in kB * T)
- Line 17: Activation energy per microtubule (in kB * T) and activation energy per actin filament (in kB * T)
- Line 18: Volume fraction of microtubules, volume fraction of actin filaments, number of columns, and number of rows
- Line 19: Number of iterations and timestep per iteration (in ms)

# Plot_Multiple
This file performs spatial analysis and plotting for three sets of simulation results.
The inputs are:
- Line 9: Case ID
- Line 10: Iteration IDs
- Line 11: Number of annuli (for spatial distribution analysis)

# Plot_Single
This file performs spatial analysis and plotting for a single set of simulation results.
The inputs are:
- Line 8: Case ID, iteration ID, and case description
- Line 9: Number of annuli (for spatial distribution analysis)
