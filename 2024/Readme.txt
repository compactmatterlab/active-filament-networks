Computational Model to Simulate Active Filament Network

EW_Model_2.3: This file runs the simulation. The inputs are:

Line 15: Case ID and iteration ID (these are strings used for saving and loading data)
Line 16: Average number of kinesin motors per microtubule, percentage (between 0 and 1) of kinesin motors that are active (vs passive), number of myosin motors per actin filament
Line 17: Gamma_crosslink for microtubules and gamma_crosslink for actin filaments
Line 18: Volume fraction (between 0 and 1) of microtubules, volume fraction (between 0 and 1) of actin filaments, number of columns, and number of rows
Line 19: Total simulation time

Plot_Single_V3: This file performs spatial analysis and plotting for a single set of simulation results. The inputs are:

Line 8: Case ID (must match filename from model output)
Line 9: Iteration ID (must match filename from model output)
Line 9: Number of annuli (for spatial distribution analysis)
