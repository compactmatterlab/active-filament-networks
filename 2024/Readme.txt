Computational Model to Simulate Active Filament Network


EW_Model_2.3: This file runs the simulation. The inputs are:

Line 15:
- Case ID and iteration ID: These are strings used for saving and loading data.

Line 16:
- Average number of kinesin motors per microtubule-microtubule interaction: It is suggested to use a value somewhere in the range of 0.1 - 1. This is the average value, and the actual value for each MT-MT interaction will be selected randomly from a poisson distribution. Note that each MT has the potential to interact with other MT's that are up to 4 grid locations away.
- Percentage of kinesin motors that are active: This value must be between 0 and 1 (decimal percentage). Active motors exert forces on microtubules, while passive motors crosslink microtubules together.
- Number of myosin motors per actin-actin interaction: It is suggested to use a value in the range of 0.1 - 10. Note that each actin filament has the potential to interact with other actin filaments that are up to 4 grid locations away.

Line 17:
- Gamma_crosslink for microtubules and gamma_crosslink for actin filaments: These parameters represent crosslinking proteins. The value is applied to each MT-MT or actin-actin interaction. Note that each filament has the potential to interact with other filaments that are up to 4 grid locations away. This parameter is sensitive to the quantities of kinesin and myosin motors in the system. It is suggested to start somewhere in the range of 1 - 100 pN*s/nm/fil.

Line 18:
- Volume fraction of microtubules and volume fraction of actin filaments: These values must be between 0 and 1 (decimal percentages). It is suggested that their sum be in the range of 0.5 - 0.7 to allow enough empty space for filaments to move properly.
- Number of columns and number of rows: Each column and row is approximately 5 um wide. A suggested starting point is 20 x 23 which is approximately 100 um x 100 um.

Line 19:
- Total simulation time: It is suggested to start with values around 5 min.


Plot_Single_V3: This file performs spatial analysis and plotting for a single set of simulation results. It loads the files saved from the model. The inputs are:

Line 8:
- Case ID: This string must match the value from the model.
Line 9:
- Iteration ID: This string must match the value from the model.
Line 10:
- Number of annuli: This defines the number of concentric rings (annuli) that will be used for the spatial analysis. The spatial analysis determines, for each filament, how many same-type and other-type filaments are located in each of the annuli around each filament and averages this data across the entire simulation space. It is suggested to start with a value of 20 annuli.
