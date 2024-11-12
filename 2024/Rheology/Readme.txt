Rheology analysis

Run the files in this order:
- EW_Model_2.4_SPH
- Plot_Sphere_Force_Rhe
- FFT_Analysis_Rhe_Extract


EW_Model_2.4_SPH: This file runs the rheology analysis simulation. In addition to the standard model inputs, the additional inputs are:

Line 23:
- FSphMax: This is the peak value of the force that is periodically applied to the sphere.
- FSphFreq: This is the frequency of the force that is periodically applied to the sphere.
- FSphXfr: This is the decimal percentage (between 0 and 1) of the sphere force that is transferred to the filaments that it contacts.

Line 24:
- SphTauIni: This is the elapsed simulation time at which the sphere is injected into the composite.


Plot_Sphere_Force_Rhe: This file loads 10 individiual iterations of the model results and computes the average sphere displacement across the 10 iterations. The inputs are:

Line 7 - Case ID
Line 8 - TimeSphInj: This is the elapsed simulation time at which the sphere is injected into the composite.
Line 9 - TimeSphAct: This is the amount of time the sphere is active in the composite.
Line 10 - Freq: This is the frequency of the periodic force applied to the sphere.
Line 11 - PeakForce: This is the peak value of the periodic force applied to the sphere.


FFT_Analysis_Rhe_Extract: This file loads the average sphere displacement and uses a fast fourier transform analysis to compute the rheological properties. The inputs are:

Line 7 - Case ID
Line 8 - Freq: This is the frequency of the periodic force applied to the sphere.
Line 9 - PeakForce: This is the peak value of the periodic force applied to the sphere.
