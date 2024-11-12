# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plot
import matplotlib.collections as collections

# Inputs
Case = 'SPH-K1-025'
TimeSphInj = 300  # s
TimeSphAct = 80  # s
Freq = 0.25  # Hz
PeakForce = 100  # pN

# Load simulation results
SphTauLst0 = np.load(Case + '_0_' + 'Sphere_Tau.npy')
SphXDisp0 = np.load(Case + '_0_' + 'Sphere_X.npy')
SphTauLst1 = np.load(Case + '_1_' + 'Sphere_Tau.npy')
SphXDisp1 = np.load(Case + '_1_' + 'Sphere_X.npy')
SphTauLst2 = np.load(Case + '_2_' + 'Sphere_Tau.npy')
SphXDisp2 = np.load(Case + '_2_' + 'Sphere_X.npy')
SphTauLst3 = np.load(Case + '_3_' + 'Sphere_Tau.npy')
SphXDisp3 = np.load(Case + '_3_' + 'Sphere_X.npy')
SphTauLst4 = np.load(Case + '_4_' + 'Sphere_Tau.npy')
SphXDisp4 = np.load(Case + '_4_' + 'Sphere_X.npy')
SphTauLst5 = np.load(Case + '_5_' + 'Sphere_Tau.npy')
SphXDisp5 = np.load(Case + '_5_' + 'Sphere_X.npy')
SphTauLst6 = np.load(Case + '_6_' + 'Sphere_Tau.npy')
SphXDisp6 = np.load(Case + '_6_' + 'Sphere_X.npy')
SphTauLst7 = np.load(Case + '_7_' + 'Sphere_Tau.npy')
SphXDisp7 = np.load(Case + '_7_' + 'Sphere_X.npy')
SphTauLst8 = np.load(Case + '_8_' + 'Sphere_Tau.npy')
SphXDisp8 = np.load(Case + '_8_' + 'Sphere_X.npy')
SphTauLst9 = np.load(Case + '_9_' + 'Sphere_Tau.npy')
SphXDisp9 = np.load(Case + '_9_' + 'Sphere_X.npy')

# Average displacement
SphTauLstAvg, SphXDispAvg = [], []
XDisp0, XDisp1, XDisp2, XDisp3, XDisp4, XDisp5, XDisp6, XDisp7, XDisp8, XDisp9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
for i in range(0, (TimeSphAct * 100) + 100):
    SphTauLstAvg.append(TimeSphInj + (i / 100))
for i in range(len(SphTauLstAvg)):
    GetValue0 = 0
    for j in range(len(SphXDisp0)):
        if SphTauLst0[j] >= SphTauLstAvg[i] and GetValue0 == 0:
            XDisp0 = SphXDisp0[j]
            GetValue0 = 1
        elif j == len(SphXDisp0 - 1) and SphTauLst0[len(SphTauLst0) - 1] < SphTauLstAvg[i] and GetValue0 == 0:
            XDisp0 = SphXDisp0[len(SphTauLst0) - 1]
            GetValue0 = 1
    GetValue1 = 0
    for j in range(len(SphXDisp1)):
        if SphTauLst1[j] >= SphTauLstAvg[i] and GetValue1 == 0:
            XDisp1 = SphXDisp1[j]
            GetValue1 = 1
        elif j == len(SphXDisp1 - 1) and SphTauLst1[len(SphTauLst1) - 1] < SphTauLstAvg[i] and GetValue1 == 0:
            XDisp1 = SphXDisp1[len(SphTauLst1) - 1]
            GetValue1 = 1
    GetValue2 = 0
    for j in range(len(SphXDisp2)):
        if SphTauLst2[j] >= SphTauLstAvg[i] and GetValue2 == 0:
            XDisp2 = SphXDisp2[j]
            GetValue2 = 1
        elif j == len(SphXDisp2 - 1) and SphTauLst2[len(SphTauLst2) - 1] < SphTauLstAvg[i] and GetValue2 == 0:
            XDisp2 = SphXDisp2[len(SphTauLst2) - 1]
            GetValue2 = 1
    GetValue3 = 0
    for j in range(len(SphXDisp3)):
        if SphTauLst3[j] >= SphTauLstAvg[i] and GetValue3 == 0:
            XDisp3 = SphXDisp3[j]
            GetValue3 = 1
        elif j == len(SphXDisp3 - 1) and SphTauLst3[len(SphTauLst3) - 1] < SphTauLstAvg[i] and GetValue3 == 0:
            XDisp3 = SphXDisp3[len(SphTauLst3) - 1]
            GetValue3 = 1
    GetValue4 = 0
    for j in range(len(SphXDisp4)):
        if SphTauLst4[j] >= SphTauLstAvg[i] and GetValue4 == 0:
            XDisp4 = SphXDisp4[j]
            GetValue4 = 1
        elif j == len(SphXDisp4 - 1) and SphTauLst4[len(SphTauLst4) - 1] < SphTauLstAvg[i] and GetValue4 == 0:
            XDisp4 = SphXDisp4[len(SphTauLst4) - 1]
            GetValue4 = 1
    GetValue5 = 0
    for j in range(len(SphXDisp5)):
        if SphTauLst5[j] >= SphTauLstAvg[i] and GetValue5 == 0:
            XDisp5 = SphXDisp5[j]
            GetValue5 = 1
        elif j == len(SphXDisp5 - 1) and SphTauLst5[len(SphTauLst5) - 1] < SphTauLstAvg[i] and GetValue5 == 0:
            XDisp5 = SphXDisp5[len(SphTauLst5) - 1]
            GetValue5 = 1
    GetValue6 = 0
    for j in range(len(SphXDisp6)):
        if SphTauLst6[j] >= SphTauLstAvg[i] and GetValue6 == 0:
            XDisp6 = SphXDisp6[j]
            GetValue6 = 1
        elif j == len(SphXDisp6 - 1) and SphTauLst6[len(SphTauLst6) - 1] < SphTauLstAvg[i] and GetValue6 == 0:
            XDisp6 = SphXDisp6[len(SphTauLst6) - 1]
            GetValue6 = 1
    GetValue7 = 0
    for j in range(len(SphXDisp7)):
        if SphTauLst7[j] >= SphTauLstAvg[i] and GetValue7 == 0:
            XDisp7 = SphXDisp7[j]
            GetValue7 = 1
        elif j == len(SphXDisp7 - 1) and SphTauLst7[len(SphTauLst7) - 1] < SphTauLstAvg[i] and GetValue7 == 0:
            XDisp7 = SphXDisp7[len(SphTauLst7) - 1]
            GetValue7 = 1
    GetValue8 = 0
    for j in range(len(SphXDisp8)):
        if SphTauLst8[j] >= SphTauLstAvg[i] and GetValue8 == 0:
            XDisp8 = SphXDisp8[j]
            GetValue8 = 1
        elif j == len(SphXDisp8 - 1) and SphTauLst8[len(SphTauLst8) - 1] < SphTauLstAvg[i] and GetValue8 == 0:
            XDisp8 = SphXDisp8[len(SphTauLst8) - 1]
            GetValue8 = 1
    GetValue9 = 0
    for j in range(len(SphXDisp9)):
        if SphTauLst9[j] >= SphTauLstAvg[i] and GetValue9 == 0:
            XDisp9 = SphXDisp9[j]
            GetValue9 = 1
        elif j == len(SphXDisp9 - 1) and SphTauLst9[len(SphTauLst9) - 1] < SphTauLstAvg[i] and GetValue9 == 0:
            XDisp9 = SphXDisp9[len(SphTauLst9) - 1]
            GetValue9 = 1
    SphXDispAvg.append((XDisp0 + XDisp1 + XDisp2 + XDisp3 + XDisp4 + XDisp5 + XDisp6 + XDisp7 + XDisp8 + XDisp9) / 10)

np.save(Case + '_XDisp_Avg', SphXDispAvg)

# Plots
plot.figure(figsize=(6, 8))
ax1 = plot.subplot2grid(shape=(2, 1), loc=(0, 0), rowspan=1, colspan=1)
ax2 = plot.subplot2grid(shape=(2, 1), loc=(1, 0), rowspan=1, colspan=1)

# Force
TimeSec, TimeMin, Force1 = [], [], []
for i in range(0, (TimeSphInj + TimeSphAct) * 100):
    TimeSec.append(i / 100)
    TimeMin.append(i / 6000)
for i in range(len(TimeSec)):
    Force1.append(PeakForce * math.sin(Freq * 2 * math.pi * (TimeSec[i] - 0 * 60)))
ax1.plot(TimeMin, Force1)
ax1.set_title("Force in +/- x direction", fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (min)', fontsize=14), ax1.set_ylabel('Force (pN)', fontsize=14)
ax1.set_xlim(TimeSphInj / 60, (TimeSphInj + TimeSphAct) / 60), ax1.set_ylim(-120, 120)

# Sphere log
SphTauLstAvgMin = []
for i in range(len(SphTauLstAvg)):
    SphTauLstAvgMin.append(SphTauLstAvg[i] / 60)
ax2.plot(SphTauLstAvgMin, SphXDispAvg, c="blue", label="X Disp (avg)", zorder=6)
ax2.plot(SphTauLst0 / 60, SphXDisp0, c="pink", label="X Disp (single)")
ax2.plot(SphTauLst1 / 60, SphXDisp1, c="pink")
ax2.plot(SphTauLst2 / 60, SphXDisp2, c="pink")
ax2.plot(SphTauLst3 / 60, SphXDisp3, c="pink")
ax2.plot(SphTauLst4 / 60, SphXDisp4, c="pink")
ax2.plot(SphTauLst5 / 60, SphXDisp5, c="pink")
ax2.plot(SphTauLst6 / 60, SphXDisp6, c="pink")
ax2.plot(SphTauLst7 / 60, SphXDisp7, c="pink")
ax2.plot(SphTauLst8 / 60, SphXDisp8, c="pink")
ax2.plot(SphTauLst9 / 60, SphXDisp9, c="pink")
ax2.set_title("Average Sphere Displacement", fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (min)', fontsize=14), ax2.set_ylabel('Displacement (μm)', fontsize=14)
ax2.set_xlim(TimeSphInj / 60, (TimeSphInj + TimeSphAct) / 60)
# ax2.set_ylim(-5, 5)
ax2.legend(ncol=1, loc=0)

plot.tight_layout(), plot.savefig(Case + '_Force_Rhe_Results.pdf')
