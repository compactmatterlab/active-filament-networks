# Import required packages
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.collections as collections
import statistics as stat
import math as ma

# Input
Case, Iter1, Iter2, Iter3, = 'x', '1_', '2_', '3_'

# Geometry
BaseColumns, BaseRows, BaseGridDist, ResFac, NumAnn = 29, 34, 5, 2, 6  # - / - / um / - / -
NumColumns, NumRows, GridDist = BaseColumns * ResFac, BaseRows * ResFac, BaseGridDist / ResFac
TotPts, NumIterations = NumColumns * NumRows, 1000000

# Load results
# Iteration 1
StateIni, LineSegIni = np.load(Case + '_' + Iter1 + 'State_Ini.npy'), np.load(Case + '_' + Iter1 + 'LineSeg_Ini.npy')
StateFin, LineSegFin = np.load(Case + '_' + Iter1 + 'State_Fin.npy'), np.load(Case + '_' + Iter1 + 'LineSeg_Fin.npy')
MTRadI1, ActRadI1 = np.load(Case + '_' + Iter1 + 'MTRad_Ini.npy'), np.load(Case + '_' + Iter1 + 'ActRad_Ini.npy')
MTCoI1, ActCoI1 = np.load(Case + '_' + Iter1 + 'MTCo_Ini.npy'), np.load(Case + '_' + Iter1 + 'ActCo_Ini.npy')
MTRadF1, ActRadF1 = np.load(Case + '_' + Iter1 + 'MTRad_Fin.npy'), np.load(Case + '_' + Iter1 + 'ActRad_Fin.npy')
MTCoF1, ActCoF1 = np.load(Case + '_' + Iter1 + 'MTCo_Fin.npy'), np.load(Case + '_' + Iter1 + 'ActCo_Fin.npy')
# Iteration 2
MTRadI2, ActRadI2 = np.load(Case + '_' + Iter2 + 'MTRad_Ini.npy'), np.load(Case + '_' + Iter2 + 'ActRad_Ini.npy')
MTCoI2, ActCoI2 = np.load(Case + '_' + Iter2 + 'MTCo_Ini.npy'), np.load(Case + '_' + Iter2 + 'ActCo_Ini.npy')
MTRadF2, ActRadF2 = np.load(Case + '_' + Iter2 + 'MTRad_Fin.npy'), np.load(Case + '_' + Iter2 + 'ActRad_Fin.npy')
MTCoF2, ActCoF2 = np.load(Case + '_' + Iter2 + 'MTCo_Fin.npy'), np.load(Case + '_' + Iter2 + 'ActCo_Fin.npy')
# Iteration 3
MTRadI3, ActRadI3 = np.load(Case + '_' + Iter3 + 'MTRad_Ini.npy'), np.load(Case + '_' + Iter3 + 'ActRad_Ini.npy')
MTCoI3, ActCoI3 = np.load(Case + '_' + Iter3 + 'MTCo_Ini.npy'), np.load(Case + '_' + Iter3 + 'ActCo_Ini.npy')
MTRadF3, ActRadF3 = np.load(Case + '_' + Iter3 + 'MTRad_Fin.npy'), np.load(Case + '_' + Iter3 + 'ActRad_Fin.npy')
MTCoF3, ActCoF3 = np.load(Case + '_' + Iter3 + 'MTCo_Fin.npy'), np.load(Case + '_' + Iter3 + 'ActCo_Fin.npy')

# Plots
plot.figure(figsize=(9, 9))
ax1 = plot.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=1, colspan=1)
ax2 = plot.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=1)
ax3 = plot.subplot2grid(shape=(2, 2), loc=(0, 1), rowspan=1, colspan=1)
ax4 = plot.subplot2grid(shape=(2, 2), loc=(1, 1), rowspan=1, colspan=1)

# Initial condition
ax1.set_title("Initial Condition", fontsize=18, fontweight='bold')
ax1.set_xlabel('Microns (μm)', fontsize=18), ax1.set_ylabel('Microns (μm)', fontsize=18)
ax1.set_xlim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
ax1.set_ylim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
MTLinesIni, ActLinesIni = [], []
for i in range(TotPts):
    if StateIni[i] == 1:
        MTLinesIni.append(LineSegIni[i])
    elif StateIni[i] == 2:
        ActLinesIni.append(LineSegIni[i])
MTLineCollIni = collections.LineCollection(MTLinesIni, color="red", linewidth=3)
ActLineCollIni = collections.LineCollection(ActLinesIni, color="green", linewidth=3)
ax1.add_collection(MTLineCollIni), ax1.add_collection(ActLineCollIni)

# Condition after simulation
ax2.set_title("Condition After Simulation", fontsize=18, fontweight='bold')
ax2.set_xlabel('Microns (μm)', fontsize=18), ax2.set_ylabel('Microns (μm)', fontsize=18)
ax2.set_xlim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
ax2.set_ylim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
MTLinesFin, ActLinesFin = [], []
for i in range(TotPts):
    if StateFin[i] == 1:
        MTLinesFin.append(LineSegFin[i])
    elif StateFin[i] == 2:
        ActLinesFin.append(LineSegFin[i])
MTLineCollFin = collections.LineCollection(MTLinesFin, color="red", linewidth=3)
ActLineCollFin = collections.LineCollection(ActLinesFin, color="green", linewidth=3)
ax2.add_collection(MTLineCollFin), ax2.add_collection(ActLineCollFin)

# Radial distribution
Rad = []
for i in range(NumAnn):
    Rad.append((i + 1) * BaseGridDist / ResFac)
MTRadIAvg, MTRadFAvg = (MTRadI1 + MTRadI2 + MTRadI3) / 3, (MTRadF1 + MTRadF2 + MTRadF3) / 3
ActRadIAvg, ActRadFAvg = (ActRadI1 + ActRadI2 + ActRadI3) / 3, (ActRadF1 + ActRadF2 + ActRadF3) / 3
MTRadISD, ActRadISD, MTRadFSD, ActRadFSD = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
for i in range(NumAnn):
    MTRadISD[i] = stat.stdev([MTRadI1[i], MTRadI2[i], MTRadI3[i]])
    ActRadISD[i] = stat.stdev([ActRadI1[i], ActRadI2[i], ActRadI3[i]])
    MTRadFSD[i] = stat.stdev([MTRadF1[i], MTRadF2[i], MTRadF3[i]])
    ActRadFSD[i] = stat.stdev([ActRadF1[i], ActRadF2[i], ActRadF3[i]])
ax3.scatter(Rad, MTRadIAvg, marker="s", c="red", alpha=0.5, edgecolors='none', s=200, label="MT, t$_i$")
ax3.scatter(Rad, ActRadIAvg, marker="s", c="green", alpha=0.5, edgecolors='none', s=200, label="Actin, t$_i$")
ax3.scatter(Rad, MTRadFAvg, marker="^", c="red", edgecolors='none', s=200, label="MT, t$_f$")
ax3.scatter(Rad, ActRadFAvg, marker="^", c="green", edgecolors='none', s=200, label="Actin, t$_f$")
ax3.errorbar(Rad, MTRadIAvg, yerr=(MTRadISD/ma.sqrt(3)), c="red")
ax3.errorbar(Rad, MTRadFAvg, yerr=(MTRadFSD/ma.sqrt(3)), c="red")
ax3.errorbar(Rad, ActRadIAvg, yerr=(ActRadISD/ma.sqrt(3)), c="green")
ax3.errorbar(Rad, ActRadFAvg, yerr=(ActRadFSD/ma.sqrt(3)), c="green")
ax3.plot(Rad, MTRadIAvg, c="red", alpha=0.5)
ax3.plot(Rad, ActRadIAvg, c="green", alpha=0.5)
ax3.plot(Rad, MTRadFAvg, c="red")
ax3.plot(Rad, ActRadFAvg, c="green")
ax3.set_title("Radial Distribution", fontsize=18, fontweight='bold'), ax3.set_xlabel('r (μm)', fontsize=18)
ax3.set_ylabel('g(r)', fontsize=18), ax3.set_ylim(0.8, 2.2)
ax3.legend(ncol=2, loc=1), ax3.set_xscale('log')
ax3.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 15]), ax3.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 15])

# Codistribution
MTCoIAvg, MTCoFAvg = (MTCoI1 + MTCoI2 + MTCoI3) / 3, (MTCoF1 + MTCoF2 + MTCoF3) / 3
ActCoIAvg, ActCoFAvg = (ActCoI1 + ActCoI2 + ActCoI3) / 3, (ActCoF1 + ActCoF2 + ActCoF3) / 3
MTCoISD, ActCoISD, MTCoFSD, ActCoFSD = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
for i in range(NumAnn):
    MTCoISD[i] = stat.stdev([MTCoI1[i], MTCoI2[i], MTCoI3[i]])
    ActCoISD[i] = stat.stdev([ActCoI1[i], ActCoI2[i], ActCoI3[i]])
    MTCoFSD[i] = stat.stdev([MTCoF1[i], MTCoF2[i], MTCoF3[i]])
    ActCoFSD[i] = stat.stdev([ActCoF1[i], ActCoF2[i], ActCoF3[i]])
ax4.scatter(Rad, MTCoIAvg, marker="s", c="red", alpha=0.5, edgecolors='none', s=200, label="MT, t$_i$")
ax4.scatter(Rad, ActCoIAvg, marker="s", c="green", alpha=0.5, edgecolors='none', s=200, label="Actin, t$_i$")
ax4.scatter(Rad, MTCoFAvg, marker="^", c="red", edgecolors='none', s=200, label="MT, t$_f$")
ax4.scatter(Rad, ActCoFAvg, marker="^", c="green", edgecolors='none', s=200, label="Actin, t$_f$")
ax4.errorbar(Rad, MTCoIAvg, yerr=(MTCoISD/ma.sqrt(3)), c="red")
ax4.errorbar(Rad, MTCoFAvg, yerr=(MTCoFSD/ma.sqrt(3)), c="red")
ax4.errorbar(Rad, ActCoIAvg, yerr=(ActCoISD/ma.sqrt(3)), c="green")
ax4.errorbar(Rad, ActCoFAvg, yerr=(ActCoFSD/ma.sqrt(3)), c="green")
ax4.plot(Rad, MTCoIAvg, c="red", alpha=0.5)
ax4.plot(Rad, ActCoIAvg, c="green", alpha=0.5)
ax4.plot(Rad, MTCoFAvg, c="red")
ax4.plot(Rad, ActCoFAvg, c="green")
ax4.set_title("Codistribution", fontsize=18, fontweight='bold'), ax4.set_xlabel('r (μm)', fontsize=18)
ax4.set_ylabel('g(r)', fontsize=18), ax4.set_ylim(0.4, 1.6)
ax4.legend(ncol=2, loc=1), ax4.set_xscale('log')
ax4.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 15]), ax4.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 15])

plot.suptitle('Case ' + Case, x=0.05, ha='left', fontsize=18, fontweight='bold')
plot.tight_layout(), plot.show()
