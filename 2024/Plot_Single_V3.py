# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plot
import matplotlib.collections as collections

# Inputs
Case = 'K1'
Iter = '65_5'
NumAnn = 20  # -

# Load simulation parameters
Input = np.load(Case + '_' + Iter + '_' + 'Input_Param.npy')
RatioMT, RatioActin, BaseColumns, BaseRows = Input[0], Input[1], int(Input[2]), int(Input[3])
BaseGridDist, ResFac = int(Input[4]), int(Input[5])

# Load simulation results
LineSegIni = np.load(Case + '_' + Iter + '_' + 'LineSeg_Ini.npy')
LineSegFin = np.load(Case + '_' + Iter + '_' + 'LineSeg_Fin.npy')
StateIni = np.load(Case + '_' + Iter + '_' + 'State_Ini.npy')
StateFin = np.load(Case + '_' + Iter + '_' + 'State_Fin.npy')
NumMT, NumActin = np.count_nonzero(StateIni == 1), np.count_nonzero(StateIni == 2)

# Define hexagonal coordinate system -- NumRows must be even to satisfy boundary conditions
NumColumns, NumRows, GridDist = BaseColumns * ResFac, BaseRows * ResFac, BaseGridDist / ResFac
TotalPts = int(NumColumns * NumRows)
Location = np.zeros((TotalPts, 3))
for i in range(TotalPts):
    Location[i, 0] = i
for i in range(NumRows):
    if i % 2 == 0:  # even rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            Location[j, 1], Location[j, 2] = j - i * NumColumns, i / (2 * math.tan(math.pi / 6))
    else:  # odd rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            Location[j, 1], Location[j, 2] = j + 0.5 - i * NumColumns, i / (2 * math.tan(math.pi / 6))

# Define neighbor relationship array using periodic boundary conditions
# Neighbor slots are 0-5 starting with top right and moving clockwise
NumNB, NBPartner = 6, [3, 4, 5, 0, 1, 2]
NB = np.zeros((TotalPts, NumNB), dtype=int)
for i in range(NumRows):
    if i == 0:  # boundary condition (bottom)
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB[j, 0] = Location[j, 0] + NumColumns
            NB[j, 2] = Location[j, 0] + (NumRows - 1) * NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 3] = Location[j, 0] + NumRows * NumColumns - 1
                NB[j, 4] = Location[j, 0] + NumColumns - 1
                NB[j, 5] = Location[j, 0] + 2 * NumColumns - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB[j, 1] = Location[j, 0] - NumColumns + 1
                NB[j, 3] = Location[j, 0] + (NumRows - 1) * NumColumns - 1
                NB[j, 4] = Location[j, 0] - 1
                NB[j, 5] = Location[j, 0] + NumColumns - 1
            else:
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 3] = Location[j, 0] + (NumRows - 1) * NumColumns - 1
                NB[j, 4] = Location[j, 0] - 1
                NB[j, 5] = Location[j, 0] + NumColumns - 1
    elif i == (NumRows - 1):  # boundary condition (top)
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB[j, 3] = Location[j, 0] - NumColumns
            NB[j, 5] = Location[j, 0] - (NumRows - 1) * NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB[j, 0] = Location[j, 0] - (NumRows - 1) * NumColumns + 1
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 2] = Location[j, 0] - NumColumns + 1
                NB[j, 4] = Location[j, 0] + NumColumns - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB[j, 0] = Location[j, 0] - NumRows * NumColumns + 1
                NB[j, 1] = Location[j, 0] - NumColumns + 1
                NB[j, 2] = Location[j, 0] - 2 * NumColumns + 1
                NB[j, 4] = Location[j, 0] - 1
            else:
                NB[j, 0] = Location[j, 0] - (NumRows - 1) * NumColumns + 1
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 2] = Location[j, 0] - NumColumns + 1
                NB[j, 4] = Location[j, 0] - 1
    elif i % 2 == 0:  # even rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB[j, 0] = Location[j, 0] + NumColumns
            NB[j, 2] = Location[j, 0] - NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 3] = Location[j, 0] - 1
                NB[j, 4] = Location[j, 0] + NumColumns - 1
                NB[j, 5] = Location[j, 0] + 2 * NumColumns - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB[j, 1] = Location[j, 0] - NumColumns + 1
                NB[j, 3] = Location[j, 0] - NumColumns - 1
                NB[j, 4] = Location[j, 0] - 1
                NB[j, 5] = Location[j, 0] + NumColumns - 1
            else:
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 3] = Location[j, 0] - NumColumns - 1
                NB[j, 4] = Location[j, 0] - 1
                NB[j, 5] = Location[j, 0] + NumColumns - 1
    else:  # odd rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB[j, 3] = Location[j, 0] - NumColumns
            NB[j, 5] = Location[j, 0] + NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB[j, 0] = Location[j, 0] + NumColumns + 1
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 2] = Location[j, 0] - NumColumns + 1
                NB[j, 4] = Location[j, 0] + NumColumns - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB[j, 0] = Location[j, 0] + 1
                NB[j, 1] = Location[j, 0] - NumColumns + 1
                NB[j, 2] = Location[j, 0] - 2 * NumColumns + 1
                NB[j, 4] = Location[j, 0] - 1
            else:
                NB[j, 0] = Location[j, 0] + NumColumns + 1
                NB[j, 1] = Location[j, 0] + 1
                NB[j, 2] = Location[j, 0] - NumColumns + 1
                NB[j, 4] = Location[j, 0] - 1


# Define function for multiple degrees of neighbor separation
def nbt(point, slot, tier):  # point / slot / tier
    if tier == 0:
        return NB[point, slot]
    else:
        return NB[nbt(point, slot, tier - 1), slot]


# Define annulus relationship array using periodic boundary conditions
AnnLi = []
for i in range(NumAnn):
    AnnLi.append([])
for i in range(NumAnn):
    for j in range(TotalPts):
        AnnLi[i].append([])
Loc = np.zeros((NumAnn, NumNB), dtype=int)
for i in range(TotalPts):
    Loc[:] = 0
    for j in range(NumAnn):
        if j == 0:
            for k in range(NumNB):
                AnnLi[j][i].append(NB[i, k])
        else:
            for k in range(NumNB):
                Loc[j, k] = nbt(i, k, j)
                AnnLi[j][i].append(Loc[j, k])
                for m in range(j):
                    if k <= 3:
                        AnnLi[j][i].append(nbt(Loc[j, k], k + 2, m))
                    else:
                        AnnLi[j][i].append(nbt(Loc[j, k], k - 4, m))

# Measure initial distribution analytics
MaxFilament = np.zeros(NumAnn)
CtIniMT, CtIniAct = np.zeros((TotalPts, NumAnn), dtype=int), np.zeros((TotalPts, NumAnn), dtype=int)
CoCtIniMT, CoCtIniAct = np.zeros((TotalPts, NumAnn), dtype=int), np.zeros((TotalPts, NumAnn), dtype=int)
RadIniMT, RadIniAct = np.zeros((TotalPts, NumAnn)), np.zeros((TotalPts, NumAnn))
CoIniMT, CoIniAct = np.zeros((TotalPts, NumAnn)), np.zeros((TotalPts, NumAnn))
RadIniMTSum, RadIniActSum = np.zeros(NumAnn), np.zeros(NumAnn)
CoIniMTSum, CoIniActSum = np.zeros(NumAnn), np.zeros(NumAnn)
RadIniMTAvg, CoIniMTAvg = np.zeros(NumAnn), np.zeros(NumAnn)
RadIniActAvg, CoIniActAvg = np.zeros(NumAnn), np.zeros(NumAnn)
State = StateIni
for i in range(NumAnn):
    MaxFilament[i] = (i + 1) * NumNB
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            for k in range(len(AnnLi[j][i])):
                if State[AnnLi[j][i][k]] == 1:
                    CtIniMT[i, j] += 1
                elif State[AnnLi[j][i][k]] == 2:
                    CoCtIniMT[i, j] += 1
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            for k in range(len(AnnLi[j][i])):
                if State[AnnLi[j][i][k]] == 2:
                    CtIniAct[i, j] += 1
                elif State[AnnLi[j][i][k]] == 1:
                    CoCtIniAct[i, j] += 1
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            RadIniMT[i, j] = CtIniMT[i, j] / MaxFilament[j]
            CoIniMT[i, j] = CoCtIniMT[i, j] / MaxFilament[j]
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            RadIniAct[i, j] = CtIniAct[i, j] / MaxFilament[j]
            CoIniAct[i, j] = CoCtIniAct[i, j] / MaxFilament[j]
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            RadIniMTSum[j] += RadIniMT[i, j]
            CoIniMTSum[j] += CoIniMT[i, j]
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            RadIniActSum[j] += RadIniAct[i, j]
            CoIniActSum[j] += CoIniAct[i, j]
RadIniMTAvg = (RadIniMTSum / NumMT) / RatioMT if NumMT > 0 else 0
CoIniMTAvg = (CoIniMTSum / NumMT) / RatioActin if RatioActin > 0 else 0
RadIniActAvg = (RadIniActSum / NumActin) / RatioActin if NumActin > 0 else 0
CoIniActAvg = (CoIniActSum / NumActin) / RatioMT if NumActin > 0 else 0

# Measure post-simulation distribution analytics
CtFinMT, CtFinAct = np.zeros((TotalPts, NumAnn), dtype=int), np.zeros((TotalPts, NumAnn), dtype=int)
CoCtFinMT, CoCtFinAct = np.zeros((TotalPts, NumAnn), dtype=int), np.zeros((TotalPts, NumAnn), dtype=int)
RadFinMT, RadFinAct = np.zeros((TotalPts, NumAnn)), np.zeros((TotalPts, NumAnn))
CoFinMT, CoFinAct = np.zeros((TotalPts, NumAnn)), np.zeros((TotalPts, NumAnn))
RadFinMTSum, RadFinActSum = np.zeros(NumAnn), np.zeros(NumAnn)
CoFinMTSum, CoFinActSum = np.zeros(NumAnn), np.zeros(NumAnn)
RadFinMTAvg, CoFinMTAvg = np.zeros(NumAnn), np.zeros(NumAnn)
RadFinActAvg, CoFinActAvg = np.zeros(NumAnn), np.zeros(NumAnn)
MTDataSet, ActDataSet = np.zeros(NumAnn), np.zeros(NumAnn)
MTCoDataSet, ActCoDataSet = np.zeros(NumAnn), np.zeros(NumAnn)
State = StateFin
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            for k in range(len(AnnLi[j][i])):
                if State[AnnLi[j][i][k]] == 1:
                    CtFinMT[i, j] += 1
                elif State[AnnLi[j][i][k]] == 2:
                    CoCtFinMT[i, j] += 1
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            for k in range(len(AnnLi[j][i])):
                if State[AnnLi[j][i][k]] == 2:
                    CtFinAct[i, j] += 1
                elif State[AnnLi[j][i][k]] == 1:
                    CoCtFinAct[i, j] += 1
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            RadFinMT[i, j] = CtFinMT[i, j] / MaxFilament[j]
            CoFinMT[i, j] = CoCtFinMT[i, j] / MaxFilament[j]
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            RadFinAct[i, j] = CtFinAct[i, j] / MaxFilament[j]
            CoFinAct[i, j] = CoCtFinAct[i, j] / MaxFilament[j]
for i in range(TotalPts):
    if State[i] == 1:  # MT
        for j in range(NumAnn):
            RadFinMTSum[j] += RadFinMT[i, j]
            CoFinMTSum[j] += CoFinMT[i, j]
    elif State[i] == 2:  # actin
        for j in range(NumAnn):
            RadFinActSum[j] += RadFinAct[i, j]
            CoFinActSum[j] += CoFinAct[i, j]

MaskCtFinMT = np.ma.masked_equal(CtFinMT, 0)
MaskCtFinAct = np.ma.masked_equal(CtFinAct, 0)

MaskCoCtFinMT = np.ma.masked_equal(CoCtFinMT, 0)
MaskCoCtFinAct = np.ma.masked_equal(CoCtFinAct, 0)

for i in range(NumAnn):
    MTDataSet[i] = np.std(MaskCtFinMT[:, i]) / MaxFilament[i]
    ActDataSet[i] = np.std(MaskCtFinAct[:, i]) / MaxFilament[i]
    MTCoDataSet[i] = np.std(MaskCoCtFinMT[:, i]) / MaxFilament[i]
    ActCoDataSet[i] = np.std(MaskCoCtFinAct[:, i]) / MaxFilament[i]

RadFinMTAvg = (RadFinMTSum / NumMT) / RatioMT if NumMT > 0 else 0
CoFinMTAvg = (CoFinMTSum / NumMT) / RatioActin if RatioActin > 0 else 0
RadFinActAvg = (RadFinActSum / NumActin) / RatioActin if NumActin > 0 else 0
CoFinActAvg = (CoFinActSum / NumActin) / RatioMT if NumActin > 0 else 0

RadMTAvgDiff, RadActAvgDiff = RadFinMTAvg - RadIniMTAvg, RadFinActAvg - RadIniActAvg
CoMTAvgDiff, CoActAvgDiff = CoFinMTAvg - CoIniMTAvg, CoFinActAvg - CoIniActAvg

# Plots
plot.figure(figsize=(8, 8))
ax1 = plot.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=1, colspan=1)
ax2 = plot.subplot2grid(shape=(2, 2), loc=(0, 1), rowspan=1, colspan=1)
ax3 = plot.subplot2grid(shape=(2, 2), loc=(1, 1), rowspan=1, colspan=1)
ax4 = plot.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=1)

# Initial condition
ax1.set_title("Initial Condition", fontsize=14, fontweight='bold')
ax1.set_xlabel('Microns (μm)', fontsize=14), ax1.set_ylabel('Microns (μm)', fontsize=14)
ax1.set_xlim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
ax1.set_ylim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
MTLinesIni, ActLinesIni = [], []
for i in range(TotalPts):
    if StateIni[i] == 1:
        MTLinesIni.append(LineSegIni[i])
    elif StateIni[i] == 2:
        ActLinesIni.append(LineSegIni[i])
MTLineCollIni = collections.LineCollection(MTLinesIni, color="red", linewidth=1.2, zorder=2)
ActLineCollIni = collections.LineCollection(ActLinesIni, color="green", linewidth=1.2, zorder=1)
ax1.add_collection(MTLineCollIni), ax1.add_collection(ActLineCollIni)

# Condition after simulation
ax2.set_title("Condition After Simulation", fontsize=14, fontweight='bold')
ax2.set_xlabel('Microns (μm)', fontsize=14), ax2.set_ylabel('Microns (μm)', fontsize=14)
ax2.set_xlim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
ax2.set_ylim(-1 * ResFac * GridDist, (NumColumns + ResFac) * GridDist)
MTLinesFin, ActLinesFin = [], []
for i in range(TotalPts):
    if StateFin[i] == 1:
        MTLinesFin.append(LineSegFin[i])
    elif StateFin[i] == 2:
        ActLinesFin.append(LineSegFin[i])
MTLineCollFin = collections.LineCollection(MTLinesFin, color="red", linewidth=1.2, alpha=0.5, zorder=2)
ActLineCollFin = collections.LineCollection(ActLinesFin, color="green", linewidth=1.2, alpha=0.5, zorder=1)
ax2.add_collection(MTLineCollFin), ax2.add_collection(ActLineCollFin)

# Radial distribution
Rad = []
for i in range(NumAnn):
    Rad.append((i + 1) * BaseGridDist / ResFac)
if NumActin > 0:
    ax3.plot(Rad, RadActAvgDiff, c="green", zorder=1)
if NumMT > 0:
    ax3.plot(Rad, RadMTAvgDiff, c="red", zorder=2)
if NumActin > 0:
    ax3.scatter(Rad, RadActAvgDiff, marker="^", c="green", edgecolors='none', s=200, label="Actin, t$_f$ - t$_i$",
                zorder=3)
if NumMT > 0:
    ax3.scatter(Rad, RadMTAvgDiff, marker="^", c="red", edgecolors='none', s=200, label="MT, t$_f$ - t$_i$", zorder=4)
ax3.set_title("Radial Distribution", fontsize=14, fontweight='bold')
ax3.set_xlabel('r (μm)', fontsize=14), ax3.set_ylabel('g(r)', fontsize=14)
ax3.legend(ncol=2, loc=1)
ax3.set_ylim(-0.4, 1.4)

# Codistribution
ax4.scatter(Rad, CoMTAvgDiff, marker="^", c="red", edgecolors='none', s=200, label="MT, t$_f$ - t$_i$", zorder=4)
ax4.scatter(Rad, CoActAvgDiff, marker="^", c="green", edgecolors='none', s=200, label="Actin, t$_f$ - t$_i$", zorder=3)
ax4.plot(Rad, CoMTAvgDiff, c="red", zorder=2)
ax4.plot(Rad, CoActAvgDiff, c="green", zorder=1)
ax4.set_title("Codistribution", fontsize=14, fontweight='bold')
ax4.set_xlabel('r (μm)', fontsize=14), ax4.set_ylabel('g(r)', fontsize=14)
ax4.legend(ncol=2, loc=4)
ax4.set_ylim(-0.6, 0.2)

plot.tight_layout(), plot.savefig(Case + '_' + Iter + '_' + 'A_Results.pdf')

np.save(Case + '_' + Iter + '_MT_Rad', RadMTAvgDiff), np.save(Case + '_' + Iter + '_MT_Co', CoMTAvgDiff)
np.save(Case + '_' + Iter + '_Act_Rad', RadActAvgDiff), np.save(Case + '_' + Iter + '_Act_Co', CoActAvgDiff)
np.save(Case + '_' + Iter + '_MT_Rad_SD', MTDataSet), np.save(Case + '_' + Iter + '_Act_Rad_SD', ActDataSet)
np.save(Case + '_' + Iter + '_MT_Co_SD', MTCoDataSet), np.save(Case + '_' + Iter + '_Act_Co_SD', ActCoDataSet)
