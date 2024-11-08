# Import required packages
import numpy as np
import math as ma
import random as ra
import time

Start = time.time()

# Constants
FPerKin, FPerMyo, K0MT, K0Act, kBT, FiLen = 6, 3, 0.64, 1.28, 4, 5  # pN/fil, pN/fil, -, -, pN*nm, um
GamFilMT, GamFilAct, GamMotKin, GamMotMyo = 0.04e-3, 0.02e-3, 6e-3, 0.3e-3  # pN*s/nm, pN*s/nm, pN*s/nm/fil, pN*s/nm/fil
K0Sph, GamSph = 0.64, 0.04e-3  # pN/fil, pN*s/nm
BaseGridDist, ResFac = 5, 4  # um, -

# Material Inputs
CaseNum, CaseIter = 'SPH-K1-M1', '0'
AvgKinPerMT, KinRatAct, AvgMyoPerAct = 1, 0.5, 1  # -, -, -
GammaCLMT, GammaCLAct = 0, 0  # pN*s/nm/fil, pN*s/nm/fil
RatioMT, RatioAct, BaseColumns, BaseRows = 0.30, 0.25, 20, 23  # -, -, -, -
TotalTime = 6  # min

# Sphere Inputs
FSphMax, FSphFreq, FSphXfr = 100, 1 * (2 * ma.pi), 1  # pN, rad/s, -
SphTauIni = 5  # min

# Save input parameters
InputList = [RatioMT, RatioAct, BaseColumns, BaseRows, BaseGridDist, ResFac, TotalTime]
np.save(CaseNum + '_' + CaseIter + '_' + 'Input_Param', InputList)

# Define hexagonal coordinate system -- NumRows must be even to satisfy boundary conditions
NumCols, NumRows, GridDist = BaseColumns * ResFac, BaseRows * ResFac, BaseGridDist / ResFac
TotPts = NumCols * NumRows
Loc = np.zeros((TotPts, 3))
for i in range(TotPts):
    Loc[i, 0] = i
for i in range(NumRows):
    if i % 2 == 0:  # even rows
        for j in range(i * NumCols, i * NumCols + NumCols):
            Loc[j, 1], Loc[j, 2] = (j - i * NumCols) * GridDist, i / (2 * ma.tan(ma.pi / 6)) * GridDist
    else:  # odd rows
        for j in range(i * NumCols, i * NumCols + NumCols):
            Loc[j, 1], Loc[j, 2] = (j + 0.5 - i * NumCols) * GridDist, i / (2 * ma.tan(ma.pi / 6)) * GridDist

# Define first degree neighbor array using periodic boundary conditions
# Indexing starts with right and moves counterclockwise
NumNB1, NB1Partner = 6, [3, 4, 5, 0, 1, 2]
NB1 = np.zeros((TotPts, NumNB1), dtype=int)
for i in range(NumRows):
    if i == 0:  # boundary condition (bottom)
        for j in range(i * NumCols, i * NumCols + NumCols):
            NB1[j, 1], NB1[j, 5] = Loc[j, 0] + NumCols, Loc[j, 0] + (NumRows - 1) * NumCols
            if j == i * NumCols:  # boundary condition (left)
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] + 1, Loc[j, 0] + 2 * NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] + NumCols - 1, Loc[j, 0] + NumRows * NumCols - 1
            elif j == (i * NumCols + NumCols - 1):  # boundary condition (right)
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] - NumCols + 1, Loc[j, 0] + NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] - 1, Loc[j, 0] + (NumRows - 1) * NumCols - 1
            else:
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] + 1, Loc[j, 0] + NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] - 1, Loc[j, 0] + (NumRows - 1) * NumCols - 1
    elif i == (NumRows - 1):  # boundary condition (top)
        for j in range(i * NumCols, i * NumCols + NumCols):
            NB1[j, 2], NB1[j, 4] = Loc[j, 0] - (NumRows - 1) * NumCols, Loc[j, 0] - NumCols
            if j == i * NumCols:  # boundary condition (left)
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] + 1, Loc[j, 0] - (NumRows - 1) * NumCols + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] + NumCols - 1, Loc[j, 0] - NumCols + 1
            elif j == (i * NumCols + NumCols - 1):  # boundary condition (right)
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] - NumCols + 1, Loc[j, 0] - NumRows * NumCols + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] - 1, Loc[j, 0] - 2 * NumCols + 1
            else:
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] + 1, Loc[j, 0] - (NumRows - 1) * NumCols + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] - 1, Loc[j, 0] - NumCols + 1
    elif i % 2 == 0:  # even rows
        for j in range(i * NumCols, i * NumCols + NumCols):
            NB1[j, 1], NB1[j, 5] = Loc[j, 0] + NumCols, Loc[j, 0] - NumCols
            if j == i * NumCols:  # boundary condition (left)
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] + 1, Loc[j, 0] + 2 * NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] + NumCols - 1, Loc[j, 0] - 1
            elif j == (i * NumCols + NumCols - 1):  # boundary condition (right)
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] - NumCols + 1, Loc[j, 0] + NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] - 1, Loc[j, 0] - NumCols - 1
            else:
                NB1[j, 0], NB1[j, 2] = Loc[j, 0] + 1, Loc[j, 0] + NumCols - 1
                NB1[j, 3], NB1[j, 4] = Loc[j, 0] - 1, Loc[j, 0] - NumCols - 1
    else:  # odd rows
        for j in range(i * NumCols, i * NumCols + NumCols):
            NB1[j, 2], NB1[j, 4] = Loc[j, 0] + NumCols, Loc[j, 0] - NumCols
            if j == i * NumCols:  # boundary condition (left)
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] + 1, Loc[j, 0] + NumCols + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] + NumCols - 1, Loc[j, 0] - NumCols + 1
            elif j == (i * NumCols + NumCols - 1):  # boundary condition (right)
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] - NumCols + 1, Loc[j, 0] + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] - 1, Loc[j, 0] - 2 * NumCols + 1
            else:
                NB1[j, 0], NB1[j, 1] = Loc[j, 0] + 1, Loc[j, 0] + NumCols + 1
                NB1[j, 3], NB1[j, 5] = Loc[j, 0] - 1, Loc[j, 0] - NumCols + 1


# Define function to identify filaments with multiple degrees of neighbor separation
def nbt(point, slot, tier):  # point / slot / tier
    if tier == 0:
        return NB1[point, slot]
    else:
        return NB1[nbt(point, slot, tier - 1), slot]


# Define second, third, and fourth degree neighbor arrays -- indexing starts with right and moves counterclockwise
NumNB2, NumNB3, NumNB4 = NumNB1 * 2, NumNB1 * 3, NumNB1 * 4
NB2, NB3 = np.zeros((TotPts, NumNB2), dtype=int), np.zeros((TotPts, NumNB3), dtype=int)
NB4 = np.zeros((TotPts, NumNB4), dtype=int)
NB2Partner = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
NB3Partner = [9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8]
NB4Partner = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for i in range(TotPts):
    for j in range(NumNB2):
        if j % 2 == 0:
            NB2[i, j] = nbt(i, int(j / 2), 1)
        elif j % 2 == 1:
            NB2[i, j] = nbt(nbt(i, int((j - 1) / 2), 1), int(((j - 1) / 2 + 2) % NumNB1), 0)
    for j in range(NumNB3):
        if j % 3 == 0:
            NB3[i, j] = nbt(i, int(j / 3), 2)
        elif j % 3 == 1:
            NB3[i, j] = nbt(nbt(i, int(j / 3), 2), int(((j - 1) / 3 + 2) % NumNB1), 0)
        elif j % 3 == 2:
            NB3[i, j] = nbt(nbt(i, int(j / 3), 2), int(((j - 1) / 3 + 2) % NumNB1), 1)
    for j in range(NumNB4):
        if j % 4 == 0:
            NB4[i, j] = nbt(i, int(j / 4), 3)
        elif j % 4 == 1:
            NB4[i, j] = nbt(nbt(i, int(j / 4), 3), int(((j - 1) / 4 + 2) % NumNB1), 0)
        elif j % 4 == 2:
            NB4[i, j] = nbt(nbt(i, int(j / 4), 3), int(((j - 1) / 4 + 2) % NumNB1), 1)
        elif j % 4 == 3:
            NB4[i, j] = nbt(nbt(i, int(j / 4), 3), int(((j - 1) / 4 + 2) % NumNB1), 2)

# Define state array and load randomized initial conditions
RatioEmpty, PossibleStates = 1 - (RatioMT + RatioAct), [1, 2, 3]  # 1=microtubule (MT), 2=actin, 3=empty space
StateList = np.random.choice(PossibleStates, TotPts, p=[RatioMT, RatioAct, RatioEmpty])
State = np.asarray(StateList, dtype=int)
NumMT, NumAct = np.count_nonzero(State == 1), np.count_nonzero(State == 2)

# Define physical geometry of filaments and load randomized initial conditions -- [x1, y1, x2, y2]
EptIn, LineSegIn = np.zeros((TotPts, 4)), []
for i in range(TotPts):
    for j in range(4):
        EptIn[i, j] = 0
        LineSegIn.append([(EptIn[i, 0], EptIn[i, 1]), (EptIn[i, 2], EptIn[i, 3])])
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:
        GeoIni = np.random.uniform(low=0, high=2 * ma.pi)
        EptIn[i, 0], EptIn[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(GeoIni), Loc[i, 2] - 0.5 * FiLen * ma.sin(GeoIni)
        EptIn[i, 2], EptIn[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(GeoIni), Loc[i, 2] + 0.5 * FiLen * ma.sin(GeoIni)
        LineSegIn[i] = [(EptIn[i, 0], EptIn[i, 1]), (EptIn[i, 2], EptIn[i, 3])]

# Save initial conditions
np.save(CaseNum + '_' + CaseIter + '_' + 'State_Ini', State)
np.save(CaseNum + '_' + CaseIter + '_' + 'LineSeg_Ini', LineSegIn)

# Define force angle matrices
FAng1, TrackerFA1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
FAng2, TrackerFA2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
FAng3, TrackerFA3 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB3), dtype=int)
FAng4, TrackerFA4 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB4), dtype=int)
for i in range(TotPts):
    for j in range(NumNB1):
        if TrackerFA1[NB1[i, j], NB1Partner[j]] == 0:
            FAng1[i, NB1[i, j]], FAng1[NB1[i, j], i] = j * ma.pi / 3, ((j + 3) % 6) * ma.pi / 3
            TrackerFA1[i, j], TrackerFA1[NB1[i, j], NB1Partner[j]] = 1, 1
    for j in range(NumNB2):
        if TrackerFA2[NB2[i, j], NB2Partner[j]] == 0:
            FAng2[i, NB2[i, j]], FAng2[NB2[i, j], i] = j * ma.pi / 6, ((j + 6) % 12) * ma.pi / 6
            TrackerFA2[i, j], TrackerFA2[NB2[i, j], NB2Partner[j]] = 1, 1
    for j in range(NumNB3):
        if TrackerFA3[NB3[i, j], NB3Partner[j]] == 0:
            FAng3[i, NB3[i, j]], FAng3[NB3[i, j], i] = j * ma.pi / 9, ((j + 9) % 18) * ma.pi / 9
            TrackerFA3[i, j], TrackerFA3[NB3[i, j], NB3Partner[j]] = 1, 1
    for j in range(NumNB4):
        if TrackerFA4[NB4[i, j], NB4Partner[j]] == 0:
            FAng4[i, NB4[i, j]], FAng4[NB4[i, j], i] = j * ma.pi / 12, ((j + 12) % 24) * ma.pi / 12
            TrackerFA4[i, j], TrackerFA4[NB4[i, j], NB4Partner[j]] = 1, 1

# Define force magnitude and direction matrices and load initial conditions
NActKin1, NActKin2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
NActKin3, NActKin4 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
NMyo1, NMyo2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
NMyo3, NMyo4 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
FDir1, FDir2 = np.ones((TotPts, TotPts)), np.ones((TotPts, TotPts))
FDir3, FDir4 = np.ones((TotPts, TotPts)), np.ones((TotPts, TotPts))
FMag1, TrackerFM1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
FMag2, TrackerFM2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
FMag3, TrackerFM3 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB3), dtype=int)
FMag4, TrackerFM4 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB4), dtype=int)
for i in range(TotPts):
    if State[i] == 1 and AvgKinPerMT != 0:  # MT
        for j in range(NumNB1):
            if State[NB1[i, j]] == 1 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # MT
                if ra.random() < 0.5:  # recast force direction as negative
                    FDir1[i, NB1[i, j]], FDir1[NB1[i, j], i] = -1, -1
                Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                NActKin1[i, NB1[i, j]], NActKin1[NB1[i, j], i] = Poi, Poi
                FMag1[i, NB1[i, j]] = FDir1[i, NB1[i, j]] * NActKin1[i, NB1[i, j]] * FPerKin
                FMag1[NB1[i, j], i] = FDir1[NB1[i, j], i] * NActKin1[NB1[i, j], i] * FPerKin
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 1 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # MT
                Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                NActKin2[i, NB2[i, j]], NActKin2[NB2[i, j], i] = Poi, Poi
                FMag2[i, NB2[i, j]] = FDir2[i, NB2[i, j]] * NActKin2[i, NB2[i, j]] * FPerKin
                FMag2[NB2[i, j], i] = FDir2[NB2[i, j], i] * NActKin2[NB2[i, j], i] * FPerKin
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1
        for j in range(NumNB3):
            if State[NB3[i, j]] == 1 and TrackerFM3[NB3[i, j], NB3Partner[j]] == 0:  # MT
                Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                NActKin3[i, NB3[i, j]], NActKin3[NB3[i, j], i] = Poi, Poi
                FMag3[i, NB3[i, j]] = FDir3[i, NB3[i, j]] * NActKin3[i, NB3[i, j]] * FPerKin
                FMag3[NB3[i, j], i] = FDir3[NB3[i, j], i] * NActKin3[NB3[i, j], i] * FPerKin
                TrackerFM3[i, j], TrackerFM3[NB3[i, j], NB3Partner[j]] = 1, 1
        for j in range(NumNB4):
            if State[NB4[i, j]] == 1 and TrackerFM4[NB4[i, j], NB4Partner[j]] == 0:  # MT
                Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                NActKin4[i, NB4[i, j]], NActKin4[NB4[i, j], i] = Poi, Poi
                FMag4[i, NB4[i, j]] = FDir4[i, NB4[i, j]] * NActKin4[i, NB4[i, j]] * FPerKin
                FMag4[NB4[i, j], i] = FDir4[NB4[i, j], i] * NActKin4[NB4[i, j], i] * FPerKin
                TrackerFM4[i, j], TrackerFM4[NB4[i, j], NB4Partner[j]] = 1, 1
    elif State[i] == 2 and AvgMyoPerAct != 0:  # actin
        for j in range(NumNB1):
            if State[NB1[i, j]] == 2 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # actin
                if ra.random() < 0.5:  # recast force direction as negative
                    FDir1[i, NB1[i, j]], FDir1[NB1[i, j], i] = -1, -1
                Poi = np.random.poisson(lam=AvgMyoPerAct)
                NMyo1[i, NB1[i, j]], NMyo1[NB1[i, j], i] = Poi, Poi
                FMag1[i, NB1[i, j]] = FDir1[i, NB1[i, j]] * NMyo1[i, NB1[i, j]] * FPerMyo
                FMag1[NB1[i, j], i] = FDir1[NB1[i, j], i] * NMyo1[NB1[i, j], i] * FPerMyo
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 2 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # actin
                Poi = np.random.poisson(lam=AvgMyoPerAct)
                NMyo2[i, NB2[i, j]], NMyo2[NB2[i, j], i] = Poi, Poi
                FMag2[i, NB2[i, j]] = FDir2[i, NB2[i, j]] * NMyo2[i, NB2[i, j]] * FPerMyo
                FMag2[NB2[i, j], i] = FDir2[NB2[i, j], i] * NMyo2[NB2[i, j], i] * FPerMyo
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1
        for j in range(NumNB3):
            if State[NB3[i, j]] == 2 and TrackerFM3[NB3[i, j], NB3Partner[j]] == 0:  # actin
                Poi = np.random.poisson(lam=AvgMyoPerAct)
                NMyo3[i, NB3[i, j]], NMyo3[NB3[i, j], i] = Poi, Poi
                FMag3[i, NB3[i, j]] = FDir3[i, NB3[i, j]] * NMyo3[i, NB3[i, j]] * FPerMyo
                FMag3[NB3[i, j], i] = FDir3[NB3[i, j], i] * NMyo3[NB3[i, j], i] * FPerMyo
                TrackerFM3[i, j], TrackerFM3[NB3[i, j], NB3Partner[j]] = 1, 1
        for j in range(NumNB4):
            if State[NB4[i, j]] == 2 and TrackerFM4[NB4[i, j], NB4Partner[j]] == 0:  # actin
                Poi = np.random.poisson(lam=AvgMyoPerAct)
                NMyo4[i, NB4[i, j]], NMyo4[NB4[i, j], i] = Poi, Poi
                FMag4[i, NB4[i, j]] = FDir4[i, NB4[i, j]] * NMyo4[i, NB4[i, j]] * FPerMyo
                FMag4[NB4[i, j], i] = FDir4[NB4[i, j], i] * NMyo4[NB4[i, j], i] * FPerMyo
                TrackerFM4[i, j], TrackerFM4[NB4[i, j], NB4Partner[j]] = 1, 1

# Define net force and delta arrays
NetF, Del, NBF, FDtDel = np.zeros((TotPts, 2)), np.zeros((NumNB1, 2)), np.zeros((NumNB1, 2)), np.zeros((TotPts, NumNB1))
for i in range(NumNB1):  # define delta matrix for neighbor slots
    Del[i, 0], Del[i, 1] = ma.cos(i * ma.pi / 3), ma.sin(i * ma.pi / 3)
for i in range(TotPts):
    if (State[i] == 1 and AvgKinPerMT != 0) or (State[i] == 2 and AvgMyoPerAct != 0):  # MT or actin
        for j in range(NumNB1):
            NetF[i, 0] += FMag1[i, NB1[i, j]] * ma.cos(FAng1[i, NB1[i, j]])
            NetF[i, 1] += FMag1[i, NB1[i, j]] * ma.sin(FAng1[i, NB1[i, j]])
        for j in range(NumNB2):
            NetF[i, 0] += FMag2[i, NB2[i, j]] * ma.cos(FAng2[i, NB2[i, j]])
            NetF[i, 1] += FMag2[i, NB2[i, j]] * ma.sin(FAng2[i, NB2[i, j]])
        for j in range(NumNB3):
            NetF[i, 0] += FMag3[i, NB3[i, j]] * ma.cos(FAng3[i, NB3[i, j]])
            NetF[i, 1] += FMag3[i, NB3[i, j]] * ma.sin(FAng3[i, NB3[i, j]])
        for j in range(NumNB4):
            NetF[i, 0] += FMag4[i, NB4[i, j]] * ma.cos(FAng4[i, NB4[i, j]])
            NetF[i, 1] += FMag4[i, NB4[i, j]] * ma.sin(FAng4[i, NB4[i, j]])
        NBF[:, 0], NBF[:, 1] = NetF[i, 0], NetF[i, 1]  # define force matrix for neighbor slots
        for j in range(NumNB1):
            FDtDel[i, j] = np.dot(NBF[j], Del[j])  # calculate dot product

# Define probability matrices and load initial conditions
SameNB = np.zeros(TotPts)
NPasKin, NActKinTot, NMyoTot = np.zeros(TotPts), np.zeros(TotPts), np.zeros(TotPts)
KConst = np.zeros((TotPts, NumNB1))
IndProb, NumPossibleMoves, CombProb = np.zeros((TotPts, TotPts)), TotPts * 3, np.zeros((TotPts, TotPts))
AllMoveIDs = np.arange(NumPossibleMoves).reshape((TotPts, 3))  # array of all MoveIDs
AllMoveIDsList = [i for i in range(NumPossibleMoves)]  # list of all MoveIDs
ProbAllMoveIDs, NormProbAllMoveIDs = np.zeros((TotPts, 3)), np.zeros((TotPts, 3))
for i in range(TotPts):
    # Count number of same neighbors
    if State[i] == 1 or State[i] == 2:  # MT or actin
        for j in range(NumNB1):
            if State[i] == State[NB1[i, j]]:
                SameNB[i] += 1
        for j in range(NumNB2):
            if State[i] == State[NB2[i, j]]:
                SameNB[i] += 1
        for j in range(NumNB3):
            if State[i] == State[NB3[i, j]]:
                SameNB[i] += 1
        for j in range(NumNB4):
            if State[i] == State[NB4[i, j]]:
                SameNB[i] += 1
    # Calculate rate constants -- k = k0/friction + VDtDel / s
    if State[i] == 1:  # MT
        for j in range(NumNB1):
            NActKinTot[i] += NActKin1[i, NB1[i, j]]
        for j in range(NumNB2):
            NActKinTot[i] += NActKin2[i, NB2[i, j]]
        for j in range(NumNB3):
            NActKinTot[i] += NActKin3[i, NB3[i, j]]
        for j in range(NumNB4):
            NActKinTot[i] += NActKin4[i, NB4[i, j]]
        NPasKin[i] = np.random.poisson(lam=(AvgKinPerMT * (1 - KinRatAct)))
        for j in range(NumNB1):
            # Friction = 1 + (GammaActMot*N + GammaPasMot*N)/GammaFil + (GammaCL*N)/GammaFil
            FrictionMT = 1 + (GamMotKin * (NActKinTot[i] + NPasKin[i] * SameNB[i])) / GamFilMT + (
                        GammaCLMT * SameNB[i]) / GamFilMT
            # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
            if FDtDel[i, j] > 0:
                VDtDelMT = FDtDel[i, j] / (
                            GamFilMT + GamMotKin * (NActKinTot[i] + NPasKin[i] * SameNB[i]) + GammaCLMT * SameNB[i])
            else:
                VDtDelMT = 0
            # K = k0/friction + VDtDel / s
            KConst[i, j] = K0MT / FrictionMT + VDtDelMT / (GridDist * 1000)
    elif State[i] == 2:  # actin
        for j in range(NumNB1):
            NMyoTot[i] += NMyo1[i, NB1[i, j]]
        for j in range(NumNB2):
            NMyoTot[i] += NMyo2[i, NB2[i, j]]
        for j in range(NumNB3):
            NMyoTot[i] += NMyo3[i, NB3[i, j]]
        for j in range(NumNB4):
            NMyoTot[i] += NMyo4[i, NB4[i, j]]
        for j in range(NumNB1):
            # Friction = 1 + (GammaMot*N)/GammaFil + (GammaCL*N)/GammaFil
            FrictionAct = 1 + (GamMotMyo * NMyoTot[i]) / GamFilAct + (GammaCLAct * SameNB[i]) / GamFilAct
            # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
            if FDtDel[i, j] > 0:
                VDtDelAct = FDtDel[i, j] / (GamFilAct + GamMotMyo * NMyoTot[i] + GammaCLAct * SameNB[i])
            else:
                VDtDelAct = 0
            # K = k0/friction + VDtDel / s
            KConst[i, j] = K0Act / FrictionAct + VDtDelAct / (GridDist * 1000)
    elif State[i] == 3:  # empty space
        for j in range(NumNB1):
            KConst[i, j] = 0
# Calculate individual probabilities -- P = k/sum(k)
for i in range(TotPts):
    for j in range(NumNB1):
        if State[i] == 3 and State[NB1[i, j]] != 3:
            IndProb[i, NB1[i, j]] = 1
        else:
            IndProb[i, NB1[i, j]] = KConst[i, j] / KConst.sum()
# Combined probability matrix
for i in range(TotPts):
    for j in range(NumNB1):
        CombProb[i, NB1[i, j]] = IndProb[i, NB1[i, j]] * IndProb[NB1[i, j], i]
# Raw probabilities of each MoveID
for i in range(TotPts):
    ProbAllMoveIDs[i, 0] = CombProb[i, NB1[i, 1]]  # move between 1-4 neighbor relationship
    ProbAllMoveIDs[i, 1] = CombProb[i, NB1[i, 0]]  # move between 0-3 neighbor relationship
    ProbAllMoveIDs[i, 2] = CombProb[i, NB1[i, 5]]  # move between 5-2 neighbor relationship

# Run simulation
NormProbNestList, NormProbList, PtMv, PtAf = [], [], [], []
DoSomethingIDList, DoSomethingProbList, Tau = [1, 0], [], 0
SphLo, SphXDisp, SphYDisp, SphAf = 0, [0], [0], []
AddSph, SphTauList, SphID = 0, [SphTauIni], 40
while Tau < (TotalTime * 60):
    # Add sphere at sphere initial tau
    if Tau >= (SphTauIni * 60) and AddSph == 0:
        EmptyPtsLstIni = []
        for i in range(TotPts):
            if State[i] == 3:
                EmptyPtsLstIni.append(i)
        SphLo = ra.choice(EmptyPtsLstIni)
        State[SphLo] = SphID
        # Update physical geometry of filaments to incorporate force conditions for filaments with applied forces
        EptInj, LineSegInj = np.zeros((TotPts, 4)), []
        for i in range(TotPts):
            for j in range(4):
                EptInj[i, j] = 0
                LineSegInj.append([(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])])
        for i in range(TotPts):
            if (State[i] == 1 and AvgKinPerMT != 0) or (State[i] == 2 and AvgMyoPerAct != 0):
                if NetF[i, 0] != 0 or NetF[i, 1] != 0:
                    EptInj[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
                    EptInj[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
                    EptInj[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
                    EptInj[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
                    LineSegInj[i] = [(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])]
        # Update physical geometry of filaments to incorporate alignment of unforced filaments with neighboring
        # filaments of different type (or random alignment if no neighbors)
        Ali, NBSh = np.zeros((TotPts, 2)), [0, 1, 2, 3, 4, 5]
        for i in range(TotPts):
            if State[i] == 1 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # MT
                j, TrackerAli = 0, 0
                ra.shuffle(NBSh)
                while j in range(len(NBSh)) and TrackerAli == 0:
                    if State[NB1[i, NBSh[j]]] == 2 and NetF[NB1[i, NBSh[j]]].any() != 0:  # actin
                        if NetF[NB1[i, NBSh[j]], 0] < 0.1:
                            MTAlign0, MTAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                        elif NetF[NB1[i, NBSh[j]], 1] < 0.1:
                            MTAlign0, MTAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                        else:
                            Select = np.random.uniform(low=0, high=1)
                            if Select <= 0.5:
                                MTAlign0 = NetF[NB1[i, NBSh[j]], 0] * (1 + ra.randrange(50, 76) / 100)
                                MTAlign1 = NetF[NB1[i, NBSh[j]], 1]
                            else:
                                MTAlign0 = NetF[NB1[i, NBSh[j]], 0]
                                MTAlign1 = NetF[NB1[i, NBSh[j]], 1] * (1 + ra.randrange(50, 76) / 100)
                        Ali[i, 0], Ali[i, 1] = MTAlign0, MTAlign1
                        EptInj[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        LineSegInj[i] = [(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])]
                        TrackerAli = 1
                    else:
                        j += 1
                if TrackerAli == 0:
                    Geo = np.random.uniform(low=0, high=2 * ma.pi)
                    EptInj[i, 0], EptInj[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(Geo), Loc[
                        i, 2] - 0.5 * FiLen * ma.sin(Geo)
                    EptInj[i, 2], EptInj[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(Geo), Loc[
                        i, 2] + 0.5 * FiLen * ma.sin(Geo)
                    LineSegInj[i] = [(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])]
            elif State[i] == 2 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # actin
                j, TrackerAli = 0, 0
                ra.shuffle(NBSh)
                while j in range(len(NBSh)) and TrackerAli == 0:
                    if State[NB1[i, NBSh[j]]] == 1 and NetF[NB1[i, NBSh[j]]].any() != 0:  # MT
                        if NetF[NB1[i, NBSh[j]], 0] < 0.1:
                            ActAlign0, ActAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                        elif NetF[NB1[i, NBSh[j]], 1] < 0.1:
                            ActAlign0, ActAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                        else:
                            Select = np.random.uniform(low=0, high=1)
                            if Select <= 0.5:
                                ActAlign0 = NetF[NB1[i, NBSh[j]], 0] * (1 + ra.randrange(50, 76) / 100)
                                ActAlign1 = NetF[NB1[i, NBSh[j]], 1]
                            else:
                                ActAlign0 = NetF[NB1[i, NBSh[j]], 0]
                                ActAlign1 = NetF[NB1[i, NBSh[j]], 1] * (1 + ra.randrange(50, 76) / 100)
                        Ali[i, 0], Ali[i, 1] = ActAlign0, ActAlign1
                        EptInj[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        EptInj[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                        LineSegInj[i] = [(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])]
                        TrackerAli = 1
                    else:
                        j += 1
                if TrackerAli == 0:
                    Geo = np.random.uniform(low=0, high=2 * ma.pi)
                    EptInj[i, 0], EptInj[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(Geo), Loc[
                        i, 2] - 0.5 * FiLen * ma.sin(Geo)
                    EptInj[i, 2], EptInj[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(Geo), Loc[
                        i, 2] + 0.5 * FiLen * ma.sin(Geo)
                    LineSegInj[i] = [(EptInj[i, 0], EptInj[i, 1]), (EptInj[i, 2], EptInj[i, 3])]
        # Save injection conditions
        np.save(CaseNum + '_' + CaseIter + '_' + 'State_Inj', State)
        np.save(CaseNum + '_' + CaseIter + '_' + 'LineSeg_Inj', LineSegInj)
        # Mark task completed
        AddSph = 1
    # Update sphere forces and probabilities
    if Tau >= (SphTauIni * 60):
        # Collect list of sphere neighbors 1-4
        SphAf.clear()
        if SphLo not in SphAf:
            SphAf.append(SphLo)
        for j in range(NumNB1):
            if NB1[SphLo, j] not in SphAf:
                SphAf.append(NB1[SphLo, j])
        for j in range(NumNB2):
            if NB2[SphLo, j] not in SphAf:
                SphAf.append(NB2[SphLo, j])
        for j in range(NumNB3):
            if NB3[SphLo, j] not in SphAf:
                SphAf.append(NB3[SphLo, j])
        for j in range(NumNB4):
            if NB4[SphLo, j] not in SphAf:
                SphAf.append(NB4[SphLo, j])
        # Reset entries
        for i in range(len(SphAf)):
            NetF[SphAf[i]] = 0  # reset entries
            FDtDel[SphAf[i]] = 0  # reset entries
        # Update filament network forces
        for i in range(len(SphAf)):
            if (State[SphAf[i]] == 1 and AvgKinPerMT != 0) or (State[SphAf[i]] == 2 and AvgMyoPerAct != 0):  # MT or actin
                for j in range(NumNB1):
                    NetF[SphAf[i], 0] += FMag1[SphAf[i], NB1[SphAf[i], j]] * ma.cos(
                        FAng1[SphAf[i], NB1[SphAf[i], j]])
                    NetF[SphAf[i], 1] += FMag1[SphAf[i], NB1[SphAf[i], j]] * ma.sin(
                        FAng1[SphAf[i], NB1[SphAf[i], j]])
                for j in range(NumNB2):
                    NetF[SphAf[i], 0] += FMag2[SphAf[i], NB2[SphAf[i], j]] * ma.cos(
                        FAng2[SphAf[i], NB2[SphAf[i], j]])
                    NetF[SphAf[i], 1] += FMag2[SphAf[i], NB2[SphAf[i], j]] * ma.sin(
                        FAng2[SphAf[i], NB2[SphAf[i], j]])
                for j in range(NumNB3):
                    NetF[SphAf[i], 0] += FMag3[SphAf[i], NB3[SphAf[i], j]] * ma.cos(
                        FAng3[SphAf[i], NB3[SphAf[i], j]])
                    NetF[SphAf[i], 1] += FMag3[SphAf[i], NB3[SphAf[i], j]] * ma.sin(
                        FAng3[SphAf[i], NB3[SphAf[i], j]])
                for j in range(NumNB4):
                    NetF[SphAf[i], 0] += FMag4[SphAf[i], NB4[SphAf[i], j]] * ma.cos(
                        FAng4[SphAf[i], NB4[SphAf[i], j]])
                    NetF[SphAf[i], 1] += FMag4[SphAf[i], NB4[SphAf[i], j]] * ma.sin(
                        FAng4[SphAf[i], NB4[SphAf[i], j]])
        # Update sphere force
        NetF[SphLo, 0] = FSphMax * ma.sin(FSphFreq * (Tau - SphTauIni * 60))
        NetF[SphLo, 1] = 0
        # Apply sphere force to neighboring filaments
        if NetF[SphLo, 0] > 0 and State[NB1[SphLo, 0]] == 1:  # MT, right
            NetF[NB1[SphLo, 0], 0] += NetF[SphLo, 0] * FSphXfr
            if GammaCLMT > 0 or KinRatAct < 1:
                for j in range(NumNB1):
                    if State[NB1[NB1[SphLo, 0], j]] == 1:
                        NetF[NB1[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB2):
                    if State[NB2[NB1[SphLo, 0], j]] == 1:
                        NetF[NB2[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB3):
                    if State[NB3[NB1[SphLo, 0], j]] == 1:
                        NetF[NB3[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
        elif NetF[SphLo, 0] < 0 and State[NB1[SphLo, 3]] == 1:  # MT, left
            NetF[NB1[SphLo, 3], 0] += NetF[SphLo, 0] * FSphXfr
            if GammaCLMT > 0 or KinRatAct < 1:
                for j in range(NumNB1):
                    if State[NB1[NB1[SphLo, 3], j]] == 1:
                        NetF[NB1[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB2):
                    if State[NB2[NB1[SphLo, 3], j]] == 1:
                        NetF[NB2[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB3):
                    if State[NB3[NB1[SphLo, 3], j]] == 1:
                        NetF[NB3[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
        elif NetF[SphLo, 0] > 0 and State[NB1[SphLo, 0]] == 2:  # actin, right
            NetF[NB1[SphLo, 0], 0] += NetF[SphLo, 0] * FSphXfr
            if GammaCLAct > 0:
                for j in range(NumNB1):
                    if State[NB1[NB1[SphLo, 0], j]] == 2:
                        NetF[NB1[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB2):
                    if State[NB2[NB1[SphLo, 0], j]] == 2:
                        NetF[NB2[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB3):
                    if State[NB3[NB1[SphLo, 0], j]] == 2:
                        NetF[NB3[NB1[SphLo, 0], j], 0] += NetF[SphLo, 0] * FSphXfr
        elif NetF[SphLo, 0] < 0 and State[NB1[SphLo, 3]] == 2:  # actin, left
            NetF[NB1[SphLo, 3], 0] += NetF[SphLo, 0] * FSphXfr
            if GammaCLAct > 0:
                for j in range(NumNB1):
                    if State[NB1[NB1[SphLo, 3], j]] == 2:
                        NetF[NB1[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB2):
                    if State[NB2[NB1[SphLo, 3], j]] == 2:
                        NetF[NB2[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
                for j in range(NumNB3):
                    if State[NB3[NB1[SphLo, 3], j]] == 2:
                        NetF[NB3[NB1[SphLo, 3], j], 0] += NetF[SphLo, 0] * FSphXfr
        for i in range(len(SphAf)):
            NBF[:, 0], NBF[:, 1] = NetF[SphAf[i], 0], NetF[SphAf[i], 1]  # define force matrix
            for j in range(NumNB1):
                FDtDel[SphAf[i], j] = np.dot(NBF[j], Del[j])  # calculate dot product
        # Calculate rate constants -- k = k0/friction + VDtDel / s
        for i in range(len(SphAf)):
            if State[SphAf[i]] == 1:  # MT
                for j in range(NumNB1):
                    # Friction = 1 + (GammaActMot*N + GammaPasMot*N)/GammaFil + (GammaCL*N)/GammaFil
                    FrictionMT = 1 + (
                            GamMotKin * (NActKinTot[SphAf[i]] + NPasKin[SphAf[i]] * SameNB[SphAf[i]])) / GamFilMT + (
                                         GammaCLMT * SameNB[SphAf[i]]) / GamFilMT
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[SphAf[i], j] > 0:
                        VDtDelMT = FDtDel[SphAf[i], j] / (GamFilMT + GamMotKin * (
                                NActKinTot[SphAf[i]] + NPasKin[SphAf[i]] * SameNB[SphAf[i]]) + GammaCLMT * SameNB[
                                                              SphAf[i]])
                    else:
                        VDtDelMT = 0
                    # K = k0/friction + VDtDel / s
                    KConst[SphAf[i], j] = K0MT / FrictionMT + VDtDelMT / (GridDist * 1000)
            elif State[SphAf[i]] == 2:  # actin
                for j in range(NumNB1):
                    # Friction = 1 + (GammaMot*N)/GammaFil + (GammaCL*N)/GammaFil
                    FrictionAct = 1 + (GamMotMyo * NMyoTot[SphAf[i]]) / GamFilAct + (
                                GammaCLAct * SameNB[SphAf[i]]) / GamFilAct
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[SphAf[i], j] > 0:
                        VDtDelAct = FDtDel[SphAf[i], j] / (
                                    GamFilAct + GamMotMyo * NMyoTot[SphAf[i]] + GammaCLAct * SameNB[SphAf[i]])
                    else:
                        VDtDelAct = 0
                    # K = k0/friction + VDtDel / s
                    KConst[SphAf[i], j] = K0Act / FrictionAct + VDtDelAct / (GridDist * 1000)
            elif State[SphAf[i]] == 3:  # empty space
                for j in range(NumNB1):
                    KConst[SphAf[i], j] = 0
            elif State[SphAf[i]] == SphID:  # sphere
                for j in range(NumNB1):
                    # Friction = 1
                    FrictionSph = 1
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[SphAf[i], j] > 0:
                        VDtDelSph = FDtDel[SphAf[i], j] / GamSph
                    else:
                        VDtDelSph = 0
                    # K = k0/friction + VDtDel / s
                    KConst[SphAf[i], j] = K0Sph / FrictionSph + VDtDelSph / (GridDist * 1000)
        # Individual probability matrix -- P = k/sum(k)
        for i in range(len(SphAf)):
            for j in range(NumNB1):
                if State[SphAf[i]] == 3 and State[NB1[SphAf[i], j]] != 3:
                    IndProb[SphAf[i], NB1[SphAf[i], j]] = 1
                else:
                    IndProb[SphAf[i], NB1[SphAf[i], j]] = KConst[SphAf[i], j] / KConst.sum()
        # Update combined probability matrices
        for i in range(len(SphAf)):
            for j in range(NumNB1):
                CombProb[SphAf[i], NB1[SphAf[i], j]] = IndProb[SphAf[i], NB1[SphAf[i], j]] * IndProb[
                    NB1[SphAf[i], j], SphAf[i]]
                CombProb[NB1[SphAf[i], j], SphAf[i]] = IndProb[SphAf[i], NB1[SphAf[i], j]] * IndProb[
                    NB1[SphAf[i], j], SphAf[i]]
        # Update raw probability array
        for i in range(len(SphAf)):
            ProbAllMoveIDs[SphAf[i], 0] = CombProb[SphAf[i], NB1[SphAf[i], 1]]
            ProbAllMoveIDs[SphAf[i], 1] = CombProb[SphAf[i], NB1[SphAf[i], 0]]
            ProbAllMoveIDs[SphAf[i], 2] = CombProb[SphAf[i], NB1[SphAf[i], 5]]
    # Check for do nothing
    DoSomethingMoveID = 1
    if np.sum(ProbAllMoveIDs) < 1:
        DoSomethingProb = np.sum(ProbAllMoveIDs)
        DoNothingProb = 1 - DoSomethingProb
        DoSomethingProbList = [DoSomethingProb, DoNothingProb]
        DoSomethingMoveID = np.random.choice(DoSomethingIDList, p=DoSomethingProbList)
    if DoSomethingMoveID == 1:
        # Normalize probabilities of each MoveID
        NormProbAllMoveIDs = ProbAllMoveIDs / np.sum(ProbAllMoveIDs)
        # Convert probability array to list
        NormProbNestList.clear(), NormProbList.clear()
        NormProbNestList = NormProbAllMoveIDs.tolist()
        for i in NormProbNestList:
            NormProbList.extend(i)
        # Select MoveID to occur and lookup corresponding PointIDs
        MoveID = np.random.choice(AllMoveIDsList, p=NormProbList)
        LookUpTup = np.nonzero(AllMoveIDs == MoveID)
        LookUp0, LookUp1 = LookUpTup[0], LookUpTup[1]
        LookUp0Int, LookUp1Int = LookUp0[0], LookUp1[0]
        PtMoveA = LookUp0Int
        if LookUp1Int == 0:
            PtMoveB = NB1[PtMoveA, 1]
        elif LookUp1Int == 1:
            PtMoveB = NB1[PtMoveA, 0]
        else:
            PtMoveB = NB1[PtMoveA, 5]
        # Move points
        State[PtMoveA], State[PtMoveB] = State[PtMoveB], State[PtMoveA]
        # Store lists of points moved and points affected
        PtMv.clear(), PtAf.clear()
        PtMv.append(PtMoveA), PtMv.append(PtMoveB)
        for i in range(len(PtMv)):
            if PtMv[i] not in PtAf:
                PtAf.append(PtMv[i])
            for j in range(NumNB1):
                if NB1[PtMv[i], j] not in PtAf:
                    PtAf.append(NB1[PtMv[i], j])
            for j in range(NumNB2):
                if NB2[PtMv[i], j] not in PtAf:
                    PtAf.append(NB2[PtMv[i], j])
            for j in range(NumNB3):
                if NB3[PtMv[i], j] not in PtAf:
                    PtAf.append(NB3[PtMv[i], j])
            for j in range(NumNB4):
                if NB4[PtMv[i], j] not in PtAf:
                    PtAf.append(NB4[PtMv[i], j])
        # Log sphere movement
        if State[PtMoveA] == SphID or State[PtMoveB] == SphID:
            SphTauList.append(Tau)
            if State[PtMoveB] == SphID:
                if LookUp1Int == 0:
                    XDisp = SphXDisp[len(SphXDisp) - 1] + ma.cos(ma.pi / 3) * GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1] + ma.sin(ma.pi / 3) * GridDist
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
                elif LookUp1Int == 1:
                    XDisp = SphXDisp[len(SphXDisp) - 1] + GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1]
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
                else:
                    XDisp = SphXDisp[len(SphXDisp) - 1] + ma.cos(ma.pi / 3) * GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1] - ma.sin(ma.pi / 3) * GridDist
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
            elif State[PtMoveA] == SphID:
                if LookUp1Int == 0:
                    XDisp = SphXDisp[len(SphXDisp) - 1] - ma.cos(ma.pi / 3) * GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1] - ma.sin(ma.pi / 3) * GridDist
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
                elif LookUp1Int == 1:
                    XDisp = SphXDisp[len(SphXDisp) - 1] - GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1]
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
                else:
                    XDisp = SphXDisp[len(SphXDisp) - 1] - ma.cos(ma.pi / 3) * GridDist
                    YDisp = SphYDisp[len(SphYDisp) - 1] + ma.sin(ma.pi / 3) * GridDist
                    SphXDisp.append(XDisp)
                    SphYDisp.append(YDisp)
        # Update force magnitude and direction matrices
        for i in range(len(PtMv)):  # reset entries
            NActKin1[PtMv[i], NB1[PtMv[i]]], NActKin1[NB1[PtMv[i]], PtMv[i]] = 0, 0
            NActKin2[PtMv[i], NB2[PtMv[i]]], NActKin2[NB2[PtMv[i]], PtMv[i]] = 0, 0
            NActKin3[PtMv[i], NB3[PtMv[i]]], NActKin3[NB3[PtMv[i]], PtMv[i]] = 0, 0
            NActKin4[PtMv[i], NB4[PtMv[i]]], NActKin4[NB4[PtMv[i]], PtMv[i]] = 0, 0
            NMyo1[PtMv[i], NB1[PtMv[i]]], NMyo1[NB1[PtMv[i]], PtMv[i]] = 0, 0
            NMyo2[PtMv[i], NB2[PtMv[i]]], NMyo2[NB2[PtMv[i]], PtMv[i]] = 0, 0
            NMyo3[PtMv[i], NB3[PtMv[i]]], NMyo3[NB3[PtMv[i]], PtMv[i]] = 0, 0
            NMyo4[PtMv[i], NB4[PtMv[i]]], NMyo4[NB4[PtMv[i]], PtMv[i]] = 0, 0
            FMag1[PtMv[i], NB1[PtMv[i]]], FMag1[NB1[PtMv[i]], PtMv[i]] = 0, 0
            FMag2[PtMv[i], NB2[PtMv[i]]], FMag2[NB2[PtMv[i]], PtMv[i]] = 0, 0
            FMag3[PtMv[i], NB3[PtMv[i]]], FMag3[NB3[PtMv[i]], PtMv[i]] = 0, 0
            FMag4[PtMv[i], NB4[PtMv[i]]], FMag4[NB4[PtMv[i]], PtMv[i]] = 0, 0
            FDir1[PtMv[i], NB1[PtMv[i]]], FDir1[NB1[PtMv[i]], PtMv[i]] = 1, 1
            FDir2[PtMv[i], NB2[PtMv[i]]], FDir2[NB2[PtMv[i]], PtMv[i]] = 1, 1
            FDir3[PtMv[i], NB3[PtMv[i]]], FDir3[NB3[PtMv[i]], PtMv[i]] = 1, 1
            FDir4[PtMv[i], NB4[PtMv[i]]], FDir4[NB4[PtMv[i]], PtMv[i]] = 1, 1
            TrackerFM1[PtMv[i]], TrackerFM2[PtMv[i]] = 0, 0
            TrackerFM3[PtMv[i]], TrackerFM4[PtMv[i]] = 0, 0
        for i in range(len(PtMv)):
            if State[PtMv[i]] == 1 and AvgKinPerMT != 0:  # MT
                for j in range(NumNB1):
                    if State[NB1[PtMv[i], j]] == 1 and TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] == 0:  # MT
                        if ra.random() < 0.5:  # recast force direction as negative
                            FDir1[PtMv[i], NB1[PtMv[i], j]], FDir1[NB1[PtMv[i], j], PtMv[i]] = -1, -1
                        Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                        NActKin1[PtMv[i], NB1[PtMv[i], j]], NActKin1[NB1[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag1[PtMv[i], NB1[PtMv[i], j]] = FDir1[PtMv[i], NB1[PtMv[i], j]] * NActKin1[
                            PtMv[i], NB1[PtMv[i], j]] * FPerKin
                        FMag1[NB1[PtMv[i], j], PtMv[i]] = FDir1[NB1[PtMv[i], j], PtMv[i]] * NActKin1[
                            NB1[PtMv[i], j], PtMv[i]] * FPerKin
                        TrackerFM1[PtMv[i], j], TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] = 1, 1
                for j in range(NumNB2):
                    if State[NB2[PtMv[i], j]] == 1 and TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] == 0:  # MT
                        Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                        NActKin2[PtMv[i], NB2[PtMv[i], j]], NActKin2[NB2[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag2[PtMv[i], NB2[PtMv[i], j]] = FDir2[PtMv[i], NB2[PtMv[i], j]] * NActKin2[
                            PtMv[i], NB2[PtMv[i], j]] * FPerKin
                        FMag2[NB2[PtMv[i], j], PtMv[i]] = FDir2[NB2[PtMv[i], j], PtMv[i]] * NActKin2[
                            NB2[PtMv[i], j], PtMv[i]] * FPerKin
                        TrackerFM2[PtMv[i], j], TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] = 1, 1
                for j in range(NumNB3):
                    if State[NB3[PtMv[i], j]] == 1 and TrackerFM3[NB3[PtMv[i], j], NB3Partner[j]] == 0:  # MT
                        Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                        NActKin3[PtMv[i], NB3[PtMv[i], j]], NActKin3[NB3[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag3[PtMv[i], NB3[PtMv[i], j]] = FDir3[PtMv[i], NB3[PtMv[i], j]] * NActKin3[
                            PtMv[i], NB3[PtMv[i], j]] * FPerKin
                        FMag3[NB3[PtMv[i], j], PtMv[i]] = FDir3[NB3[PtMv[i], j], PtMv[i]] * NActKin3[
                            NB3[PtMv[i], j], PtMv[i]] * FPerKin
                        TrackerFM3[PtMv[i], j], TrackerFM3[NB3[PtMv[i], j], NB3Partner[j]] = 1, 1
                for j in range(NumNB4):
                    if State[NB4[PtMv[i], j]] == 1 and TrackerFM4[NB4[PtMv[i], j], NB4Partner[j]] == 0:  # MT
                        Poi = np.random.poisson(lam=(AvgKinPerMT * KinRatAct))
                        NActKin4[PtMv[i], NB4[PtMv[i], j]], NActKin4[NB4[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag4[PtMv[i], NB4[PtMv[i], j]] = FDir4[PtMv[i], NB4[PtMv[i], j]] * NActKin4[
                            PtMv[i], NB4[PtMv[i], j]] * FPerKin
                        FMag4[NB4[PtMv[i], j], PtMv[i]] = FDir4[NB4[PtMv[i], j], PtMv[i]] * NActKin4[
                            NB4[PtMv[i], j], PtMv[i]] * FPerKin
                        TrackerFM4[PtMv[i], j], TrackerFM4[NB4[PtMv[i], j], NB4Partner[j]] = 1, 1
            elif State[PtMv[i]] == 2 and AvgMyoPerAct != 0:  # actin
                for j in range(NumNB1):
                    if State[NB1[PtMv[i], j]] == 2 and TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] == 0:  # actin
                        if ra.random() < 0.5:  # recast force direction as negative
                            FDir1[PtMv[i], NB1[PtMv[i], j]], FDir1[NB1[PtMv[i], j], PtMv[i]] = -1, -1
                        Poi = np.random.poisson(lam=AvgMyoPerAct)
                        NMyo1[PtMv[i], NB1[PtMv[i], j]], NMyo1[NB1[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag1[PtMv[i], NB1[PtMv[i], j]] = FDir1[PtMv[i], NB1[PtMv[i], j]] * NMyo1[
                            PtMv[i], NB1[PtMv[i], j]] * FPerMyo
                        FMag1[NB1[PtMv[i], j], PtMv[i]] = FDir1[NB1[PtMv[i], j], PtMv[i]] * NMyo1[
                            NB1[PtMv[i], j], PtMv[i]] * FPerMyo
                        TrackerFM1[PtMv[i], j], TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] = 1, 1
                for j in range(NumNB2):
                    if State[NB2[PtMv[i], j]] == 2 and TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] == 0:  # actin
                        Poi = np.random.poisson(lam=AvgMyoPerAct)
                        NMyo2[PtMv[i], NB2[PtMv[i], j]], NMyo2[NB2[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag2[PtMv[i], NB2[PtMv[i], j]] = FDir2[PtMv[i], NB2[PtMv[i], j]] * NMyo2[
                            PtMv[i], NB2[PtMv[i], j]] * FPerMyo
                        FMag2[NB2[PtMv[i], j], PtMv[i]] = FDir2[NB2[PtMv[i], j], PtMv[i]] * NMyo2[
                            NB2[PtMv[i], j], PtMv[i]] * FPerMyo
                        TrackerFM2[PtMv[i], j], TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] = 1, 1
                for j in range(NumNB3):
                    if State[NB3[PtMv[i], j]] == 2 and TrackerFM3[NB3[PtMv[i], j], NB3Partner[j]] == 0:  # actin
                        Poi = np.random.poisson(lam=AvgMyoPerAct)
                        NMyo3[PtMv[i], NB3[PtMv[i], j]], NMyo3[NB3[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag3[PtMv[i], NB3[PtMv[i], j]] = FDir3[PtMv[i], NB3[PtMv[i], j]] * NMyo3[
                            PtMv[i], NB3[PtMv[i], j]] * FPerMyo
                        FMag3[NB3[PtMv[i], j], PtMv[i]] = FDir3[NB3[PtMv[i], j], PtMv[i]] * NMyo3[
                            NB3[PtMv[i], j], PtMv[i]] * FPerMyo
                        TrackerFM3[PtMv[i], j], TrackerFM3[NB3[PtMv[i], j], NB3Partner[j]] = 1, 1
                for j in range(NumNB4):
                    if State[NB4[PtMv[i], j]] == 2 and TrackerFM4[NB4[PtMv[i], j], NB4Partner[j]] == 0:  # actin
                        Poi = np.random.poisson(lam=AvgMyoPerAct)
                        NMyo4[PtMv[i], NB4[PtMv[i], j]], NMyo4[NB4[PtMv[i], j], PtMv[i]] = Poi, Poi
                        FMag4[PtMv[i], NB4[PtMv[i], j]] = FDir4[PtMv[i], NB4[PtMv[i], j]] * NMyo4[
                            PtMv[i], NB4[PtMv[i], j]] * FPerMyo
                        FMag4[NB4[PtMv[i], j], PtMv[i]] = FDir4[NB4[PtMv[i], j], PtMv[i]] * NMyo4[
                            NB4[PtMv[i], j], PtMv[i]] * FPerMyo
                        TrackerFM4[PtMv[i], j], TrackerFM4[NB4[PtMv[i], j], NB4Partner[j]] = 1, 1
        # Update net force arrays
        for i in range(len(PtAf)):
            NetF[PtAf[i]] = 0  # reset entries
            FDtDel[PtAf[i]] = 0  # reset entries
        # Apply sphere force to neighboring filaments
        for i in range(len(PtMv)):
            if (State[PtMv[i]] == SphID) and Tau >= (SphTauIni * 60):  # sphere
                if NetF[PtMv[i], 0] > 0 and State[NB1[PtMv[i], 0]] == 1:  # MT, right
                    NetF[NB1[PtMv[i], 0], 0] += NetF[PtMv[i], 0] * FSphXfr
                    if GammaCLMT > 0 or KinRatAct < 1:
                        for j in range(NumNB1):
                            if State[NB1[NB1[PtMv[i], 0], j]] == 1:
                                NetF[NB1[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB2):
                            if State[NB2[NB1[PtMv[i], 0], j]] == 1:
                                NetF[NB2[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB3):
                            if State[NB3[NB1[PtMv[i], 0], j]] == 1:
                                NetF[NB3[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                elif NetF[PtMv[i], 0] < 0 and State[NB1[PtMv[i], 3]] == 1:  # MT, left
                    NetF[NB1[PtMv[i], 3], 0] += NetF[PtMv[i], 0] * FSphXfr
                    if GammaCLMT > 0 or KinRatAct < 1:
                        for j in range(NumNB1):
                            if State[NB1[NB1[PtMv[i], 3], j]] == 1:
                                NetF[NB1[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB2):
                            if State[NB2[NB1[PtMv[i], 3], j]] == 1:
                                NetF[NB2[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB3):
                            if State[NB3[NB1[PtMv[i], 3], j]] == 1:
                                NetF[NB3[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                elif NetF[PtMv[i], 0] > 0 and State[NB1[PtMv[i], 0]] == 2:  # act, right
                    NetF[NB1[PtMv[i], 0], 0] += NetF[PtMv[i], 0] * FSphXfr
                    if GammaCLAct > 0:
                        for j in range(NumNB1):
                            if State[NB1[NB1[PtMv[i], 0], j]] == 2:
                                NetF[NB1[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB2):
                            if State[NB2[NB1[PtMv[i], 0], j]] == 2:
                                NetF[NB2[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB3):
                            if State[NB3[NB1[PtMv[i], 0], j]] == 2:
                                NetF[NB3[NB1[PtMv[i], 0], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                elif NetF[PtMv[i], 0] < 0 and State[NB1[PtMv[i], 3]] == 2:  # act, left
                    NetF[NB1[PtMv[i], 3], 0] += NetF[PtMv[i], 0] * FSphXfr
                    if GammaCLAct > 0:
                        for j in range(NumNB1):
                            if State[NB1[NB1[PtMv[i], 3], j]] == 2:
                                NetF[NB1[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB2):
                            if State[NB2[NB1[PtMv[i], 3], j]] == 2:
                                NetF[NB2[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
                        for j in range(NumNB3):
                            if State[NB3[NB1[PtMv[i], 3], j]] == 2:
                                NetF[NB3[NB1[PtMv[i], 3], j], 0] += NetF[PtMv[i], 0] * FSphXfr
        for i in range(len(PtAf)):
            if (State[PtAf[i]] == 1 and AvgKinPerMT != 0) or (State[PtAf[i]] == 2 and AvgMyoPerAct != 0):  # MT or actin
                for j in range(NumNB1):
                    NetF[PtAf[i], 0] += FMag1[PtAf[i], NB1[PtAf[i], j]] * ma.cos(FAng1[PtAf[i], NB1[PtAf[i], j]])
                    NetF[PtAf[i], 1] += FMag1[PtAf[i], NB1[PtAf[i], j]] * ma.sin(FAng1[PtAf[i], NB1[PtAf[i], j]])
                for j in range(NumNB2):
                    NetF[PtAf[i], 0] += FMag2[PtAf[i], NB2[PtAf[i], j]] * ma.cos(FAng2[PtAf[i], NB2[PtAf[i], j]])
                    NetF[PtAf[i], 1] += FMag2[PtAf[i], NB2[PtAf[i], j]] * ma.sin(FAng2[PtAf[i], NB2[PtAf[i], j]])
                for j in range(NumNB3):
                    NetF[PtAf[i], 0] += FMag3[PtAf[i], NB3[PtAf[i], j]] * ma.cos(FAng3[PtAf[i], NB3[PtAf[i], j]])
                    NetF[PtAf[i], 1] += FMag3[PtAf[i], NB3[PtAf[i], j]] * ma.sin(FAng3[PtAf[i], NB3[PtAf[i], j]])
                for j in range(NumNB4):
                    NetF[PtAf[i], 0] += FMag4[PtAf[i], NB4[PtAf[i], j]] * ma.cos(FAng4[PtAf[i], NB4[PtAf[i], j]])
                    NetF[PtAf[i], 1] += FMag4[PtAf[i], NB4[PtAf[i], j]] * ma.sin(FAng4[PtAf[i], NB4[PtAf[i], j]])
                NBF[:, 0], NBF[:, 1] = NetF[PtAf[i], 0], NetF[PtAf[i], 1]  # define force matrix
                for j in range(NumNB1):
                    FDtDel[PtAf[i], j] = np.dot(NBF[j], Del[j])  # calculate dot product
            elif (State[PtAf[i]] == SphID) and Tau >= (SphTauIni * 60):  # sphere
                NetF[PtAf[i], 0] = FSphMax * ma.sin(FSphFreq * (Tau - SphTauIni * 60))
                NetF[PtAf[i], 1] = 0
                NBF[:, 0], NBF[:, 1] = NetF[PtAf[i], 0], NetF[PtAf[i], 1]  # define force matrix
                for j in range(NumNB1):
                    FDtDel[PtAf[i], j] = np.dot(NBF[j], Del[j])  # calculate dot product
        # Update individual probability matrix
        for i in range(len(PtAf)):
            SameNB[PtAf[i]] = 0  # reset entries
            NPasKin[PtAf[i]] = 0  # reset entries
            NActKinTot[PtAf[i]] = 0  # reset entries
            NMyoTot[PtAf[i]] = 0  # reset entries
            KConst[PtAf[i]] = 0  # reset entries
            IndProb[PtAf[i]] = 0  # reset entries
            if State[PtAf[i]] == 1 or State[PtAf[i]] == 2:  # MT or actin
                # Count number of same neighbors
                for j in range(NumNB1):
                    if State[PtAf[i]] == State[NB1[PtAf[i], j]]:
                        SameNB[PtAf[i]] += 1
                for j in range(NumNB2):
                    if State[PtAf[i]] == State[NB2[PtAf[i], j]]:
                        SameNB[PtAf[i]] += 1
                for j in range(NumNB3):
                    if State[PtAf[i]] == State[NB3[PtAf[i], j]]:
                        SameNB[PtAf[i]] += 1
                for j in range(NumNB4):
                    if State[PtAf[i]] == State[NB4[PtAf[i], j]]:
                        SameNB[PtAf[i]] += 1
            # Calculate rate constants -- k = k0/friction + VDtDel / s
            if State[PtAf[i]] == 1:  # MT
                for j in range(NumNB1):
                    NActKinTot[PtAf[i]] += NActKin1[PtAf[i], NB1[PtAf[i], j]]
                for j in range(NumNB2):
                    NActKinTot[PtAf[i]] += NActKin2[PtAf[i], NB2[PtAf[i], j]]
                for j in range(NumNB3):
                    NActKinTot[PtAf[i]] += NActKin3[PtAf[i], NB3[PtAf[i], j]]
                for j in range(NumNB4):
                    NActKinTot[PtAf[i]] += NActKin4[PtAf[i], NB4[PtAf[i], j]]
                NPasKin[PtAf[i]] = np.random.poisson(lam=(AvgKinPerMT * (1 - KinRatAct)))
                for j in range(NumNB1):
                    # Friction = 1 + (GammaActMot*N + GammaPasMot*N)/GammaFil + (GammaCL*N)/GammaFil
                    FrictionMT = 1 + (
                                GamMotKin * (NActKinTot[PtAf[i]] + NPasKin[PtAf[i]] * SameNB[PtAf[i]])) / GamFilMT + (
                                             GammaCLMT * SameNB[PtAf[i]]) / GamFilMT
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[PtAf[i], j] > 0:
                        VDtDelMT = FDtDel[PtAf[i], j] / (GamFilMT + GamMotKin * (
                                    NActKinTot[PtAf[i]] + NPasKin[PtAf[i]] * SameNB[PtAf[i]]) + GammaCLMT * SameNB[
                                                             PtAf[i]])
                    else:
                        VDtDelMT = 0
                    # K = k0/friction + VDtDel / s
                    KConst[PtAf[i], j] = K0MT / FrictionMT + VDtDelMT / (GridDist * 1000)
            elif State[PtAf[i]] == 2:  # actin
                for j in range(NumNB1):
                    NMyoTot[PtAf[i]] += NMyo1[PtAf[i], NB1[PtAf[i], j]]
                for j in range(NumNB2):
                    NMyoTot[PtAf[i]] += NMyo2[PtAf[i], NB2[PtAf[i], j]]
                for j in range(NumNB3):
                    NMyoTot[PtAf[i]] += NMyo3[PtAf[i], NB3[PtAf[i], j]]
                for j in range(NumNB4):
                    NMyoTot[PtAf[i]] += NMyo4[PtAf[i], NB4[PtAf[i], j]]
                for j in range(NumNB1):
                    # Friction = 1 + (GammaMot*N)/GammaFil + (GammaCL*N)/GammaFil
                    FrictionAct = 1 + (GamMotMyo * NMyoTot[PtAf[i]]) / GamFilAct + (
                            GammaCLAct * SameNB[PtAf[i]]) / GamFilAct
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[PtAf[i], j] > 0:
                        VDtDelAct = FDtDel[PtAf[i], j] / (
                                GamFilAct + GamMotMyo * NMyoTot[PtAf[i]] + GammaCLAct * SameNB[PtAf[i]])
                        # print(VDtDelAct)
                    else:
                        VDtDelAct = 0
                    # K = k0/friction + VDtDel / s
                    KConst[PtAf[i], j] = K0Act / FrictionAct + VDtDelAct / (GridDist * 1000)
            elif State[PtAf[i]] == 3:  # empty space
                for j in range(NumNB1):
                    KConst[PtAf[i], j] = 0
            elif State[PtAf[i]] == SphID:  # sphere
                for j in range(NumNB1):
                    # Friction = 1 + GammaMot/GammaFil + GammaCL/GammaFil
                    FrictionSph = 1
                    # VDtDel = FDtDel / (GammaFil + GammaMot*N + GammaCL*N)
                    if FDtDel[PtAf[i], j] > 0:
                        VDtDelSph = FDtDel[PtAf[i], j] / GamSph
                    else:
                        VDtDelSph = 0
                    # K = k0/friction + VDtDel / s
                    KConst[PtAf[i], j] = K0Sph / FrictionSph + VDtDelSph / (GridDist * 1000)
        # Individual probability matrix -- P = k/sum(k)
        for i in range(len(PtAf)):
            for j in range(NumNB1):
                if State[PtAf[i]] == 3 and State[NB1[PtAf[i], j]] != 3:
                    IndProb[PtAf[i], NB1[PtAf[i], j]] = 1
                else:
                    IndProb[PtAf[i], NB1[PtAf[i], j]] = KConst[PtAf[i], j] / KConst.sum()
        # Update combined probability matrices
        for i in range(len(PtAf)):
            for j in range(NumNB1):
                CombProb[PtAf[i], NB1[PtAf[i], j]] = IndProb[PtAf[i], NB1[PtAf[i], j]] * IndProb[
                    NB1[PtAf[i], j], PtAf[i]]
                CombProb[NB1[PtAf[i], j], PtAf[i]] = IndProb[PtAf[i], NB1[PtAf[i], j]] * IndProb[
                    NB1[PtAf[i], j], PtAf[i]]
        # Update raw probability array
        for i in range(len(PtAf)):
            ProbAllMoveIDs[PtAf[i], 0] = CombProb[PtAf[i], NB1[PtAf[i], 1]]
            ProbAllMoveIDs[PtAf[i], 1] = CombProb[PtAf[i], NB1[PtAf[i], 0]]
            ProbAllMoveIDs[PtAf[i], 2] = CombProb[PtAf[i], NB1[PtAf[i], 5]]
    # Calculate elapsed tau value
    Tau += np.random.exponential(1 / KConst.sum())

# Update physical geometry of filaments to incorporate force conditions for filaments with applied forces
Ept, LineSeg = np.zeros((TotPts, 4)), []
for i in range(TotPts):
    for j in range(4):
        Ept[i, j] = 0
        LineSeg.append([(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])])
for i in range(TotPts):
    if (State[i] == 1 and AvgKinPerMT != 0) or (State[i] == 2 and AvgMyoPerAct != 0):
        if NetF[i, 0] != 0 or NetF[i, 1] != 0:
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
# Update physical geometry of filaments to incorporate alignment of unforced filaments with neighboring filaments
# of different type (or random alignment if no neighbors)
Ali, NBSh = np.zeros((TotPts, 2)), [0, 1, 2, 3, 4, 5]
for i in range(TotPts):
    if State[i] == 1 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # MT
        j, TrackerAli = 0, 0
        ra.shuffle(NBSh)
        while j in range(len(NBSh)) and TrackerAli == 0:
            if State[NB1[i, NBSh[j]]] == 2 and NetF[NB1[i, NBSh[j]]].any() != 0:  # actin
                if NetF[NB1[i, NBSh[j]], 0] < 0.1:
                    MTAlign0, MTAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, NBSh[j]], 1] < 0.1:
                    MTAlign0, MTAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    Select = np.random.uniform(low=0, high=1)
                    if Select <= 0.5:
                        MTAlign0 = NetF[NB1[i, NBSh[j]], 0] * (1 + ra.randrange(50, 76) / 100)
                        MTAlign1 = NetF[NB1[i, NBSh[j]], 1]
                    else:
                        MTAlign0 = NetF[NB1[i, NBSh[j]], 0]
                        MTAlign1 = NetF[NB1[i, NBSh[j]], 1] * (1 + ra.randrange(50, 76) / 100)
                Ali[i, 0], Ali[i, 1] = MTAlign0, MTAlign1
                Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
                TrackerAli = 1
            else:
                j += 1
        if TrackerAli == 0:
            Geo = np.random.uniform(low=0, high=2 * ma.pi)
            Ept[i, 0], Ept[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(Geo), Loc[i, 2] - 0.5 * FiLen * ma.sin(Geo)
            Ept[i, 2], Ept[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(Geo), Loc[i, 2] + 0.5 * FiLen * ma.sin(Geo)
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
    elif State[i] == 2 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # actin
        j, TrackerAli = 0, 0
        ra.shuffle(NBSh)
        while j in range(len(NBSh)) and TrackerAli == 0:
            if State[NB1[i, NBSh[j]]] == 1 and NetF[NB1[i, NBSh[j]]].any() != 0:  # MT
                if NetF[NB1[i, NBSh[j]], 0] < 0.1:
                    ActAlign0, ActAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, NBSh[j]], 1] < 0.1:
                    ActAlign0, ActAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    Select = np.random.uniform(low=0, high=1)
                    if Select <= 0.5:
                        ActAlign0 = NetF[NB1[i, NBSh[j]], 0] * (1 + ra.randrange(50, 76) / 100)
                        ActAlign1 = NetF[NB1[i, NBSh[j]], 1]
                    else:
                        ActAlign0 = NetF[NB1[i, NBSh[j]], 0]
                        ActAlign1 = NetF[NB1[i, NBSh[j]], 1] * (1 + ra.randrange(50, 76) / 100)
                Ali[i, 0], Ali[i, 1] = ActAlign0, ActAlign1
                Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
                Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
                LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
                TrackerAli = 1
            else:
                j += 1
        if TrackerAli == 0:
            Geo = np.random.uniform(low=0, high=2 * ma.pi)
            Ept[i, 0], Ept[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(Geo), Loc[i, 2] - 0.5 * FiLen * ma.sin(Geo)
            Ept[i, 2], Ept[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(Geo), Loc[i, 2] + 0.5 * FiLen * ma.sin(Geo)
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]

# Save post-simulation conditions
np.save(CaseNum + '_' + CaseIter + '_' + 'State_Fin', State)
np.save(CaseNum + '_' + CaseIter + '_' + 'LineSeg_Fin', LineSeg)
np.save(CaseNum + '_' + CaseIter + '_' + 'Sphere_Tau', SphTauList)
np.save(CaseNum + '_' + CaseIter + '_' + 'Sphere_X', SphXDisp)
np.save(CaseNum + '_' + CaseIter + '_' + 'Sphere_Y', SphYDisp)

End = time.time()
print(round((End - Start) / 60), "minutes")
