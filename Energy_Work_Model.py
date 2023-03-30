# Import required packages
import numpy as np
import math as ma
import random as ra
import time

Start = time.time()

# Constants
FPerKinesin, FPerMyosin, kBT, FiLen, MWorkDist = 6, 3, 4, 5, 5  # pN/fil / pN/fil / pN*nm / um / nm
BaseGridDist, ResFac = 5, 2  # um / -

# Inputs
CaseNum, CaseIter = 'Num_', 'Iter_'
KinesinPerMT, MyosinPerActin = 15, 0  # - / -
EcMT, EcActin, EcEmpty = 0 * kBT, 0 * kBT, 0  # pN*nm / pN*nm / pN*nm
EaMT, EaActin, EaEmpty = 1.0 * kBT, 0.5 * kBT, 0  # pN*nm / pN*nm / pN*nm
RatioMT, RatioActin, BaseColumns, BaseRows = 0.15, 0.40, 29, 34  # - / - / - / -
NumIterations, TimeStep = 100000, 0.25  # - / s

# Save input parameters
InputList = [RatioMT, RatioActin, BaseColumns, BaseRows, BaseGridDist, ResFac]
np.save(CaseNum + CaseIter + 'Input_Param', InputList)

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


# Define second degree neighbor array -- indexing starts with right and moves counterclockwise
NumNB2 = NumNB1 * 2
NB2, NB2Partner = np.zeros((TotPts, NumNB2), dtype=int), [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
for i in range(TotPts):
    for j in range(NumNB2):
        if j % 2 == 0:
            NB2[i, j] = nbt(i, int(j / 2), 1)
        else:
            NB2[i, j] = nbt(nbt(i, int((j - 1) / 2), 1), int(((j - 1) / 2 + 2) % NumNB1), 0)

# Define state array and load randomized initial conditions
RatioEmpty, PossibleStates = 1 - (RatioMT + RatioActin), [1, 2, 3]  # 1=microtubule (MT), 2=actin, 3=empty space
StateList = np.random.choice(PossibleStates, TotPts, p=[RatioMT, RatioActin, RatioEmpty])
State = np.asarray(StateList, dtype=int)
NumMT, NumActin = np.count_nonzero(State == 1), np.count_nonzero(State == 2)
NumKinesin, NumMyosin = KinesinPerMT * NumMT, MyosinPerActin * NumActin

# Define physical geometry of filaments and load randomized initial conditions -- [x1, y1, x2, y2]
Ept, LineSeg = np.zeros((TotPts, 4)), []
for i in range(TotPts):
    for j in range(4):
        Ept[i, j] = 0
        LineSeg.append([(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])])
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:
        GeoIni = np.random.uniform(low=0, high=2 * ma.pi)
        Ept[i, 0], Ept[i, 1] = Loc[i, 1] - 0.5 * FiLen * ma.cos(GeoIni), Loc[i, 2] - 0.5 * FiLen * ma.sin(GeoIni)
        Ept[i, 2], Ept[i, 3] = Loc[i, 1] + 0.5 * FiLen * ma.cos(GeoIni), Loc[i, 2] + 0.5 * FiLen * ma.sin(GeoIni)
        LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]

# Save initial conditions
np.save(CaseNum + CaseIter + 'State_Ini', State), np.save(CaseNum + CaseIter + 'LineSeg_Ini', LineSeg)

# Define force angle matrices
FAng1, TrackerFA1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
FAng2, TrackerFA2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
for i in range(TotPts):
    for j in range(NumNB1):
        if TrackerFA1[NB1[i, j], NB1Partner[j]] == 0:
            FAng1[i, NB1[i, j]], FAng1[NB1[i, j], i] = j * ma.pi / 3, ((j + 3) % 6) * ma.pi / 3
            TrackerFA1[i, j], TrackerFA1[NB1[i, j], NB1Partner[j]] = 1, 1
    for j in range(NumNB2):
        if TrackerFA2[NB2[i, j], NB2Partner[j]] == 0:
            FAng2[i, NB2[i, j]], FAng2[NB2[i, j], i] = j * ma.pi / 6, ((j + 6) % 12) * ma.pi / 6
            TrackerFA2[i, j], TrackerFA2[NB2[i, j], NB2Partner[j]] = 1, 1

# Define force magnitude and direction matrices and load initial conditions
FMT = (NumKinesin / NumMT * FPerKinesin) if RatioMT != 0 else 0
FAct = (NumMyosin / NumActin * FPerMyosin) if RatioActin != 0 else 0
FDir1, FDir2 = np.ones((TotPts, TotPts)), np.ones((TotPts, TotPts))
FMag1, TrackerFM1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
FMag2, TrackerFM2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
for i in range(TotPts):
    if State[i] == 1 and FMT != 0:  # MT
        for j in range(NumNB1):
            if State[NB1[i, j]] == 1 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # MT
                if ra.random() < 0.5:  # recast force direction as negative
                    FDir1[i, NB1[i, j]], FDir1[NB1[i, j], i] = -1, -1
                FMag1[i, NB1[i, j]], FMag1[NB1[i, j], i] = FDir1[i, NB1[i, j]] * FMT, FDir1[NB1[i, j], i] * FMT
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 1 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # MT
                FMag2[i, NB2[i, j]], FMag2[NB2[i, j], i] = FDir2[i, NB2[i, j]] * FMT, FDir2[NB2[i, j], i] * FMT
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1
    elif State[i] == 2 and FAct != 0:  # actin
        for j in range(NumNB1):
            if State[NB1[i, j]] == 2 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # actin
                if ra.random() < 0.5:  # recast force direction as negative
                    FDir1[i, NB1[i, j]], FDir1[NB1[i, j], i] = -1, -1
                FMag1[i, NB1[i, j]], FMag1[NB1[i, j], i] = FDir1[i, NB1[i, j]] * FAct, FDir1[NB1[i, j], i] * FAct
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 2 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # actin
                FMag2[i, NB2[i, j]], FMag2[NB2[i, j], i] = FDir2[i, NB2[i, j]] * FAct, FDir2[NB2[i, j], i] * FAct
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1

# Define net force and delta arrays
NetF, Del, NBF, FDtDel = np.zeros((TotPts, 2)), np.zeros((NumNB1, 2)), np.zeros((NumNB1, 2)), np.zeros((TotPts, NumNB1))
for i in range(NumNB1):  # define delta matrix for neighbor slots
    Del[i, 0], Del[i, 1] = ma.cos(i * ma.pi / 3), ma.sin(i * ma.pi / 3)
for i in range(TotPts):
    if (State[i] == 1 and FMT != 0) or (State[i] == 2 and FAct != 0):  # MT or actin
        for j in range(NumNB1):
            NetF[i, 0] += FMag1[i, NB1[i, j]] * ma.cos(FAng1[i, NB1[i, j]])
            NetF[i, 1] += FMag1[i, NB1[i, j]] * ma.sin(FAng1[i, NB1[i, j]])
        for j in range(NumNB2):
            NetF[i, 0] += FMag2[i, NB2[i, j]] * ma.cos(FAng2[i, NB2[i, j]])
            NetF[i, 1] += FMag2[i, NB2[i, j]] * ma.sin(FAng2[i, NB2[i, j]])
        NBF[:, 0], NBF[:, 1] = NetF[i, 0], NetF[i, 1]  # define force matrix for neighbor slots
        for j in range(NumNB1):
            FDtDel[i, j] = np.dot(NBF[j], Del[j])  # calculate dot product

# Apply initial force conditions to physical geometry of filaments with applied forces
for i in range(TotPts):
    if (State[i] == 1 and FMT != 0) or (State[i] == 2 and FAct != 0):
        if NetF[i, 0] != 0 or NetF[i, 1] != 0:
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]

# Apply initial force conditions to physical geometry of filaments to incorporate alignment of unforced filaments with
# neighboring filaments of different type (or random alignment if no neighbors)
Ali = np.zeros((TotPts, 2))
for i in range(TotPts):
    if State[i] == 1 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # MT
        j, TrackerAli = 0, 0
        while j in range(NumNB1) and TrackerAli == 0:
            if State[NB1[i, j]] == 2 and NetF[NB1[i, j]].any() != 0:  # actin
                if NetF[NB1[i, j], 0] < 0.1:
                    MTAlign0, MTAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, j], 1] < 0.1:
                    MTAlign0, MTAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    MTAlign0, MTAlign1 = NetF[NB1[i, j], 0], NetF[NB1[i, j], 1] * (1 + ra.randrange(50, 76) / 100)
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
            Ali[i, 0], Ali[i, 1] = ra.randrange(-9, 9, 2), ra.randrange(-9, 9, 2)
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
    elif State[i] == 2 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # actin
        j, TrackerAli = 0, 0
        while j in range(NumNB1) and TrackerAli == 0:
            if State[NB1[i, j]] == 1 and NetF[NB1[i, j]].any() != 0:  # MT
                if NetF[NB1[i, j], 0] < 0.1:
                    ActAlign0, ActAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, j], 1] < 0.1:
                    ActAlign0, ActAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    ActAlign0, ActAlign1 = NetF[NB1[i, j], 0], NetF[NB1[i, j], 1] * (1 + ra.randrange(50, 76) / 100)
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
            Ali[i, 0], Ali[i, 1] = ra.randrange(-9, 9, 2), ra.randrange(-9, 9, 2)
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]

# Define probability matrices and load initial conditions
SameNB, Delta, NBForce, KConst = np.zeros(TotPts), np.zeros((NumNB1, 2)), np.zeros((NumNB1, 2)), np.zeros(NumNB1)
IndProb, NumPossibleMoves, CombProb = np.zeros((TotPts, TotPts)), TotPts * 3, np.zeros((TotPts, TotPts))
AllMoveIDs = np.arange(NumPossibleMoves).reshape((TotPts, 3))  # array of all MoveIDs
AllMoveIDsList = [i for i in range(NumPossibleMoves)]  # list of all MoveIDs
ProbAllMoveIDs, NormProbAllMoveIDs = np.zeros((TotPts, 3)), np.zeros((TotPts, 3))
for i in range(NumNB1):  # define delta matrix for neighbor slots
    Delta[i, 0], Delta[i, 1] = MWorkDist * ma.cos(i * ma.pi / 3), MWorkDist * ma.sin(i * ma.pi / 3)
for i in range(TotPts):
    # Count number of same neighbors
    SameNB[i] = 0
    if State[i] == 1 or State[i] == 2:  # MT or actin
        for j in range(NumNB1):
            if State[i] == State[NB1[i, j]]:
                SameNB[i] += 1
        for j in range(NumNB2):
            if State[i] == State[NB2[i, j]]:
                SameNB[i] += 1
    # Define force matrix for neighbor slots
    NBForce[:, 0] = NetF[i, 0]
    NBForce[:, 1] = NetF[i, 1]
    # Calculate rate constants -- k = 10^10*e^(-(Ea+N*Ec-F.delta)/(kB*T))
    if State[i] == 1:  # MT
        for j in range(NumNB1):
            KConst[j] = 10e10 * ma.exp(-1 * (EaMT + SameNB[i] * EcMT - np.dot(NBForce[j], Delta[j])) / kBT)
    elif State[i] == 2:  # actin
        for j in range(NumNB1):
            KConst[j] = 10e10 * ma.exp(-1 * (EaActin + SameNB[i] * EcActin - np.dot(NBForce[j], Delta[j])) / kBT)
    else:  # empty space
        for j in range(NumNB1):
            KConst[j] = 10e10 * ma.exp(-1 * (EaEmpty + SameNB[i] * EcEmpty - np.dot(NBForce[j], Delta[j])) / kBT)
    # Calculate individual probabilities -- P = 1-e^(-k*dt)
    for j in range(NumNB1):
        if State[i] == 3 and State[NB1[i, j]] == 3:
            IndProb[i, NB1[i, j]] = 0
        else:
            IndProb[i, NB1[i, j]] = 1 - ma.exp(-1 * KConst[j] * TimeStep)
# Combined probability matrix
for i in range(TotPts):
    for j in range(NumNB1):
        CombProb[i, NB1[i, j]] = IndProb[i, NB1[i, j]] * IndProb[NB1[i, j], i]
# Raw probabilities of each MoveID
for i in range(TotPts):
    ProbAllMoveIDs[i, 0] = CombProb[i, NB1[i, 1]]  # move between 1-4 neighbor relationship
    ProbAllMoveIDs[i, 1] = CombProb[i, NB1[i, 0]]  # move between 0-3 neighbor relationship
    ProbAllMoveIDs[i, 2] = CombProb[i, NB1[i, 5]]  # move between 5-2 neighbor relationship
# Normalized probabilities of each MoveID
if np.sum(ProbAllMoveIDs) < 1:
    print("Warning: Sum of all probabilities < 1: ", np.sum(ProbAllMoveIDs))
NormProbAllMoveIDs = ProbAllMoveIDs / np.sum(ProbAllMoveIDs)

# Run simulation
NormProbNestList, NormProbList, PtMv, PtAf = [], [], [], []
for t in range(NumIterations):
    # Convert probability array to list
    NormProbNestList.clear(), NormProbList.clear()
    NormProbNestList = NormProbAllMoveIDs.tolist()
    for i in NormProbNestList:
        NormProbList.extend(i)
    # Select MoveID to occur and lookup corresponding PointIDs
    MoveID = np.random.choice(AllMoveIDsList, p=NormProbList)
    LookUp = np.asarray(AllMoveIDs == MoveID).nonzero()
    PtMoveA = int(LookUp[0])
    if int(LookUp[1]) == 0:
        PtMoveB = NB1[PtMoveA, 1]
    elif int(LookUp[1]) == 1:
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
    # Update force magnitude and direction matrices
    for i in range(len(PtMv)):  # reset entries
        FMag1[PtMv[i], NB1[PtMv[i]]], FMag1[NB1[PtMv[i]], PtMv[i]] = 0, 0
        FDir1[PtMv[i], NB1[PtMv[i]]], FDir1[NB1[PtMv[i]], PtMv[i]] = 1, 1
        FMag2[PtMv[i], NB2[PtMv[i]]], FMag2[NB2[PtMv[i]], PtMv[i]] = 0, 0
        FDir2[PtMv[i], NB2[PtMv[i]]], FDir2[NB2[PtMv[i]], PtMv[i]] = 1, 1
    for i in range(len(PtMv)):
        if State[PtMv[i]] == 1 and FMT != 0:  # MT
            for j in range(NumNB1):
                if State[NB1[PtMv[i], j]] == 1 and TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] == 0:  # MT
                    if ra.random() < 0.5:  # recast force direction as negative
                        FDir1[PtMv[i], NB1[PtMv[i], j]], FDir1[NB1[PtMv[i], j], PtMv[i]] = -1, -1
                    FMag1[PtMv[i], NB1[PtMv[i], j]] = FDir1[PtMv[i], NB1[PtMv[i], j]] * FMT
                    FMag1[NB1[PtMv[i], j], PtMv[i]] = FDir1[NB1[PtMv[i], j], PtMv[i]] * FMT
                    TrackerFM1[PtMv[i], j], TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] = 1, 1
            for j in range(NumNB2):
                if State[NB2[PtMv[i], j]] == 1 and TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] == 0:  # MT
                    FMag2[PtMv[i], NB2[PtMv[i], j]] = FDir2[PtMv[i], NB2[PtMv[i], j]] * FMT
                    FMag2[NB2[PtMv[i], j], PtMv[i]] = FDir2[NB2[PtMv[i], j], PtMv[i]] * FMT
                    TrackerFM2[PtMv[i], j], TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] = 1, 1
        elif State[PtMv[i]] == 2 and FAct != 0:  # actin
            for j in range(NumNB1):
                if State[NB1[PtMv[i], j]] == 2 and TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] == 0:  # actin
                    if ra.random() < 0.5:  # recast force direction as negative
                        FDir1[PtMv[i], NB1[PtMv[i], j]], FDir1[NB1[PtMv[i], j], PtMv[i]] = -1, -1
                    FMag1[PtMv[i], NB1[PtMv[i], j]] = FDir1[PtMv[i], NB1[PtMv[i], j]] * FAct
                    FMag1[NB1[PtMv[i], j], PtMv[i]] = FDir1[NB1[PtMv[i], j], PtMv[i]] * FAct
                    TrackerFM1[PtMv[i], j], TrackerFM1[NB1[PtMv[i], j], NB1Partner[j]] = 1, 1
            for j in range(NumNB2):
                if State[NB2[PtMv[i], j]] == 2 and TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] == 0:  # actin
                    FMag2[PtMv[i], NB2[PtMv[i], j]] = FDir2[PtMv[i], NB2[PtMv[i], j]] * FAct
                    FMag2[NB2[PtMv[i], j], PtMv[i]] = FDir2[NB2[PtMv[i], j], PtMv[i]] * FAct
                    TrackerFM2[PtMv[i], j], TrackerFM2[NB2[PtMv[i], j], NB2Partner[j]] = 1, 1
    # Update net force arrays for points affected
    for i in range(len(PtAf)):
        NetF[PtAf[i]] = 0  # reset entries
        if (State[PtAf[i]] == 1 and FMT != 0) or (State[PtAf[i]] == 2 and FAct != 0):  # MT or actin
            for j in range(NumNB1):
                NetF[PtAf[i], 0] += FMag1[PtAf[i], NB1[PtAf[i], j]] * ma.cos(FAng1[PtAf[i], NB1[PtAf[i], j]])
                NetF[PtAf[i], 1] += FMag1[PtAf[i], NB1[PtAf[i], j]] * ma.sin(FAng1[PtAf[i], NB1[PtAf[i], j]])
            for j in range(NumNB2):
                NetF[PtAf[i], 0] += FMag2[PtAf[i], NB2[PtAf[i], j]] * ma.cos(FAng2[PtAf[i], NB2[PtAf[i], j]])
                NetF[PtAf[i], 1] += FMag2[PtAf[i], NB2[PtAf[i], j]] * ma.sin(FAng2[PtAf[i], NB2[PtAf[i], j]])
            NBF[:, 0], NBF[:, 1] = NetF[PtAf[i], 0], NetF[PtAf[i], 1]  # define force matrix for neighbor slots
            for j in range(NumNB1):
                FDtDel[PtAf[i], j] = np.dot(NBF[j], Del[j])  # calculate dot product
    # Update individual probability matrix
    for i in range(len(PtAf)):
        SameNB[PtAf[i]] = 0  # reset entries
        if State[PtAf[i]] == 1 or State[PtAf[i]] == 2:  # MT or actin
            # Count number of same neighbors
            for j in range(NumNB1):
                if State[PtAf[i]] == State[NB1[PtAf[i], j]]:
                    SameNB[PtAf[i]] += 1
            for j in range(NumNB2):
                if State[PtAf[i]] == State[NB2[PtAf[i], j]]:
                    SameNB[PtAf[i]] += 1
        # Define force matrix for neighbor slots
        NBForce[:, 0] = NetF[PtAf[i], 0]
        NBForce[:, 1] = NetF[PtAf[i], 1]
        # Calculate rate constants -- k = 10^10*e^(-(Ea+N*Ec-F.delta)/(kB*T))
        if State[PtAf[i]] == 1:  # MT
            for j in range(NumNB1):
                KConst[j] = 10e10 * ma.exp(-1 * (EaMT + SameNB[PtAf[i]] * EcMT - np.dot(NBForce[j], Delta[j])) / kBT)
        elif State[PtAf[i]] == 2:  # actin
            for j in range(NumNB1):
                KConst[j] = 10e10 * ma.exp(
                    -1 * (EaActin + SameNB[PtAf[i]] * EcActin - np.dot(NBForce[j], Delta[j])) / kBT)
        else:  # empty space
            for j in range(NumNB1):
                KConst[j] = 10e10 * ma.exp(
                    -1 * (EaEmpty + SameNB[PtAf[i]] * EcEmpty - np.dot(NBForce[j], Delta[j])) / kBT)
        # Individual probability matrix -- P = 1-e^(-k*dt)
        for j in range(NumNB1):
            if State[PtAf[i]] == 3 and State[NB1[PtAf[i], j]] == 3:
                IndProb[PtAf[i], NB1[PtAf[i], j]] = 0
            else:
                IndProb[PtAf[i], NB1[PtAf[i], j]] = 1 - ma.exp(-1 * KConst[j] * TimeStep)
    # Update combined probability matrices
    for i in range(len(PtAf)):
        for j in range(NumNB1):
            CombProb[PtAf[i], NB1[PtAf[i], j]] = IndProb[PtAf[i], NB1[PtAf[i], j]] * IndProb[NB1[PtAf[i], j], PtAf[i]]
            CombProb[NB1[PtAf[i], j], PtAf[i]] = IndProb[PtAf[i], NB1[PtAf[i], j]] * IndProb[NB1[PtAf[i], j], PtAf[i]]
    # Update raw probability array
    for i in range(len(PtAf)):
        ProbAllMoveIDs[PtAf[i], 0] = CombProb[PtAf[i], NB1[PtAf[i], 1]]
        ProbAllMoveIDs[PtAf[i], 1] = CombProb[PtAf[i], NB1[PtAf[i], 0]]
        ProbAllMoveIDs[PtAf[i], 2] = CombProb[PtAf[i], NB1[PtAf[i], 5]]
    # Update normalized probability array
    if np.sum(ProbAllMoveIDs) < 1:
        print("Warning: Sum of all probabilities < 1: ", np.sum(ProbAllMoveIDs))
    NormProbAllMoveIDs = ProbAllMoveIDs / np.sum(ProbAllMoveIDs)

# Update physical geometry of filaments to incorporate force conditions for filaments with applied forces
for i in range(TotPts):
    Ept[i] = 0  # reset entries
    LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
    if (State[i] == 1 and FMT != 0) or (State[i] == 2 and FAct != 0):
        if NetF[i, 0] != 0 or NetF[i, 1] != 0:
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
# Update physical geometry of filaments to incorporate alignment of unforced filaments with neighboring filaments
# of different type (or random alignment if no neighbors)
Ali.fill(0)  # reset entries
for i in range(TotPts):
    if State[i] == 1 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # MT
        j, TrackerAli = 0, 0
        while j in range(NumNB1) and TrackerAli == 0:
            if State[NB1[i, j]] == 2 and NetF[NB1[i, j]].any() != 0:  # actin
                if NetF[NB1[i, j], 0] < 0.1:
                    MTAlign0, MTAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, j], 1] < 0.1:
                    MTAlign0, MTAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    MTAlign0, MTAlign1 = NetF[NB1[i, j], 0], NetF[NB1[i, j], 1] * (1 + ra.randrange(50, 76) / 100)
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
            Ali[i, 0], Ali[i, 1] = ra.randrange(-9, 9, 2), ra.randrange(-9, 9, 2)
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]
    elif State[i] == 2 and NetF[i, 0] == 0 and NetF[i, 1] == 0:  # actin
        j, TrackerAli = 0, 0
        while j in range(NumNB1) and TrackerAli == 0:
            if State[NB1[i, j]] == 1 and NetF[NB1[i, j]].any() != 0:  # MT
                if NetF[NB1[i, j], 0] < 0.1:
                    ActAlign0, ActAlign1 = ra.randrange(-25, 27, 2) / 100, 1
                elif NetF[NB1[i, j], 1] < 0.1:
                    ActAlign0, ActAlign1 = 1, ra.randrange(-25, 27, 2) / 100
                else:
                    ActAlign0, ActAlign1 = NetF[NB1[i, j], 0], NetF[NB1[i, j], 1] * (1 + ra.randrange(50, 76) / 100)
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
            Ali[i, 0], Ali[i, 1] = ra.randrange(-9, 9, 2), ra.randrange(-9, 9, 2)
            Ept[i, 0] = Loc[i, 1] - 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 1] = Loc[i, 2] - 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 2] = Loc[i, 1] + 0.5 * FiLen * ma.cos(ma.atan2(Ali[i, 1], Ali[i, 0]))
            Ept[i, 3] = Loc[i, 2] + 0.5 * FiLen * ma.sin(ma.atan2(Ali[i, 1], Ali[i, 0]))
            LineSeg[i] = [(Ept[i, 0], Ept[i, 1]), (Ept[i, 2], Ept[i, 3])]

# Save post-simulation conditions
np.save(CaseNum + CaseIter + 'State_Fin', State), np.save(CaseNum + CaseIter + 'LineSeg_Fin', LineSeg)

End = time.time()
print(round((End - Start) / 60), "minutes")
