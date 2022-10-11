# Import required packages
import numpy as np
import math as ma
import random
import time

Start = time.time()

# Constants
ForcePerKinesin, ForcePerMyosin, kBT, FilamentLen = 6, 3, 4, 5  # pN/fil / pN/fil / pN*nm / um
GammaFilMT, GammaFilAct = 0.01, 0.005  # pN*ms/nm / pN*ms/nm
GammaMotKin, GammaMotMyo = 6, 0.3  # (pN*ms/nm)/fil / (pN*ms/nm)/fil
BaseGridDist, ResFac = 5, 2  # um / -

# Inputs
CaseNum, CaseIteration = 'x_', 'x_'
KinesinPerMT, MyosinPerActin = 0, 0  # - / -
GammaCrosslinkMT, GammaCrosslinkActin = 0, 0  # (pN*ms/nm)/fil / (pN*ms/nm)/fil
RatioMT, RatioActin, NumIterations, TimeStep = 0.15, 0.40, 1000000, 100  # - / - / - / ms
BaseColumns, BaseRows, NumAnn = 29, 34, 6  # - / - / -

# Define hexagonal coordinate system -- NumRows must be even to satisfy boundary conditions
NumColumns, NumRows, GridDist = BaseColumns * ResFac, BaseRows * ResFac, BaseGridDist / ResFac
TotPts = NumColumns * NumRows
Loc = np.zeros((TotPts, 3))
for i in range(TotPts):
    Loc[i, 0] = i
for i in range(NumRows):
    if i % 2 == 0:  # even rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            Loc[j, 1] = (j - i * NumColumns) * GridDist
            Loc[j, 2] = i / (2 * ma.tan(ma.pi / 6)) * GridDist
    else:  # odd rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            Loc[j, 1] = (j + 0.5 - i * NumColumns) * GridDist
            Loc[j, 2] = i / (2 * ma.tan(ma.pi / 6)) * GridDist

# Define first degree neighbor array using periodic boundary conditions
# Indexing starts with right and moves counterclockwise
NumNB1, NB1Partner = 6, [3, 4, 5, 0, 1, 2]
NB1 = np.zeros((TotPts, NumNB1), dtype=int)
for i in range(NumRows):
    if i == 0:  # boundary condition (bottom)
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB1[j, 1] = Loc[j, 0] + NumColumns
            NB1[j, 5] = Loc[j, 0] + (NumRows - 1) * NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 2] = Loc[j, 0] + 2 * NumColumns - 1
                NB1[j, 3] = Loc[j, 0] + NumColumns - 1
                NB1[j, 4] = Loc[j, 0] + NumRows * NumColumns - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB1[j, 0] = Loc[j, 0] - NumColumns + 1
                NB1[j, 2] = Loc[j, 0] + NumColumns - 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 4] = Loc[j, 0] + (NumRows - 1) * NumColumns - 1
            else:
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 2] = Loc[j, 0] + NumColumns - 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 4] = Loc[j, 0] + (NumRows - 1) * NumColumns - 1
    elif i == (NumRows - 1):  # boundary condition (top)
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB1[j, 2] = Loc[j, 0] - (NumRows - 1) * NumColumns
            NB1[j, 4] = Loc[j, 0] - NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 1] = Loc[j, 0] - (NumRows - 1) * NumColumns + 1
                NB1[j, 3] = Loc[j, 0] + NumColumns - 1
                NB1[j, 5] = Loc[j, 0] - NumColumns + 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB1[j, 0] = Loc[j, 0] - NumColumns + 1
                NB1[j, 1] = Loc[j, 0] - NumRows * NumColumns + 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 5] = Loc[j, 0] - 2 * NumColumns + 1
            else:
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 1] = Loc[j, 0] - (NumRows - 1) * NumColumns + 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 5] = Loc[j, 0] - NumColumns + 1
    elif i % 2 == 0:  # even rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB1[j, 1] = Loc[j, 0] + NumColumns
            NB1[j, 5] = Loc[j, 0] - NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 2] = Loc[j, 0] + 2 * NumColumns - 1
                NB1[j, 3] = Loc[j, 0] + NumColumns - 1
                NB1[j, 4] = Loc[j, 0] - 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB1[j, 0] = Loc[j, 0] - NumColumns + 1
                NB1[j, 2] = Loc[j, 0] + NumColumns - 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 4] = Loc[j, 0] - NumColumns - 1
            else:
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 2] = Loc[j, 0] + NumColumns - 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 4] = Loc[j, 0] - NumColumns - 1
    else:  # odd rows
        for j in range(i * NumColumns, i * NumColumns + NumColumns):
            NB1[j, 2] = Loc[j, 0] + NumColumns
            NB1[j, 4] = Loc[j, 0] - NumColumns
            if j == i * NumColumns:  # boundary condition (left)
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 1] = Loc[j, 0] + NumColumns + 1
                NB1[j, 3] = Loc[j, 0] + NumColumns - 1
                NB1[j, 5] = Loc[j, 0] - NumColumns + 1
            elif j == (i * NumColumns + NumColumns - 1):  # boundary condition (right)
                NB1[j, 0] = Loc[j, 0] - NumColumns + 1
                NB1[j, 1] = Loc[j, 0] + 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 5] = Loc[j, 0] - 2 * NumColumns + 1
            else:
                NB1[j, 0] = Loc[j, 0] + 1
                NB1[j, 1] = Loc[j, 0] + NumColumns + 1
                NB1[j, 3] = Loc[j, 0] - 1
                NB1[j, 5] = Loc[j, 0] - NumColumns + 1


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
RatioEmpty = 1 - (RatioMT + RatioActin)
PossibleStates = [1, 2, 3]  # 1=microtubule (MT), 2=actin, 3=empty space
StateList = np.random.choice(PossibleStates, TotPts, p=[RatioMT, RatioActin, RatioEmpty])
State = np.asarray(StateList, dtype=int)
NumMT, NumActin = np.count_nonzero(State == 1), np.count_nonzero(State == 2)
NumKinesin, NumMyosin = KinesinPerMT * NumMT, MyosinPerActin * NumActin

# Define physical geometry of filaments and load randomized initial conditions -- [x1, y1, x2, y2]
Endpts, LineSeg = np.zeros((TotPts, 4)), []
for i in range(TotPts):
    for j in range(4):
        Endpts[i, j] = 0
        LineSeg.append([(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])])
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:
        GeoIni = np.random.uniform(low=0, high=2 * ma.pi)
        Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(GeoIni)
        Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(GeoIni)
        Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(GeoIni)
        Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(GeoIni)
        LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]

# Define force angle matrices
ForceAng1, TrackerFA1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
ForceAng2, TrackerFA2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
for i in range(TotPts):
    for j in range(NumNB1):
        if TrackerFA1[NB1[i, j], NB1Partner[j]] == 0:
            ForceAng1[i, NB1[i, j]], ForceAng1[NB1[i, j], i] = j * ma.pi / 3, ((j + 3) % 6) * ma.pi / 3
            TrackerFA1[i, j], TrackerFA1[NB1[i, j], NB1Partner[j]] = 1, 1
    for j in range(NumNB2):
        if TrackerFA2[NB2[i, j], NB2Partner[j]] == 0:
            ForceAng2[i, NB2[i, j]], ForceAng2[NB2[i, j], i] = j * ma.pi / 6, ((j + 6) % 12) * ma.pi / 6
            TrackerFA2[i, j], TrackerFA2[NB2[i, j], NB2Partner[j]] = 1, 1

# Define force direction matrix and load initial conditions
ForceDir, TrackerFDIni = np.ones((TotPts, TotPts)), np.zeros((TotPts, TotPts), dtype=int)
for i in range(TotPts):
    for j in range(TotPts):
        if TrackerFDIni[i, j] == 0:
            if random.random() <= 0.5:  # recast force direction as negative
                ForceDir[i, j], ForceDir[j, i] = -1, -1
            TrackerFDIni[i, j], TrackerFDIni[j, i] = 1, 1

# Define force magnitude matrices and load initial conditions
ForceMT = (NumKinesin / NumMT * ForcePerKinesin) if RatioMT != 0 else 0
ForceActin = (NumMyosin / NumActin * ForcePerMyosin) if RatioActin != 0 else 0
ForceMag1, TrackerFM1 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
ForceMag2, TrackerFM2 = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB2), dtype=int)
for i in range(TotPts):
    if State[i] == 1:  # MT
        for j in range(NumNB1):
            if State[NB1[i, j]] == 1 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # MT
                ForceMag1[i, NB1[i, j]] = ForceDir[i, NB1[i, j]] * ForceMT
                ForceMag1[NB1[i, j], i] = ForceDir[NB1[i, j], i] * ForceMT
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 1 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # MT
                ForceMag2[i, NB2[i, j]] = ForceDir[i, NB2[i, j]] * ForceMT
                ForceMag2[NB2[i, j], i] = ForceDir[NB2[i, j], i] * ForceMT
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1
    elif State[i] == 2:  # actin
        for j in range(NumNB1):
            if State[NB1[i, j]] == 2 and TrackerFM1[NB1[i, j], NB1Partner[j]] == 0:  # actin
                ForceMag1[i, NB1[i, j]] = ForceDir[i, NB1[i, j]] * ForceActin
                ForceMag1[NB1[i, j], i] = ForceDir[NB1[i, j], i] * ForceActin
                TrackerFM1[i, j], TrackerFM1[NB1[i, j], NB1Partner[j]] = 1, 1
        for j in range(NumNB2):
            if State[NB2[i, j]] == 2 and TrackerFM2[NB2[i, j], NB2Partner[j]] == 0:  # actin
                ForceMag2[i, NB2[i, j]] = ForceDir[i, NB2[i, j]] * ForceActin
                ForceMag2[NB2[i, j], i] = ForceDir[NB2[i, j], i] * ForceActin
                TrackerFM2[i, j], TrackerFM2[NB2[i, j], NB2Partner[j]] = 1, 1

# Define net force array
NetF = np.zeros((TotPts, 2))
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:  # MT or actin
        for j in range(NumNB1):
            NetF[i, 0] += ForceMag1[i, NB1[i, j]] * ma.cos(ForceAng1[i, NB1[i, j]])
            NetF[i, 1] += ForceMag1[i, NB1[i, j]] * ma.sin(ForceAng1[i, NB1[i, j]])
        for j in range(NumNB2):
            NetF[i, 0] += ForceMag2[i, NB2[i, j]] * ma.cos(ForceAng2[i, NB2[i, j]])
            NetF[i, 1] += ForceMag2[i, NB2[i, j]] * ma.sin(ForceAng2[i, NB2[i, j]])

# Apply initial force conditions to physical geometry of filaments with applied forces
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:
        if NetF[i, 0] != 0 or NetF[i, 1] != 0:
            Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(ma.atan2(NetF[i, 1], NetF[i, 0]))
            Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(ma.atan2(NetF[i, 1], NetF[i, 0]))
            LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]

# Apply initial force conditions to physical geometry of filaments to incorporate alignment of unforced filaments with
# neighboring filaments of different type (or random alignment if no neighbors)
PlNF = np.zeros((TotPts, 2))
for i in range(TotPts):
    if State[i] == 1:  # MT
        if NetF[i, 0] == 0 and NetF[i, 1] == 0:
            j, TrackerPl = 0, 0
            while j in range(NumNB1) and TrackerPl == 0:
                if State[NB1[i, j]] == 2 and NetF[NB1[i, j]].any() != 0:  # actin
                    if NetF[NB1[i, j], 0] < 0.1:
                        MTAlign0 = random.randrange(-25, 27, 2) / 100
                        MTAlign1 = 1
                    elif NetF[NB1[i, j], 1] < 0.1:
                        MTAlign0 = 1
                        MTAlign1 = random.randrange(-25, 27, 2) / 100
                    else:
                        MTAlign0 = NetF[NB1[i, j], 0]
                        MTAlign1 = NetF[NB1[i, j], 1] * (1 + random.randrange(50, 76) / 100)
                    PlNF[i, 0], PlNF[i, 1] = MTAlign0, MTAlign1
                    Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]
                    TrackerPl = 1
                else:
                    j += 1
            if TrackerPl == 0:
                PlNF[i, 0], PlNF[i, 1] = random.randrange(-9, 9, 2), random.randrange(-9, 9, 2)
                Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]
    elif State[i] == 2:  # actin
        if NetF[i, 0] == 0 and NetF[i, 1] == 0:
            j, TrackerPl = 0, 0
            while j in range(NumNB1) and TrackerPl == 0:
                if State[NB1[i, j]] == 1 and NetF[NB1[i, j]].any() != 0:  # MT
                    if NetF[NB1[i, j], 0] < 0.1:
                        ActinAlign0 = random.randrange(-25, 27, 2) / 100
                        ActinAlign1 = 1
                    elif NetF[NB1[i, j], 1] < 0.1:
                        ActinAlign0 = 1
                        ActinAlign1 = random.randrange(-25, 27, 2) / 100
                    else:
                        ActinAlign0 = NetF[NB1[i, j], 0]
                        ActinAlign1 = NetF[NB1[i, j], 1] * (1 + random.randrange(50, 76) / 100)
                    PlNF[i, 0], PlNF[i, 1] = ActinAlign0, ActinAlign1
                    Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                    LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]
                    TrackerPl = 1
                else:
                    j += 1
            if TrackerPl == 0:
                PlNF[i, 0], PlNF[i, 1] = random.randrange(-9, 9, 2), random.randrange(-9, 9, 2)
                Endpts[i, 0] = Loc[i, 1] - 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 1] = Loc[i, 2] - 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 2] = Loc[i, 1] + 0.5 * FilamentLen * ma.cos(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                Endpts[i, 3] = Loc[i, 2] + 0.5 * FilamentLen * ma.sin(ma.atan2(PlNF[i, 1], PlNF[i, 0]))
                LineSeg[i] = [(Endpts[i, 0], Endpts[i, 1]), (Endpts[i, 2], Endpts[i, 3])]

# Define probability matrices and load initial conditions
Delta, NBForce, GConstMot, GConstTot = np.zeros((NumNB1, 2)), np.zeros((NumNB1, 2)), np.zeros(TotPts), np.zeros(TotPts)
DConst, SigmaConst, MuConst = np.zeros(TotPts), np.zeros(TotPts), np.zeros((TotPts, NumNB1))
IndivProb, NumPossibleMoves = np.zeros((TotPts, TotPts)), TotPts * 3
CombProb, TrackerCP = np.zeros((TotPts, TotPts)), np.zeros((TotPts, NumNB1), dtype=int)
AllMoveIDs = np.arange(NumPossibleMoves).reshape((TotPts, 3))  # array of all MoveIDs
AllMoveIDsList = [i for i in range(NumPossibleMoves)]  # list of all MoveIDs
ProbAllMoveIDs, NormProbAllMoveIDs = np.zeros((TotPts, 3)), np.zeros((TotPts, 3))
for i in range(NumNB1):  # define delta matrix for neighbor slots
    Delta[i, 0], Delta[i, 1] = ma.cos(i * ma.pi / 3), ma.sin(i * ma.pi / 3)
for i in range(TotPts):
    if State[i] == 1 or State[i] == 2:  # MT or actin
        # Count number of same neighbors
        SameNB = 0
        for j in range(NumNB1):
            if State[i] == State[NB1[i, j]]:
                SameNB += 1
        for j in range(NumNB2):
            if State[i] == State[NB2[i, j]]:
                SameNB += 1
        # Define force matrix for neighbor slots
        NBForce[:, 0] = NetF[i, 0]
        NBForce[:, 1] = NetF[i, 1]
        # Calculate gamma constant
        if State[i] == 1:  # MT
            if KinesinPerMT != 0:
                for j in range(NumNB1):
                    if State[j] == 1:
                        GConstMot[i] += GammaMotKin * KinesinPerMT  # gamma motor
                for j in range(NumNB2):
                    if State[j] == 1:
                        GConstMot[i] += GammaMotKin * KinesinPerMT  # gamma motor
            GConstTot[i] = GammaFilMT + GConstMot[i] + GammaCrosslinkMT * SameNB  # gamma total
        elif State[i] == 2:  # actin
            if MyosinPerActin != 0:
                for j in range(NumNB1):
                    if State[j] == 2:
                        GConstMot[i] += GammaMotMyo * MyosinPerActin  # gamma motor
                for j in range(NumNB2):
                    if State[j] == 2:
                        GConstMot[i] += GammaMotMyo * MyosinPerActin  # gamma motor
            GConstTot[i] = GammaFilAct + GConstMot[i] + GammaCrosslinkActin * SameNB  # gamma total
        # Calculate D constant, sigma constant, mu constant, and individual probability
        # D -- D = kBT/gamma
        DConst[i] = kBT / GConstTot[i]
        # Sigma -- sigma = sqrt(2*D*timestep)
        SigmaConst[i] = ma.sqrt(2 * DConst[i] * TimeStep)
        for j in range(NumNB1):
            # Mu -- mu = (F/gamma)*timestep
            MuConst[i, j] = (np.dot(NBForce[j], Delta[j]) / GConstTot[i]) * TimeStep
            # Individual probability -- P = 1-0.5*(1+erf((L-mu)/(sigma*sqrt(2))))
            IndivProb[i, NB1[i, j]] = 1 - 0.5 * (1 + ma.erf((GridDist - MuConst[i, j]) / (SigmaConst[i] * ma.sqrt(2))))
    else:
        for j in range(NumNB1):
            if State[i] == 3 and State[NB1[i, j]] != 3:
                IndivProb[i, NB1[i, j]] = 1
            elif State[i] == 3 and State[NB1[i, j]] == 3:
                IndivProb[i, NB1[i, j]] = 0
# Combined probability matrix
for i in range(TotPts):
    for j in range(NumNB1):
        if TrackerCP[NB1[i, j], NB1Partner[j]] == 0:
            CombProb[i, NB1[i, j]] = IndivProb[i, NB1[i, j]] * IndivProb[NB1[i, j], i]
            CombProb[NB1[i, j], i] = IndivProb[i, NB1[i, j]] * IndivProb[NB1[i, j], i]
            TrackerCP[i, j], TrackerCP[NB1[i, j], NB1Partner[j]] = 1, 1
# Raw probabilities of each MoveID
for i in range(TotPts):
    ProbAllMoveIDs[i, 0] = CombProb[i, NB1[i, 1]]  # move between 1-4 neighbor relationship
    ProbAllMoveIDs[i, 1] = CombProb[i, NB1[i, 0]]  # move between 0-3 neighbor relationship
    ProbAllMoveIDs[i, 2] = CombProb[i, NB1[i, 5]]  # move between 5-2 neighbor relationship
# Normalized probabilities of each MoveID
if np.sum(ProbAllMoveIDs) < 1:
    print("Warning: Sum of all probabilities < 1: ", np.sum(ProbAllMoveIDs))
NormProbAllMoveIDs = ProbAllMoveIDs / np.sum(ProbAllMoveIDs)

# Save initial condition
np.save(CaseNum + CaseIteration + 'State_Ini', State)
np.save(CaseNum + CaseIteration + 'LineSeg_Ini', LineSeg)

# Measure initial distribution analytics
MTDist, ActDist = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
MTCoDist, ActCoDist = np.zeros((TotPts, TotPts)), np.zeros((TotPts, TotPts))
MTRadCt, ActRadCt, MTRadSubCt = np.zeros((TotPts, NumAnn)), np.zeros((TotPts, NumAnn)), np.zeros(NumAnn)
ActRadSubCt, MTCoCt, ActCoCt = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
MTRadI, ActRadI, MTCoI, ActCoI = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
MTRadIN, ActRadIN, MTCoIN, ActCoIN = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
NumMTInBounds, NumActinInBounds = 0, 0
for i in range(TotPts):
    if State[i] == 1 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        for j in range(TotPts):
            if State[j] == 1:
                MTDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
            elif State[j] == 2:
                MTCoDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
    elif State[i] == 2 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        for j in range(TotPts):
            if State[j] == 2:
                ActDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
            elif State[j] == 1:
                ActCoDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
for i in range(TotPts):
    if State[i] == 1 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        NumMTInBounds += 1
        for j in range(NumAnn):
            for k in range(TotPts):
                if State[k] == 1 and j * GridDist < round(MTDist[i, k], 1) <= (j + 1) * GridDist:
                    MTRadCt[i, j] += 1
                elif State[k] == 2 and j * GridDist < round(MTCoDist[i, k], 1) <= (j + 1) * GridDist:
                    MTCoCt[j] += 1
    elif State[i] == 2 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        NumActinInBounds += 1
        for j in range(NumAnn):
            for k in range(TotPts):
                if State[k] == 2 and j * GridDist < round(ActDist[i, k], 1) <= (j + 1) * GridDist:
                    ActRadCt[i, j] += 1
                elif State[k] == 1 and j * GridDist < round(ActCoDist[i, k], 1) <= (j + 1) * GridDist:
                    ActCoCt[j] += 1
for i in range(TotPts):
    for j in range(NumAnn):
        MTRadSubCt[j] += MTRadCt[i, j]
        ActRadSubCt[j] += ActRadCt[i, j]
for i in range(NumAnn):
    MTRadI[i] = (MTRadSubCt[i] / NumMTInBounds) / (RatioMT * (i + 1) * NumNB1)
    ActRadI[i] = (ActRadSubCt[i] / NumActinInBounds) / (RatioActin * (i + 1) * NumNB1)
    MTCoI[i] = (MTCoCt[i] / NumMTInBounds) / (RatioActin * (i + 1) * NumNB1)
    ActCoI[i] = (ActCoCt[i] / NumActinInBounds) / (RatioMT * (i + 1) * NumNB1)

# Save initial distribution analytics
np.save(CaseNum + CaseIteration + 'MTRad_Ini', MTRadI)
np.save(CaseNum + CaseIteration + 'ActRad_Ini', ActRadI)
np.save(CaseNum + CaseIteration + 'MTCo_Ini', MTCoI)
np.save(CaseNum + CaseIteration + 'ActCo_Ini', ActCoI)

# Run simulation
TrackerFD = np.zeros((TotPts, NumNB1), dtype=int)
for t in range(NumIterations):
    # Convert probability array to list
    NormProbNestList, NormProbList = NormProbAllMoveIDs.tolist(), []
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
    # Move Points
    State[PtMoveA], State[PtMoveB] = State[PtMoveB], State[PtMoveA]
    PtsMvd = [PtMoveA, PtMoveB]
    # Update force direction and magnitude matrices
    for i in range(len(PtsMvd)):  # reset entries
        for j in range(NumNB1):
            ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]], ForceMag1[NB1[PtsMvd[i], j], PtsMvd[i]] = 0, 0
        for j in range(NumNB2):
            ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]], ForceMag2[NB2[PtsMvd[i], j], PtsMvd[i]] = 0, 0
    TrackerFD.fill(0)  # reset force direction tracker
    if State[PtMoveA] == 1 and State[PtMoveB] == 1:  # both are MT
        for i in range(len(PtsMvd)):
            for j in range(NumNB1):
                if State[NB1[PtsMvd[i], j]] == 1:  # MT
                    if TrackerFD[NB1[PtsMvd[i], j], NB1Partner[j]] == 0:  # reverse direction
                        ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] = -1 * ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]]
                        ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] = -1 * ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]]
                        TrackerFD[PtsMvd[i], j], TrackerFD[NB1[PtsMvd[i], j], NB1Partner[j]] = 1, 1
            for j in range(NumNB1):
                if State[NB1[PtsMvd[i], j]] == 1:  # MT
                    ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] * ForceMT
                    ForceMag1[NB1[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] * ForceMT
            for j in range(NumNB2):
                if State[NB2[PtsMvd[i], j]] == 1:  # MT
                    ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB2[PtsMvd[i], j]] * ForceMT
                    ForceMag2[NB2[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB2[PtsMvd[i], j], PtsMvd[i]] * ForceMT
    elif State[PtMoveA] == 2 and State[PtMoveB] == 2:  # both are actin
        for i in range(len(PtsMvd)):  # calculate new entries
            for j in range(NumNB1):
                if State[NB1[PtsMvd[i], j]] == 2:  # actin
                    if TrackerFD[NB1[PtsMvd[i], j], NB1Partner[j]] == 0:  # reverse direction
                        ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] = -1 * ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]]
                        ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] = -1 * ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]]
                        TrackerFD[PtsMvd[i], j], TrackerFD[NB1[PtsMvd[i], j], NB1Partner[j]] = 1, 1
            for j in range(NumNB1):
                if State[NB1[PtsMvd[i], j]] == 2:  # actin
                    ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] * ForceActin
                    ForceMag1[NB1[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] * ForceActin
            for j in range(NumNB2):
                if State[NB2[PtsMvd[i], j]] == 2:  # actin
                    ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB2[PtsMvd[i], j]] * ForceActin
                    ForceMag2[NB2[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB2[PtsMvd[i], j], PtsMvd[i]] * ForceActin
    else:  # all other cases
        for i in range(len(PtsMvd)):  # calculate new entries
            if State[PtsMvd[i]] == 1:  # MT
                for j in range(NumNB1):
                    if State[NB1[PtsMvd[i], j]] == 1:  # MT
                        ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] * ForceMT
                        ForceMag1[NB1[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] * ForceMT
                for j in range(NumNB2):
                    if State[NB2[PtsMvd[i], j]] == 1:  # MT
                        ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB2[PtsMvd[i], j]] * ForceMT
                        ForceMag2[NB2[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB2[PtsMvd[i], j], PtsMvd[i]] * ForceMT
            elif State[PtsMvd[i]] == 2:  # actin
                for j in range(NumNB1):
                    if State[NB1[PtsMvd[i], j]] == 2:  # actin
                        ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB1[PtsMvd[i], j]] * ForceActin
                        ForceMag1[NB1[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB1[PtsMvd[i], j], PtsMvd[i]] * ForceActin
                for j in range(NumNB2):
                    if State[NB2[PtsMvd[i], j]] == 2:  # actin
                        ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] = ForceDir[PtsMvd[i], NB2[PtsMvd[i], j]] * ForceActin
                        ForceMag2[NB2[PtsMvd[i], j], PtsMvd[i]] = ForceDir[NB2[PtsMvd[i], j], PtsMvd[i]] * ForceActin
    # Update net force arrays
    for i in range(len(PtsMvd)):
        NetF[PtsMvd[i], 0], NetF[PtsMvd[i], 1] = 0, 0  # reset entries
        for j in range(NumNB1):
            NetF[PtsMvd[i], 0] += ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] * ma.cos(
                ForceAng1[PtsMvd[i], NB1[PtsMvd[i], j]])
            NetF[PtsMvd[i], 1] += ForceMag1[PtsMvd[i], NB1[PtsMvd[i], j]] * ma.sin(
                ForceAng1[PtsMvd[i], NB1[PtsMvd[i], j]])
        for j in range(NumNB2):
            NetF[PtsMvd[i], 0] += ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] * ma.cos(
                ForceAng2[PtsMvd[i], NB2[PtsMvd[i], j]])
            NetF[PtsMvd[i], 1] += ForceMag2[PtsMvd[i], NB2[PtsMvd[i], j]] * ma.sin(
                ForceAng2[PtsMvd[i], NB2[PtsMvd[i], j]])
    # Update physical geometry of filaments to incorporate force conditions for filaments with applied forces
    for i in range(len(PtsMvd)):
        for j in range(4):
            Endpts[PtsMvd[i], j] = 0  # reset entries
        LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                              (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
        if State[PtsMvd[i]] == 1 or State[PtsMvd[i]] == 2:
            if NetF[PtsMvd[i], 0] != 0 or NetF[PtsMvd[i], 1] != 0:
                Endpts[PtsMvd[i], 0] = Loc[PtsMvd[i], 1] - 0.5 * FilamentLen * ma.cos(
                    ma.atan2(NetF[PtsMvd[i], 1], NetF[PtsMvd[i], 0]))
                Endpts[PtsMvd[i], 1] = Loc[PtsMvd[i], 2] - 0.5 * FilamentLen * ma.sin(
                    ma.atan2(NetF[PtsMvd[i], 1], NetF[PtsMvd[i], 0]))
                Endpts[PtsMvd[i], 2] = Loc[PtsMvd[i], 1] + 0.5 * FilamentLen * ma.cos(
                    ma.atan2(NetF[PtsMvd[i], 1], NetF[PtsMvd[i], 0]))
                Endpts[PtsMvd[i], 3] = Loc[PtsMvd[i], 2] + 0.5 * FilamentLen * ma.sin(
                    ma.atan2(NetF[PtsMvd[i], 1], NetF[PtsMvd[i], 0]))
                LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                                      (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
    # Update physical geometry of filaments to incorporate alignment of unforced filaments with neighboring filaments
    # of different type (or random alignment if no neighbors)
    for i in range(len(PtsMvd)):
        PlNF[PtsMvd[i], 0], PlNF[PtsMvd[i], 1] = 0, 0  # reset entries
    for i in range(len(PtsMvd)):
        if State[PtsMvd[i]] == 1:  # MT
            if NetF[PtsMvd[i], 0] == 0 and NetF[PtsMvd[i], 1] == 0:
                j, TrackerPl = 0, 0
                while j in range(NumNB1) and TrackerPl == 0:
                    if State[NB1[PtsMvd[i], j]] == 2 and NetF[NB1[PtsMvd[i], j]].any() != 0:  # actin
                        if NetF[NB1[PtsMvd[i], j], 0] < 0.1:
                            MTAlign0 = random.randrange(-25, 27, 2) / 100
                            MTAlign1 = 1
                        elif NetF[NB1[PtsMvd[i], j], 1] < 0.1:
                            MTAlign0 = 1
                            MTAlign1 = random.randrange(-25, 27, 2) / 100
                        else:
                            MTAlign0 = NetF[NB1[PtsMvd[i], j], 0]
                            MTAlign1 = NetF[NB1[PtsMvd[i], j], 1] * (1 + random.randrange(50, 76) / 100)
                        PlNF[PtsMvd[i], 0], PlNF[PtsMvd[i], 1] = MTAlign0, MTAlign1
                        Endpts[PtsMvd[i], 0] = Loc[PtsMvd[i], 1] - 0.5 * FilamentLen * ma.cos(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 1] = Loc[PtsMvd[i], 2] - 0.5 * FilamentLen * ma.sin(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 2] = Loc[PtsMvd[i], 1] + 0.5 * FilamentLen * ma.cos(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 3] = Loc[PtsMvd[i], 2] + 0.5 * FilamentLen * ma.sin(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                                              (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
                        TrackerPl = 1
                    else:
                        j += 1
                if TrackerPl == 0:
                    PlNF[PtsMvd[i], 0], PlNF[PtsMvd[i], 1] = random.randrange(-9, 9, 2), random.randrange(-9, 9, 2)
                    Endpts[PtsMvd[i], 0] = Loc[PtsMvd[i], 1] - 0.5 * FilamentLen * ma.cos(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 1] = Loc[PtsMvd[i], 2] - 0.5 * FilamentLen * ma.sin(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 2] = Loc[PtsMvd[i], 1] + 0.5 * FilamentLen * ma.cos(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 3] = Loc[PtsMvd[i], 2] + 0.5 * FilamentLen * ma.sin(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                                          (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
        elif State[PtsMvd[i]] == 2:  # actin
            if NetF[PtsMvd[i], 0] == 0 and NetF[PtsMvd[i], 1] == 0:
                j, TrackerPl = 0, 0
                while j in range(NumNB1) and TrackerPl == 0:
                    if State[NB1[PtsMvd[i], j]] == 1 and NetF[NB1[PtsMvd[i], j]].any() != 0:  # MT
                        if NetF[NB1[PtsMvd[i], j], 0] < 0.1:
                            ActinAlign0 = random.randrange(-25, 27, 2) / 100
                            ActinAlign1 = 1
                        elif NetF[NB1[PtsMvd[i], j], 1] < 0.1:
                            ActinAlign0 = 1
                            ActinAlign1 = random.randrange(-25, 27, 2) / 100
                        else:
                            ActinAlign0 = NetF[NB1[PtsMvd[i], j], 0]
                            ActinAlign1 = NetF[NB1[PtsMvd[i], j], 1] * (1 + random.randrange(50, 76) / 100)
                        PlNF[PtsMvd[i], 0], PlNF[PtsMvd[i], 1] = ActinAlign0, ActinAlign1
                        Endpts[PtsMvd[i], 0] = Loc[PtsMvd[i], 1] - 0.5 * FilamentLen * ma.cos(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 1] = Loc[PtsMvd[i], 2] - 0.5 * FilamentLen * ma.sin(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 2] = Loc[PtsMvd[i], 1] + 0.5 * FilamentLen * ma.cos(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        Endpts[PtsMvd[i], 3] = Loc[PtsMvd[i], 2] + 0.5 * FilamentLen * ma.sin(
                            ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                        LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                                              (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
                        TrackerPl = 1
                    else:
                        j += 1
                if TrackerPl == 0:
                    PlNF[PtsMvd[i], 0], PlNF[PtsMvd[i], 1] = random.randrange(-9, 9, 2), random.randrange(-9, 9, 2)
                    Endpts[PtsMvd[i], 0] = Loc[PtsMvd[i], 1] - 0.5 * FilamentLen * ma.cos(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 1] = Loc[PtsMvd[i], 2] - 0.5 * FilamentLen * ma.sin(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 2] = Loc[PtsMvd[i], 1] + 0.5 * FilamentLen * ma.cos(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    Endpts[PtsMvd[i], 3] = Loc[PtsMvd[i], 2] + 0.5 * FilamentLen * ma.sin(
                        ma.atan2(PlNF[PtsMvd[i], 1], PlNF[PtsMvd[i], 0]))
                    LineSeg[PtsMvd[i]] = [(Endpts[PtsMvd[i], 0], Endpts[PtsMvd[i], 1]),
                                          (Endpts[PtsMvd[i], 2], Endpts[PtsMvd[i], 3])]
    # Update individual probability matrices
    for i in range(len(PtsMvd)):
        if State[PtsMvd[i]] == 1 or State[PtsMvd[i]] == 2:  # MT or actin
            # Count number of same neighbors
            SameNB = 0
            for j in range(NumNB1):
                if State[PtsMvd[i]] == State[NB1[PtsMvd[i], j]]:
                    SameNB += 1
            for j in range(NumNB2):
                if State[PtsMvd[i]] == State[NB2[PtsMvd[i], j]]:
                    SameNB += 1
            # Define force matrix for neighbor slots
            NBForce[:, 0] = NetF[PtsMvd[i], 0]
            NBForce[:, 1] = NetF[PtsMvd[i], 1]
            # Calculate gamma constant
            GConstMot[PtsMvd[i]] = 0
            if State[PtsMvd[i]] == 1:  # MT
                if KinesinPerMT != 0:
                    for j in range(NumNB1):
                        if State[j] == 1:
                            GConstMot[PtsMvd[i]] += GammaMotKin * KinesinPerMT  # gamma motor
                    for j in range(NumNB2):
                        if State[j] == 1:
                            GConstMot[PtsMvd[i]] += GammaMotKin * KinesinPerMT  # gamma motor
                GConstTot[PtsMvd[i]] = GammaFilMT + GConstMot[PtsMvd[i]] + GammaCrosslinkMT * SameNB  # gamma total
            elif State[PtsMvd[i]] == 2:  # actin
                if MyosinPerActin != 0:
                    for j in range(NumNB1):
                        if State[j] == 2:
                            GConstMot[PtsMvd[i]] += GammaMotMyo * MyosinPerActin  # gamma motor
                    for j in range(NumNB2):
                        if State[j] == 2:
                            GConstMot[PtsMvd[i]] += GammaMotMyo * MyosinPerActin  # gamma motor
                GConstTot[PtsMvd[i]] = GammaFilAct + GConstMot[PtsMvd[i]] + GammaCrosslinkActin * SameNB  # gamma total
            # Calculate D constant, sigma constant, mu constant, and individual probability
            # D -- D = kBT/gamma
            DConst[PtsMvd[i]] = kBT / GConstTot[PtsMvd[i]]
            # Sigma -- sigma = sqrt(2*D*timestep)
            SigmaConst[PtsMvd[i]] = ma.sqrt(2 * DConst[PtsMvd[i]] * TimeStep)
            for j in range(NumNB1):
                # Mu -- mu = (F/gamma)*timestep
                MuConst[PtsMvd[i], j] = (np.dot(NBForce[j], Delta[j]) / GConstTot[PtsMvd[i]]) * TimeStep
                # Individual probability -- P = 1-0.5*(1+erf((L-mu)/(sigma*sqrt(2))))
                IndivProb[PtsMvd[i], NB1[PtsMvd[i], j]] = 1 - 0.5 * (
                            1 + ma.erf((GridDist - MuConst[PtsMvd[i], j]) / (SigmaConst[PtsMvd[i]] * ma.sqrt(2))))
        else:
            for j in range(NumNB1):
                if State[PtsMvd[i]] == 3 and State[NB1[PtsMvd[i], j]] != 3:
                    IndivProb[PtsMvd[i], NB1[PtsMvd[i], j]] = 1
                elif State[PtsMvd[i]] == 3 and State[NB1[PtsMvd[i], j]] == 3:
                    IndivProb[PtsMvd[i], NB1[PtsMvd[i], j]] = 0
    # Update combined probability matrices
    for i in range(len(PtsMvd)):
        for j in range(NumNB1):
            CombProb[PtsMvd[i], NB1[PtsMvd[i], j]] = IndivProb[PtsMvd[i], NB1[PtsMvd[i], j]] \
                                                     * IndivProb[NB1[PtsMvd[i], j], PtsMvd[i]]
            CombProb[NB1[PtsMvd[i], j], PtsMvd[i]] = IndivProb[PtsMvd[i], NB1[PtsMvd[i], j]] \
                                                     * IndivProb[NB1[PtsMvd[i], j], PtsMvd[i]]
    # Update raw probability array
    for i in range(len(PtsMvd)):
        ProbAllMoveIDs[PtsMvd[i], 0] = CombProb[PtsMvd[i], NB1[PtsMvd[i], 1]]
        ProbAllMoveIDs[PtsMvd[i], 1] = CombProb[PtsMvd[i], NB1[PtsMvd[i], 0]]
        ProbAllMoveIDs[PtsMvd[i], 2] = CombProb[PtsMvd[i], NB1[PtsMvd[i], 5]]
    # Update normalized probability array
    if np.sum(ProbAllMoveIDs) < 1:
        print("Warning: Sum of all probabilities < 1: ", np.sum(ProbAllMoveIDs))
    NormProbAllMoveIDs = ProbAllMoveIDs / np.sum(ProbAllMoveIDs)

# Save final condition and move log
np.save(CaseNum + CaseIteration + 'State_Fin', State)
np.save(CaseNum + CaseIteration + 'LineSeg_Fin', LineSeg)

# Measure distribution analytics after simulation
MTDist.fill(0), ActDist.fill(0), MTCoDist.fill(0), ActCoDist.fill(0)
MTRadCt.fill(0), ActRadCt.fill(0), MTRadSubCt.fill(0), ActRadSubCt.fill(0), MTCoCt.fill(0), ActCoCt.fill(0)
MTRadF, ActRadF, MTCoF, ActCoF = np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn), np.zeros(NumAnn)
NumMTInBounds, NumActinInBounds = 0, 0
for i in range(TotPts):
    if State[i] == 1 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        for j in range(TotPts):
            if State[j] == 1:
                MTDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
            elif State[j] == 2:
                MTCoDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
    elif State[i] == 2 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        for j in range(TotPts):
            if State[j] == 2:
                ActDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
            elif State[j] == 1:
                ActCoDist[i, j] = ma.sqrt((Loc[j, 1] - Loc[i, 1]) ** 2 + (Loc[j, 2] - Loc[i, 2]) ** 2)
for i in range(TotPts):
    if State[i] == 1 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        NumMTInBounds += 1
        for j in range(NumAnn):
            for k in range(TotPts):
                if State[k] == 1 and j * GridDist < round(MTDist[i, k], 1) <= (j + 1) * GridDist:
                    MTRadCt[i, j] += 1
                elif State[k] == 2 and j * GridDist < round(MTCoDist[i, k], 1) <= (j + 1) * GridDist:
                    MTCoCt[j] += 1
    elif State[i] == 2 and (NumAnn + 1) * GridDist <= Loc[i, 1] <= (NumColumns * GridDist - (NumAnn + 1) * GridDist) \
            and NumAnn * GridDist <= Loc[i, 2] <= (NumRows / (2 * ma.tan(ma.pi / 6)) * GridDist - NumAnn * GridDist):
        NumActinInBounds += 1
        for j in range(NumAnn):
            for k in range(TotPts):
                if State[k] == 2 and j * GridDist < round(ActDist[i, k], 1) <= (j + 1) * GridDist:
                    ActRadCt[i, j] += 1
                elif State[k] == 1 and j * GridDist < round(ActCoDist[i, k], 1) <= (j + 1) * GridDist:
                    ActCoCt[j] += 1
for i in range(TotPts):
    for j in range(NumAnn):
        MTRadSubCt[j] += MTRadCt[i, j]
        ActRadSubCt[j] += ActRadCt[i, j]
for i in range(NumAnn):
    MTRadF[i] = (MTRadSubCt[i] / NumMTInBounds) / (RatioMT * (i + 1) * NumNB1)
    ActRadF[i] = (ActRadSubCt[i] / NumActinInBounds) / (RatioActin * (i + 1) * NumNB1)
    MTCoF[i] = (MTCoCt[i] / NumMTInBounds) / (RatioActin * (i + 1) * NumNB1)
    ActCoF[i] = (ActCoCt[i] / NumActinInBounds) / (RatioMT * (i + 1) * NumNB1)

# Save final distribution analytics
np.save(CaseNum + CaseIteration + 'MTRad_Fin', MTRadF)
np.save(CaseNum + CaseIteration + 'ActRad_Fin', ActRadF)
np.save(CaseNum + CaseIteration + 'MTCo_Fin', MTCoF)
np.save(CaseNum + CaseIteration + 'ActCo_Fin', ActCoF)

End = time.time()
print(round((End - Start) / 60), "minutes")
