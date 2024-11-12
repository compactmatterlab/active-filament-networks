# Import required packages
import numpy as np
import math as ma
import xlsxwriter

# Inputs
Case = 'SPH-K1-025'
Freq = 0.25  # Hz
PeakForce = 100  # pN

# Load simulation results
XDisp = np.load(Case + 'XDisp_Avg.npy')

# Time and force arrays
Time, Force = np.zeros(len(XDisp)), np.zeros(len(XDisp))
for i in range(len(XDisp)):
    Time[i] = i * 0.01
    Force[i] = PeakForce * ma.sin(Freq * 2 * ma.pi * Time[i])

# Trim data
XDispTrim = np.zeros(len(XDisp) - 100)
TimeTrim = np.zeros(len(Time) - 100)
ForceTrim = np.zeros(len(Force) - 100)
for i in range(len(XDispTrim)):
    XDispTrim[i] = XDisp[i + 100]
    TimeTrim[i] = Time[i + 100]
    ForceTrim[i] = Force[i + 100]

FftForce = np.fft.fft(ForceTrim)
NForce = FftForce.size
FftForceNorm = 2 * np.abs(FftForce) / NForce
FreqForce = np.fft.fftfreq(NForce, d=0.01)

FftDisp = np.fft.fft(XDispTrim)
NDisp = FftDisp.size
FftDispNorm = 2 * np.abs(FftDisp) / NDisp
FreqDisp = np.fft.fftfreq(NDisp, d=0.01)

workbook = xlsxwriter.Workbook(Case + '.xlsx')
worksheet = workbook.add_worksheet()

row, col = 0, 0

worksheet.write(0, col, 'Freq F (hz)')
worksheet.write(0, col + 1, 'FFT F (pn)')
worksheet.write(0, col + 2, 'Phi F (rad)')
worksheet.write(0, col + 3, 'Freq S (hz)')
worksheet.write(0, col + 4, 'FFT X (um)')
worksheet.write(0, col + 5, 'Phi X (rad)')
worksheet.write(0, col + 6, 'G prime')
worksheet.write(0, col + 7, 'G dbl prime')
worksheet.write(0, col + 8, 'Phi shift (deg)')

for i in range(len(FftDispNorm)):
    worksheet.write(i + 1, col, FreqForce[i])
    worksheet.write(i + 1, col + 1, FftForceNorm[i])
    worksheet.write(i + 1, col + 2, np.angle(FftForce[i]))
    worksheet.write(i + 1, col + 3, FreqDisp[i])
    worksheet.write(i + 1, col + 4, FftDispNorm[i])
    worksheet.write(i + 1, col + 5, np.angle(FftDisp[i]))
    worksheet.write(i + 1, col + 6, (
                (FftForceNorm[i] / FftDispNorm[i]) * ma.cos(np.angle(FftForce[i]) - np.angle(FftDisp[i])) * (
                    1 / 0.625)))
    worksheet.write(i + 1, col + 7, (
                (FftForceNorm[i] / FftDispNorm[i]) * ma.sin(np.angle(FftForce[i]) - np.angle(FftDisp[i])) * (
                    1 / 0.625)))
    worksheet.write(i + 1, col + 8, (np.angle(FftForce[i]) - np.angle(FftDisp[i])) * (180 / ma.pi))

workbook.close()
