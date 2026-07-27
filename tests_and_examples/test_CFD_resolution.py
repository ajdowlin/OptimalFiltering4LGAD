import calib_waveform
import pedestals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy.stats import norm
import traceback

def calcTOA_inv(method, pulsex, pulsey, n, peak, perc, v=False):
    """
    Falling-edge CFD TOA using a linear fit around the threshold crossing.

    n = number of extra points on each side of the crossing.
        n=0 -> 2-point fit
        n=1 -> 4-point fit
        n=2 -> 6-point fit
    """

    pulsex = np.asarray(pulsex)
    pulsey = np.asarray(pulsey)

    peakLoc = np.argmin(pulsey)

    if method == "fixed":
        percPeak = peak * perc
    elif method == "calc":
        peakLeft = max(0, peakLoc - 2)
        peakRight = min(len(pulsey), peakLoc + 1)
        amp = np.average(pulsey[peakLeft:peakRight])
        percPeak = amp * perc
    else:
        raise ValueError("method must be 'fixed' or 'calc'")

    # Find falling-edge threshold crossing
    indexRight = 0
    while indexRight < len(pulsey) and pulsey[indexRight] > percPeak:
        indexRight += 1

    if indexRight == 0 or indexRight >= len(pulsey):
        raise ValueError("Threshold crossing not found or at boundary.")

    indexLeft = indexRight - 1

    # Fit range: n extra points left of indexLeft and right of indexRight
    fitStart = max(0, indexLeft - n)
    fitStop  = min(len(pulsey), indexRight + n + 1)

    xFit = pulsex[fitStart:fitStop]
    yFit = pulsey[fitStart:fitStop]

    if len(xFit) < 2:
        raise ValueError("Not enough points for linear fit.")

    # y = m*x + b
    slope, yintercept = np.polyfit(xFit, yFit, 1)

    if slope == 0:
        raise ValueError("Linear fit slope is zero.")

    TOA = (percPeak - yintercept) / slope

    if v:
        return TOA, indexRight, indexLeft, percPeak, slope, yintercept, xFit, yFit
    else:
        return TOA

    
data_path = "/Volumes/T9/Scan_7_16_26/x1981-y28830-board0-nSkippedEvents1.csv"
ped_path = "/Volumes/T9/fullDynamicPedestals_6_17_26.csv"
ch_sig = 2
ch_ref = 0
maxEvents = -1
TOA_min = 0
TOA_max = 50
toPlot = -1

slopes = pedestals.getPedestalSlopes(ped_path,plots=True,avgSlope = False)

tCalib_even_sig = "/Volumes/T9/Ch2_98511Events_EvenWinCalib.csv" #"/Volumes/T9/Ch3_98602Events_EvenWinCalib.csv"
tCalib_odd_sig =  "/Volumes/T9/Ch2_100059Events_OddWinCalib.csv" #"/Volumes/T9/Ch3_99968Events_OddWinCalib.csv"

tCalib_even_ref = "/Volumes/T9/Ch0_496643Events_EvenWinCalib.csv"
tCalib_odd_ref =  "/Volumes/T9/Ch0_503400Events_OddWinCalib.csv"

df_even_ref = pd.read_csv(tCalib_even_ref)
tCalib_ref_even = df_even_ref["bin_width_ps"].to_numpy(dtype=float)

df_odd_ref = pd.read_csv(tCalib_odd_ref)
tCalib_ref_odd = df_odd_ref["bin_width_ps"].to_numpy(dtype=float)


tCalib_sig_even = calib_waveform.loadTimeCalibration(tCalib_even_sig)
tCalib_sig_odd = calib_waveform.loadTimeCalibration(tCalib_odd_sig)



df = pd.read_csv(data_path)
events = []
TOAs = []
if maxEvents == -1:
    events = df["event"].unique()
else:
    events = range(maxEvents)

for event_num in events:
    event_df = df[df["event"] == event_num]
    try:
        start_window = df[df["event"] == event_num]["start_window"].iloc[0]
    except Exception as e:
        print(f"Error occurred: {e} at waveform ", event_num) 
        print("can't find start window evnt: ", event_num)
        continue

    if event_num%100==0:
            print("Processing Event Number: ", event_num)
    try:
        raw_waveform_sig = event_df[f"ch{ch_sig}"].to_numpy()
        waveform_sig, times_sig, flags_sig, bad_sig = calib_waveform.process_waveform(raw_waveform_sig,
                                                                                      slopes,
                                                                                      start_window,
                                                                                      ch_sig,
                                                                                      baseline = [0,50],
                                                                                      badSample_range = [-1000, 2000],
                                                                                      waveform_bounds = [20, 20],
                                                                                      tCalib = [tCalib_sig_even,tCalib_sig_odd],
                                                                                      invert = False,
                                                                                      flagMax = 0,
                                                                                      ievnt = event_num,
                                                                                      sampleOff = 200,
                                                                                      v=False
                                                                                      )
        raw_waveform_ref = event_df[f"ch{ch_ref}"].to_numpy()
        waveform_ref, times_ref, flags_ref, bad_ref = calib_waveform.process_waveform(raw_waveform_ref,
                                                                                      slopes,
                                                                                      start_window,
                                                                                      ch_ref,
                                                                                      baseline = [0,50],
                                                                                      badSample_range = [-600, 4000],
                                                                                      waveform_bounds = [30, 5],
                                                                                      tCalib = [tCalib_ref_even,tCalib_ref_odd],
                                                                                      invert = True,
                                                                                      flagMax = 0,
                                                                                      ievnt = event_num,
                                                                                      sampleOff= 200,
                                                                                      v=False
                                                                                      )

        if event_num == toPlot:
            plt.figure(figsize=(8,6))
            plt.plot(times_ref, waveform_ref, label = 'ref')
            plt.plot(times_sig, waveform_sig, label = 'sig')
            plt.xlabel("Samples (100ps)")
            plt.ylabel("mV")
            plt.legend()
            print(times_ref)
            print(times_sig)
            #print(waveform_ref)
            #print(waveform_sig)
            plt.show()
        
        TOA_sig_CFD = calcTOA_inv('calc', times_sig, -waveform_sig, 0,  0, 0.65, v=False)
        #TOA_ref = calcTOA_inv('calc',times_ref,waveform_ref,0, 0, 0.65,v=False)
        TOA_ref = calcTOA_inv("fixed",times_ref,waveform_ref,0,-125,1.0,v=False)
        
    except Exception as e:
        bad_sig = True
        bad_ref = True
        print(f"Error occurred: {e} at waveform ", event_num)

    if bad_sig or bad_ref:
        print("bad sig or ref evnt: ", event_num)
        continue

    TOA = abs(TOA_sig_CFD - TOA_ref)
    if TOA > TOA_min and TOA < TOA_max:
        TOAs.append(TOA)
    else:
        print("TOA out of range: ", event_num, "absolute of TOA: ", TOA)

plt.figure(figsize=(8,6))
        
counts, bins, _ = plt.hist(TOAs, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(TOAs)

x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(TOAs) * bin_width

plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
print(np.std(TOAs))
plt.legend()
plt.show()
        
        
