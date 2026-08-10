import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import calib_waveform
import pedestals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import norm
import traceback
import CFD
import make_filter

data_path = "/Volumes/T9/LGAD_10kEvents_ch012ref_ch34567sig.csv"
ped_path = "/Volumes/T9/fullDynamicPedestals_isel_3350_3300.csv"
ch_sig = 3
ch_ref = 1
maxEvents = 10000
TOA_min = 36.5
TOA_max = 50
toPlot = -1

slopes = pedestals.getPedestalSlopes(ped_path,plots=True,avgSlope = False)

tCalib_even_sig = "/Volumes/T9/ch0123_timeCalib_2Mevents_8_3_26/Ch3_-1000Thresh_EvenWinCalib.csv"
tCalib_odd_sig = "/Volumes/T9/ch0123_timeCalib_2Mevents_8_3_26/Ch3_-1000Thresh_OddWinCalib.csv"

tCalib_even_ref = "/Volumes/T9/ch0123_timeCalib_2Mevents_8_3_26/Ch1_-1000Thresh_EvenWinCalib.csv"
tCalib_odd_ref =  "/Volumes/T9/ch0123_timeCalib_2Mevents_8_3_26/Ch1_-1000Thresh_OddWinCalib.csv"

df_even_ref = pd.read_csv(tCalib_even_ref)
tCalib_ref_even = df_even_ref["bin_width_ps"].to_numpy(dtype=float)

df_odd_ref = pd.read_csv(tCalib_odd_ref)
tCalib_ref_odd = df_odd_ref["bin_width_ps"].to_numpy(dtype=float)

df_even_sig = pd.read_csv(tCalib_even_sig)
tCalib_sig_even = df_even_sig["bin_width_ps"].to_numpy(dtype=float)

df_odd_sig = pd.read_csv(tCalib_odd_sig)
tCalib_sig_odd = df_odd_sig["bin_width_ps"].to_numpy(dtype=float)

waveform_file = pd.read_csv("/Volumes/T9/LGAD_10kEvents_ch012ref_ch34567sig_results/ch3_MaxAlignedMeanVariance.csv")
waveform_template = waveform_file["mean"]
waveform_template_times = waveform_file["time"]# - waveform_file["time"][np.argmax(waveform_template)]

df = pd.read_csv(data_path)
events = []
TOAs = []
Amps = []
Taus = []
Maxs = []
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
                                                                                      badSample_range = [-1000, 400],
                                                                                      waveform_bounds = [15, 15],
                                                                                      tCalib = [tCalib_sig_even,tCalib_sig_odd],
                                                                                      invert = False,
                                                                                      flagMax = 0,
                                                                                      ievnt = event_num,
                                                                                      sampleOff_time = 0,
                                                                                      sampleOff_ped = 0,
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
                                                                                      sampleOff_time= -87,
                                                                                      sampleOff_ped = 0,
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

        maxTime = times_sig[np.argmax(waveform_sig)]
        maxVal = np.max(waveform_sig)
        times_sig_centered = times_sig - maxTime
        #times_sig_centered = range(len(times_sig)) - np.argmax(waveform_sig)
        weights = make_filter.fromTemplateFilter(waveform_template, waveform_template_times, times_sig_centered, baseline = False)
        Amp = np.dot(weights[0],waveform_sig)
        AmpTau = np.dot(weights[1],waveform_sig)
        Tau = -AmpTau/Amp
        TOA_sig = maxTime+Tau
        TOA_ref = CFD.calcTOA_inv("fixed",times_ref,waveform_ref,0,-150,1.0,v=False)
        
    except Exception as e:
        bad_sig = True
        bad_ref = True
        print(f"Error occurred: {e} at waveform ", event_num)
        traceback.print_exc()

    if bad_sig or bad_ref:
        print("bad sig or ref evnt: ", event_num)
        continue

    TOA = abs(TOA_sig - TOA_ref)
    if TOA > TOA_min and TOA < TOA_max:
        TOAs.append(TOA)
        Amps.append(Amp)
        Taus.append(Tau)
        Maxs.append(maxVal)
    else:
        print("TOA out of range: ", event_num, "absolute of TOA: ", TOA)

plt.figure(figsize=(8,6))
        
counts, bins, _ = plt.hist(TOAs, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(TOAs)

x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(TOAs) * bin_width

plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Samples [100ps]")
plt.ylabel("Counts")
print("Time from Filter STD: ",np.std(TOAs))
print("Time from Filter Mean: ",np.mean(TOAs))
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(Maxs, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(Maxs)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(Maxs) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Maximum Value (mV)")
plt.ylabel("Counts")
print("Maximum Val STD: ",np.std(Maxs))
print("Maximum Val Mean: ",np.mean(Maxs))
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(Amps, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(Amps)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(Amps) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Maximum Value (mV)")
plt.ylabel("Counts")
print("Amplitude from Filter STD: ",np.std(Amps))
print("Amplitude from Filter Mean: ",np.mean(Amps))
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(Taus, bins=50, alpha=0.7, label="Data")
plt.xlabel("Maximum Value (mV)")
plt.ylabel("Counts")
plt.legend()
plt.show()
        
        
