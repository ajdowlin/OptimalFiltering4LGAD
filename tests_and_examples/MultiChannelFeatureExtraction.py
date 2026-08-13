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
ped_path = "/Users/andrewdowling/OptimalFiltering4LGAD/calib_files/fullDynamicPedestals_isel_3350_3300.csv"

chs_sig = [3, 4, 5, 6]
chs_ref = [1]
xLocs = [1250, 1750, 1250, 1750] # x locations of LGAD pad for signal channels
yLocs = [250, 250, 750, 750]# y locations of LGAD pad for signal channels 

maxEvents = 3000
toPlot = -1
refilter = True
printFreq = 500

tOffsets_sig = [0,0,0,0] #Offsets for time calib axis
tOffsets_ref = [-87] #Offsets for time calib axis

TOA_min = 34
TOA_max = 50

print("Loading pedestals...")
slopes = pedestals.getPedestalSlopes(ped_path,plots=True,avgSlope = False)

tCalibs_sig_even = []
tCalibs_sig_odd = []

tCalibs_ref_even = []
tCalibs_ref_odd = []

acq_file = Path(data_path)
output_dir = acq_file.parent / f"{acq_file.stem}_resolutions"
output_dir.mkdir(exist_ok=True)

print("Loading time calibration for Signal Channel(s)")
for ch_sig in chs_sig:
    
    tCalib_even_sig = f"/Users/andrewdowling/OptimalFiltering4LGAD/calib_files/Ch{ch_sig}_-1000Thresh_EvenWinCalib.csv"
    tCalib_odd_sig = f"/Users/andrewdowling/OptimalFiltering4LGAD/calib_files/Ch{ch_sig}_-1000Thresh_OddWinCalib.csv"
    
    df_even_sig = pd.read_csv(tCalib_even_sig)
    tCalib_sig_even = df_even_sig["bin_width_ps"].to_numpy(dtype=float)
    
    df_odd_sig = pd.read_csv(tCalib_odd_sig)
    tCalib_sig_odd = df_odd_sig["bin_width_ps"].to_numpy(dtype=float)

    tCalibs_sig_even.append(tCalib_sig_even)
    tCalibs_sig_odd.append(tCalib_sig_odd)
    
print("Loading time calibration for Reference Channel(s)")
for ch_ref in chs_ref:
    
    tCalib_even_ref = f"/Users/andrewdowling/OptimalFiltering4LGAD/calib_files/Ch{ch_ref}_-1000Thresh_EvenWinCalib.csv"
    tCalib_odd_ref = f"/Users/andrewdowling/OptimalFiltering4LGAD/calib_files/Ch{ch_ref}_-1000Thresh_OddWinCalib.csv"
    
    df_even_ref = pd.read_csv(tCalib_even_ref)
    tCalib_ref_even = df_even_ref["bin_width_ps"].to_numpy(dtype=float)
    
    df_odd_ref = pd.read_csv(tCalib_odd_ref)
    tCalib_ref_odd = df_odd_ref["bin_width_ps"].to_numpy(dtype=float)

    tCalibs_ref_even.append(tCalib_ref_even)
    tCalibs_ref_odd.append(tCalib_ref_odd)

print("Loading signal shape templates for Signal Channel(s)")
template_means_sig = []
template_times_sig = []
template_vars_sig = []

for ch_sig in chs_sig:
    waveform_file = pd.read_csv(f"/Volumes/T9/LGAD_10kEvents_ch012ref_ch34567sig_results/ch{ch_sig}_RefAlignedMeanVariance.csv")
    waveform_template = waveform_file["mean"]
    waveform_template_times = waveform_file["time"] - waveform_file["time"][np.argmax(waveform_template)]
    waveform_var = waveform_file["variance"]

    template_means_sig.append(waveform_template)
    template_times_sig.append(waveform_template_times)
    template_vars_sig.append(waveform_var)

print("Reading data...")
df = pd.read_csv(data_path)
if maxEvents == -1:
    events = df["event"].unique()
else:
    events = range(maxEvents)
    
nEvents = len(events)
nSigChannels = len(chs_sig)
nRefChannels = len(chs_ref)

TOAs_OF = np.full((nSigChannels, nEvents), np.nan)
TOAs_CFD = np.full((nSigChannels, nEvents), np.nan)
Amps = np.full((nSigChannels, nEvents), np.nan)
maxVals = np.full((nSigChannels, nEvents), np.nan)

TOAs_ref = np.full((nRefChannels, nEvents), np.nan)


print("Beginning loop over data")

for event_num in events:
    event_df = df[df["event"] == event_num]
    try:
        start_window = df[df["event"] == event_num]["start_window"].iloc[0]
    except Exception as e:
        print(f"Error occurred: {e} at waveform ", event_num) 
        print("can't find start window evnt: ", event_num)
        continue
    if event_num%printFreq==0:
            print("Processing Event Number: ", event_num)

    bad_ref = False
    bad_sig = False
    for i, ch_ref in enumerate(chs_ref):
        try:
            raw_waveform_ref = event_df[f"ch{ch_ref}"].to_numpy()
            waveform_ref, times_ref, flags_ref, bad_ref = calib_waveform.process_waveform(raw_waveform_ref,
                                                                                          slopes,
                                                                                          start_window,
                                                                                          ch_ref,
                                                                                          baseline = [0,50],
                                                                                          badSample_range = [-600, 4000],
                                                                                          waveform_bounds = [30, 5],
                                                                                          tCalib = [tCalibs_ref_even[i],tCalibs_ref_odd[i]],
                                                                                          invert = True,
                                                                                          flagMax = 0,
                                                                                          ievnt = event_num,
                                                                                          sampleOff_time= tOffsets_ref[i],
                                                                                          sampleOff_ped = 0,
                                                                                          v=False
                                                                                          )
            TOA_ref = CFD.calcTOA_inv("fixed",times_ref,waveform_ref,0,-150,1.0,v=False)
        except Exception as e:
            print(f"Reference error occurred: {e} at waveform ", event_num, "Ch: ", ch_ref)
            traceback.print_exc()
            continue
        if bad_ref:
            print("Reference calibration error occurred: at waveform ", event_num, "Ch: ", ch_ref)
            continue
        TOAs_ref[i][event_num] = TOA_ref

    for i, ch_sig in enumerate(chs_sig):
        try:
            raw_waveform_sig = event_df[f"ch{ch_sig}"].to_numpy()
            waveform_sig, times_sig, flags_sig, bad_sig = calib_waveform.process_waveform(raw_waveform_sig,
                                                                                          slopes,
                                                                                          start_window,
                                                                                          ch_sig,
                                                                                          baseline = [0,50],
                                                                                          badSample_range = [-1000, 400],
                                                                                          waveform_bounds = [15, 15],
                                                                                          tCalib = [tCalibs_sig_even[i],tCalibs_sig_odd[i]],
                                                                                          invert = False,
                                                                                          flagMax = 0,
                                                                                          ievnt = event_num,
                                                                                          sampleOff_time = tOffsets_sig[i],
                                                                                          sampleOff_ped = 0,
                                                                                          v=False
                                                                                          )
            TOA_sig_CFD = CFD.calcTOA_inv('calc', times_sig, -waveform_sig, 0,  0, 0.88, v=False)
            
            center_idx = np.argmax(waveform_sig)
            center_time = times_sig[center_idx]
            maxVal = np.max(waveform_sig)
            times_sig_centered = times_sig - center_time
            weights = make_filter.fromTemplateFilter(template_means_sig[i], template_times_sig[i], times_sig_centered, baseline = False)
            Amp = np.dot(weights[0],waveform_sig)
            AmpTau = np.dot(weights[1],waveform_sig)
            Tau = -AmpTau/Amp
            TOA_sig = center_time+Tau
            if abs(Tau) > 0.75 and refilter:
                new_center_idx = np.argmin(np.abs(times_sig - TOA_sig))
                new_center_time = times_sig[new_center_idx]
                '''
                print(
                    "Refiltering event:", event_num,
                    "sample shift:", new_center_idx - center_idx,
                    "old Tau:", Tau
                )
                '''
                times_sig_centered = times_sig - new_center_time
                
                weights = make_filter.fromTemplateFilter(template_means_sig[i], template_times_sig[i], times_sig_centered, baseline = False)
            
                Amp = np.dot(weights[0], waveform_sig)
                AmpTau = np.dot(weights[1], waveform_sig)
                Tau = -AmpTau / Amp
                
                TOA_sig = new_center_time + Tau
                #print("new Tau:", Tau)
                
        except Exception as e:
            print(f"Signal error occurred: {e} at waveform ", event_num, "Ch: ", ch_sig)
            traceback.print_exc()
            continue
        if bad_sig:
            print("Signal calibration error occurred: at waveform ", event_num, "Ch: ", ch_sig)
            continue
            
        TOAs_CFD[i][event_num] = TOA_sig_CFD
        TOAs_OF[i][event_num] = TOA_sig
        Amps[i][event_num] = Amp
        maxVals[i][event_num] = maxVal

    
print("Finished loop over events")
print("Computing Individual Channel Time Resolutions")
for i,ch_sig in enumerate(chs_sig):
    OF_TOA = []
    CFD_TOA = []
    for event_num in events:        
        TOA_ref = np.nanmean(TOAs_ref[:, event_num])
        TOA_sig_OF = TOAs_OF[i, event_num]
        TOA_sig_CFD = TOAs_CFD[i, event_num]
        if np.isfinite(TOA_ref) and np.isfinite(TOA_sig_OF) and np.isfinite(TOA_sig_CFD):
            TOA_OF = -TOA_sig_OF + TOA_ref
            TOA_CFD = -TOA_sig_CFD + TOA_ref
            if TOA_OF > TOA_min and TOA_OF < TOA_max:
                OF_TOA.append(TOA_OF)
                CFD_TOA.append(TOA_CFD)
            else:
                print("TOA out of range, signal channel: ", ch_sig)
        else:
            print("Signal or Reference is nan for event: ", event_num, ", skipping")
            continue

    plt.figure(figsize=(8,6))
    counts, bins, _ = plt.hist(OF_TOA, bins=50, alpha=0.7, label="Data")
    mu, sigma = norm.fit(OF_TOA)
    x = np.linspace(bins[0], bins[-1], 1000)
    bin_width = bins[1] - bins[0]
    y = norm.pdf(x, mu, sigma) * len(OF_TOA) * bin_width
    plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
    plt.xlabel("Samples [100ps]")
    plt.ylabel("Counts")
    print(f"Ch{ch_sig} TOA from Filter STD: ",np.std(OF_TOA))
    print(f"Ch{ch_sig} TOA from Filter Mean: ",np.mean(OF_TOA))
    plt.legend()
    plt.savefig(
        output_dir / f"ch{ch_sig}_OF_TOA_Resolution.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8,6))
    counts, bins, _ = plt.hist(CFD_TOA, bins=50, alpha=0.7, label="Data")
    mu, sigma = norm.fit(CFD_TOA)
    x = np.linspace(bins[0], bins[-1], 1000)
    bin_width = bins[1] - bins[0]
    y = norm.pdf(x, mu, sigma) * len(CFD_TOA) * bin_width
    plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
    plt.xlabel("Samples [100ps]")
    plt.ylabel("Counts")
    print(f"Ch{ch_sig} TOA from CFD STD: ",np.std(CFD_TOA))
    print(f"Ch{ch_sig} TOA from CFD Mean: ",np.mean(CFD_TOA))
    plt.legend()
    plt.savefig(
        output_dir / f"ch{ch_sig}_CFD_TOA_Resolution.png",
        dpi=300
    )
    plt.close()

    
OF_TOA = []
CFD_TOA = []
print("Computing resolution when combining channel TOA info")
for event_num in events:
    OF_TOA_ch = []
    CFD_TOA_ch = []
    OF_AMP_ch = []
    maxVal_AMP_ch = []
    TOA_ref = np.nanmean(TOAs_ref[:, event_num])
    if np.isnan(TOA_ref):
        print("Reference is nan for event: ", event_num, ", skipping")
        continue
    for i,ch_sig in enumerate(chs_sig):
        TOA_sig_OF = TOAs_OF[i, event_num]
        TOA_sig_CFD = TOAs_CFD[i, event_num]
        Amp_sig_OF = Amps[i,event_num]
        Amp_sig_maxVal = maxVals[i,event_num]

        OF_TOA_ch.append(TOA_sig_OF)
        CFD_TOA_ch.append(TOA_sig_CFD)
        OF_AMP_ch.append(Amp_sig_OF)
        maxVal_AMP_ch.append(Amp_sig_maxVal)

    if np.isnan(OF_TOA_ch).any() or np.isnan(TOA_ref):
        print("Signal or Reference is nan for event: ", event_num, ", skipping")
        continue
    avgTOA_OF = np.dot(OF_TOA_ch, OF_AMP_ch)/np.sum(OF_AMP_ch)
    avgTOA_CFD = np.dot(CFD_TOA_ch, maxVal_AMP_ch)/np.sum(maxVal_AMP_ch)
    #avgTOA_OF = np.dot(OF_TOA_ch, np.ones(len(chs_sig)))/np.sum(np.ones(len(chs_sig)))
    #avgTOA_CFD = np.dot(CFD_TOA_ch, np.ones(len(chs_sig)))/np.sum(np.ones(len(chs_sig)))
    
    TOA_OF = -avgTOA_OF + TOA_ref
    TOA_CFD = -avgTOA_CFD + TOA_ref
    
    if TOA_OF > TOA_min and TOA_OF < TOA_max:
        OF_TOA.append(TOA_OF)
        CFD_TOA.append(TOA_CFD)
    
        
        
        
plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(OF_TOA, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(OF_TOA)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(OF_TOA) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Samples [100ps]")
plt.ylabel("Counts")
print(f"TOA from Filter STD: ",np.std(OF_TOA))
print(f"TOA from Filter Mean: ",np.mean(OF_TOA))
plt.legend()
plt.savefig(
    output_dir / f"combined_OF_TOA_Resolution.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(CFD_TOA, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(CFD_TOA)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(CFD_TOA) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Samples [100ps]")
plt.ylabel("Counts")
print(f"TOA from CFD STD: ",np.std(CFD_TOA))
print(f"TOA from CFD Mean: ",np.mean(CFD_TOA))
plt.legend()
plt.savefig(
    output_dir / f"combined_CFD_TOA_Resolution.png",
    dpi=300
)
plt.close()
        
        

x_vals_OF = []
y_vals_OF = []

x_vals_maxVal = []
y_vals_maxVal = []
    
for event_num in events:
    OF_AMP_ch = []
    maxVal_AMP_ch = []
    TOA_ref = np.nanmean(TOAs_ref[:, event_num])
    if np.isnan(TOA_ref):
        print("Reference is nan for event: ", event_num, ", skipping")
        continue
    for i,ch_sig in enumerate(chs_sig):
        Amp_sig_OF = Amps[i,event_num]
        Amp_sig_maxVal = maxVals[i,event_num]

        OF_AMP_ch.append(Amp_sig_OF)
        maxVal_AMP_ch.append(Amp_sig_maxVal)

    if np.isnan(OF_AMP_ch).any() or np.isnan(maxVal_AMP_ch).any():
        print("Signal or Reference is nan for event: ", event_num, ", skipping")
        continue
    avgX_OF = np.dot(xLocs, OF_AMP_ch)/np.sum(OF_AMP_ch)
    avgY_OF = np.dot(yLocs, OF_AMP_ch)/np.sum(OF_AMP_ch)

    avgX_maxVal = np.dot(xLocs, maxVal_AMP_ch)/np.sum(maxVal_AMP_ch)
    avgY_maxVal = np.dot(yLocs, maxVal_AMP_ch)/np.sum(maxVal_AMP_ch)
    
    x_vals_OF.append(avgX_OF)
    y_vals_OF.append(avgY_OF)

    x_vals_maxVal.append(avgX_maxVal)
    y_vals_maxVal.append(avgY_maxVal)


plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(x_vals_OF, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(x_vals_OF)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(x_vals_OF) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("X value (microns?)")
plt.ylabel("Counts")
print(f"x value from Filter STD: ",np.std(x_vals_OF))
print(f"x value from Filter Mean: ",np.mean(x_vals_OF))
plt.legend()
plt.savefig(
    output_dir / f"combined_OF_X_Resolution.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(y_vals_OF, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(y_vals_OF)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(y_vals_OF) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Y value (microns?)")
plt.ylabel("Counts")
print(f"y value from Filter STD: ",np.std(y_vals_OF))
print(f"y value from Filter Mean: ",np.mean(y_vals_OF))
plt.legend()
plt.savefig(
    output_dir / f"combined_OF_Y_Resolution.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(x_vals_maxVal, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(x_vals_maxVal)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(x_vals_maxVal) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("X value (microns?)")
plt.ylabel("Counts")
print(f"x value from maxVal STD: ",np.std(x_vals_maxVal))
print(f"x value from maxVal Mean: ",np.mean(x_vals_maxVal))
plt.legend()
plt.savefig(
    output_dir / f"combined_maxVal_X_Resolution.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(8,6))
counts, bins, _ = plt.hist(y_vals_maxVal, bins=50, alpha=0.7, label="Data")
mu, sigma = norm.fit(y_vals_maxVal)
x = np.linspace(bins[0], bins[-1], 1000)
bin_width = bins[1] - bins[0]
y = norm.pdf(x, mu, sigma) * len(y_vals_maxVal) * bin_width
plt.plot(x, y, lw=2,label=fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
plt.xlabel("Y value (microns?)")
plt.ylabel("Counts")
print(f"y value from maxVal STD: ",np.std(y_vals_maxVal))
print(f"y value from maxVal Mean: ",np.mean(y_vals_maxVal))
plt.legend()
plt.savefig(
    output_dir / f"combined_maxVal_Y_Resolution.png",
    dpi=300
)
plt.close()
