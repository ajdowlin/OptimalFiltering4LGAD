import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import calib_waveform
import pedestals
import analyze_waveform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy.stats import norm
import traceback

data_path = "/Volumes/T9/LGAD_10kEvents_ch012ref_ch34567sig.csv"
ped_path = "/Volumes/T9/fullDynamicPedestals_isel_3350_3300.csv"

acq_file = Path(data_path)
output_dir = acq_file.parent / f"{acq_file.stem}_results"
output_dir.mkdir(exist_ok=True)

ch_sig = 3
ch_ref = 1
maxEvents = 10000
TOA_min = 0
TOA_max = 50
toPlot = -1
makePlots = True

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


ref_aligned_segments, common_time = analyze_waveform.getSegments_refAligned(datapath = data_path,
                                                                            slopes = slopes,
                                                                            ch_refs = ch_ref,
                                                                            ch_sig = ch_sig,
                                                                            ref_tCalib = [tCalib_ref_even,tCalib_ref_odd],
                                                                            sig_tCalib = [tCalib_sig_even,tCalib_sig_odd],
                                                                            ref_bounds = [30, 5],
                                                                            sig_bounds = [20, 20],
                                                                            dqMax = 0,
                                                                            maxEvents = -1,
                                                                            sampleOff_ref_time = -87,
                                                                            sampleOff_ref_ped = 0,
                                                                            sampleOff_sig_time = 0,
                                                                            sampleOff_sig_ped = 0,
                                                                            printFreq = 1000)
ref_aligned_segments = np.asarray(ref_aligned_segments)
common_time = np.asarray(common_time)

# Mean waveform
ref_mean = np.mean(ref_aligned_segments, axis=0)
# Covariance and variance
ref_covariance = np.cov(ref_aligned_segments, rowvar=False)
ref_variance = np.diag(ref_covariance)
print("Total Variance: ", np.sum(ref_variance))

# Save mean and variance
ref_output = pd.DataFrame({
    "time": common_time,
    "mean": ref_mean,
    "variance": ref_variance
})

ref_output.to_csv(
    output_dir / f"ch{ch_sig}_RefAlignedMeanVariance.csv",
    index=False
)

print("Saved reference-aligned mean and variance.")


ref_eigvals, ref_eigvecs = np.linalg.eigh(ref_covariance)
if makePlots:

    plt.figure(figsize=(8, 6))

    plt.plot(common_time, ref_mean, marker="o", markersize=3)
    
    plt.xlabel("Time [ps]")
    plt.ylabel("Mean amplitude [mV]")
    plt.title(f"Channel {ch_sig} Reference-Aligned Average Waveform")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        output_dir / f"ch{ch_sig}_RefAlignedMean.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    
    plt.plot(common_time, ref_variance, marker="o", markersize=3)

    plt.xlabel("Time [ps]")
    plt.ylabel("Variance [mV$^2$]")
    plt.title(f"Channel {ch_sig} Reference-Aligned Variance")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        output_dir / f"ch{ch_sig}_RefAlignedVariance.png",
        dpi=300
    )
    plt.close()
    
    for i in range(ref_eigvecs.shape[1]):
        plt.figure()
        plt.plot(common_time, ref_eigvecs[:, i], marker="o")
        plt.axhline(y = 0.0,linestyle='--')
        plt.title(
            f"Data Eigenmode {i}, "
            f"{100 * ref_eigvals[i] / ref_eigvals.sum():.2f}% variance"
        )
        plt.xlabel("Time")
        plt.savefig(output_dir / f"DataEigenmode_RefAligned{i}.png")
        plt.close()

max_aligned_segments, common_time = analyze_waveform.getSegments_maxAligned(datapath = data_path,
                                                                            slopes = slopes,
                                                                            ch_sig = ch_sig,
                                                                            sig_tCalib = [tCalib_sig_even,tCalib_sig_odd],
                                                                            sig_bounds = [20, 20],
                                                                            dqMax = 0,
                                                                            maxEvents = -1,
                                                                            sampleOff_sig_time = 0,
                                                                            sampleOff_sig_ped = 0,
                                                                            printFreq = 1000)


max_aligned_segments = np.asarray(max_aligned_segments)
common_time = np.asarray(common_time)

# Mean waveform
max_mean = np.mean(max_aligned_segments, axis=0)
# Covariance and variance
max_covariance = np.cov(max_aligned_segments, rowvar=False)
max_variance = np.diag(max_covariance)
print("Total Variance: ", np.sum(max_variance))

# Save mean and variance
max_output = pd.DataFrame({
    "time": common_time,
    "mean": max_mean,
    "variance": max_variance
})

max_output.to_csv(
    output_dir / f"ch{ch_sig}_MaxAlignedMeanVariance.csv",
    index=False
)

print("Saved maximum-aligned mean and variance.")


max_eigvals, max_eigvecs = np.linalg.eigh(max_covariance)
if makePlots:

    plt.figure(figsize=(8, 6))

    plt.plot(common_time, max_mean, marker="o", markersize=3)
    
    plt.xlabel("Time [ps]")
    plt.ylabel("Mean amplitude [mV]")
    plt.title(f"Channel {ch_sig} Max-Aligned Average Waveform")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        output_dir / f"ch{ch_sig}_MaxAlignedMean.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    
    plt.plot(common_time, max_variance, marker="o", markersize=3)

    plt.xlabel("Time [ps]")
    plt.ylabel("Variance [mV$^2$]")
    plt.title(f"Channel {ch_sig} Maximum-Aligned Variance")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        output_dir / f"ch{ch_sig}_MaxAlignedVariance.png",
        dpi=300
    )
    plt.close()
    
    for i in range(max_eigvecs.shape[1]):
        plt.figure()
        plt.plot(common_time, max_eigvecs[:, i], marker="o")
        plt.axhline(y = 0.0,linestyle='--')
        plt.title(
            f"Data Eigenmode {i}, "
            f"{100 * max_eigvals[i] / max_eigvals.sum():.2f}% variance"
        )
        plt.xlabel("Time")
        plt.savefig(output_dir / f"DataEigenmode_MaxAligned{i}.png")
        plt.close()
