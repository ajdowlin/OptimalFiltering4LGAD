import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import calib_waveform
import pedestals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy.stats import norm
import traceback

#data_path = "/Volumes/T9/Scan_7_16_26/x1981-y28830-board0-nSkippedEvents1.csv"
data_path = "/Volumes/T9/LGAD_10kEvents_ch012ref_ch34567sig.csv"
ped_path = "/Volumes/T9/fullDynamicPedestals_6_17_26.csv"
ch_sig = 7
ch_ref = 2
maxEvents = 1000
TOA_min = 0
TOA_max = 50
events = [5]
channels = [0,3]

slopes = pedestals.getPedestalSlopes(ped_path,plots=True,avgSlope = False)

tCalib_even_sig = "/Volumes/T9/ch0123_timeCalib_2Mevents_7_31_26_capture_1/Ch3_-400Thresh_EvenWinCalib.csv"
tCalib_odd_sig = "/Volumes/T9/ch0123_timeCalib_2Mevents_7_31_26_capture_1/Ch3_-400Thresh_OddWinCalib.csv"

tCalib_even_ref = "/Volumes/T9/ch0123_timeCalib_2Mevents_7_31_26_capture_1/Ch2_-400Thresh_EvenWinCalib.csv"
tCalib_odd_ref =  "/Volumes/T9/ch0123_timeCalib_2Mevents_7_31_26_capture_1/Ch2_-400Thresh_OddWinCalib.csv"

df_even_ref = pd.read_csv(tCalib_even_ref)
tCalib_ref_even = df_even_ref["bin_width_ps"].to_numpy(dtype=float)

df_odd_ref = pd.read_csv(tCalib_odd_ref)
tCalib_ref_odd = df_odd_ref["bin_width_ps"].to_numpy(dtype=float)

df_even_sig = pd.read_csv(tCalib_even_sig)
tCalib_sig_even = df_even_sig["bin_width_ps"].to_numpy(dtype=float)

df_odd_sig = pd.read_csv(tCalib_odd_sig)
tCalib_sig_odd = df_odd_sig["bin_width_ps"].to_numpy(dtype=float)



df = pd.read_csv(data_path)
plt.figure(figsize=(8,6))

df = pd.read_csv(data_path)

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
    for ch in channels:
        try:
            raw_waveform_sig = event_df[f"ch{ch}"].to_numpy()
            inv = False
            if ch in [0,1,2]:
                inv =True
                
            waveform_sig, times_sig, flags_sig, bad_sig = calib_waveform.process_waveform(raw_waveform_sig,
                                                                                          slopes,
                                                                                          start_window,
                                                                                          ch,
                                                                                          baseline = [0,50],
                                                                                          badSample_range = [-500, 5000],
                                                                                          waveform_bounds = [20, 20],
                                                                                          tCalib = None,#[tCalib_sig_even,tCalib_sig_odd],
                                                                                          invert = inv,
                                                                                          flagMax = 0,
                                                                                          ievnt = event_num,
                                                                                          sampleOff_time = 0,
                                                                                          sampleOff_ped = 0,
                                                                                          v=False
                                                                                          )


            plt.plot(times_sig, waveform_sig, label = f'ch{ch}')
            plt.xlabel("Samples (100ps)")
            plt.ylabel("mV")
            
        except:
            print(f"bad waveform, ch{ch}, event{event_num}")
            continue
plt.legend()
plt.show()
