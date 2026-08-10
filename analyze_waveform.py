"""
analyze_waveform.py

Waveform analysis:
- Retrieve waveform segments
- Determine template shape

PCA Analysis (planned)
"""

import calib_waveform
from CFD import calcTOA_inv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from scipy.interpolate import CubicSpline

def getSegments_refAligned(datapath,
                           slopes,
                           ch_refs,
                           ch_sig,
                           ref_tCalib,
                           sig_tCalib,
                           ref_bounds,
                           sig_bounds,
                           dqMax = 0,
                           maxEvents = -1,
                           sampleOff_ref_time = 0,
                           sampleOff_ref_ped = 0,
                           sampleOff_sig_time = 0,
                           sampleOff_sig_ped = 0,
                           printFreq = 1000):

    """
    Aligns waveforms with reference pulse
    returns 

    datapath: pathname of data file
    pedpath: pedestal array (per-channel only slopes or per-channel-per-pipeline slopes)
    ch_ref: reference channel nos. list of channels or int (one channel)
    ch_sig: signal channel no.
    ref_tCalib: reference channel timing calibration
    sig_tCalib: signal channel timing calibration
    ref_bounds: 2 value list of [left, right]
    sig_bounds: 2 value list of [left, right]
     - left: points to the left of waveform max to keep
     - right: points to the right of waveform max to keep
    dqMax: reject events with samples that have a flagged sample > dqMax
    maxEvents: max events to run over
     - set to -1 to run over all events
    sampleOff: sample offset needed to use calibration files
    
    """
    print("Loading data...")
    acq_file = Path(datapath)
    df = pd.read_csv(datapath)
    print("Loaded data")

    segments_sig = []
    segments_sig_time = []
    
    if maxEvents == -1:
        events = df["event"].unique()
    else:
        events = range(maxEvents)

    print(f"Begin event loop over {len(events)} events")
    for event_num in events:

        #Extract Event No. and Start Window
        
        event_df = df[df["event"] == event_num]
        start_window = event_df["start_window"].iloc[0]
        if event_num % printFreq == 0:
            print("Processing Event Number: ", event_num)

        #Calibration of Signal Channel
        
        raw_waveform_sig = event_df[f"ch{ch_sig}"].to_numpy()
        waveform_sig, times_sig, flags_sig, bad_sig = calib_waveform.process_waveform(raw_waveform_sig,
                                                                                      slopes,
                                                                                      start_window,
                                                                                      ch_sig,
                                                                                      baseline = [0,50],
                                                                                      badSample_range = [-1000, 2000],
                                                                                      waveform_bounds = sig_bounds,
                                                                                      tCalib = sig_tCalib,
                                                                                      invert = False,
                                                                                      flagMax = 0,
                                                                                      ievnt = event_num,
                                                                                      sampleOff_time = sampleOff_sig_time,
                                                                                      sampleOff_ped = sampleOff_sig_ped,
                                                                                      v=False
                                                                                      )
        if bad_sig:
            print(f"Bad signal event. Channel: {ch_sig}, Event: {event_num}")
            continue

        #Calibration of Reference Channel(s)
        
        if isinstance(ch_refs, int):
            ch_refs = [ch_refs]
            ref_bounds = [ref_bounds]
            sampleOff_ref_time = [sampleOff_ref_time]
            sampleOff_ref_ped = [sampleOff_ref_ped]
            ref_tCalib = [ref_tCalib]
            
        elif not isinstance(ch_refs, list):
            print("Reference channel(s) must be given as int or list")

        TOAs_ref = []
        bad_refs = False
        for i, ch_ref in enumerate(ch_refs):
            raw_waveform_ref = event_df[f"ch{ch_ref}"].to_numpy()
            waveform_ref, times_ref, flags_ref, bad_ref = calib_waveform.process_waveform(raw_waveform_ref,
                                                                                          slopes,
                                                                                          start_window,
                                                                                          ch_ref,
                                                                                          baseline = [0,50],
                                                                                          badSample_range = [-1000, 2000],
                                                                                          waveform_bounds = ref_bounds[i],
                                                                                          tCalib = ref_tCalib[i],
                                                                                          invert = True,
                                                                                          flagMax = 0,
                                                                                          ievnt = event_num,
                                                                                          sampleOff_time = sampleOff_ref_time[i],
                                                                                          sampleOff_ped = sampleOff_ref_ped[i],
                                                                                          v=False
                                                                                          )
            if bad_ref:
                print(f"Bad reference event. Channel: {ch_ref}, Event: {event_num}")
                bad_refs = True
                continue

            #Timing of Reference Channel(s)
            try:
                TOA_ref = calcTOA_inv("fixed", times_ref, waveform_ref, 0, -150, 1.0, v=False)
            except:
                bad_ref = True
            if bad_ref:
                print(f"Bad reference event. Channel: {ch_ref}, Event: {event_num}")
                bad_refs = True
                continue
            else:
                TOAs_ref.append(TOA_ref)

        if bad_refs:
            print(f"Some reference event(s). Event: {event_num}")
            continue
        TOA = np.average(TOAs_ref)

        #Shift signal time axis by arrival time of reference pulse
        times_sig = times_sig - TOA
        segments_sig.append(waveform_sig)
        segments_sig_time.append(times_sig)

    
    #Done with loop...
    #Interpolate signal waveforms onto common time axis for analysis

    segments_time = np.asarray(segments_sig_time)
    segments = np.asarray(segments_sig)
    
    avg_time_first = np.mean(segments_time[:, 0])
    avg_time_last  = np.mean(segments_time[:, -1])
    
    print("Average time at index 0: ", avg_time_first)
    print("Average time at last index: ", avg_time_last)
    avg_time_first = int(round(avg_time_first))
    avg_time_last = int(round(avg_time_last))
    print("Start time for time axis: ", avg_time_first)
    print("Last time for time axis: ", avg_time_last)
    print("Length of output time axis: ",avg_time_last - avg_time_first, "samples")
    print("Length of waveforms used: ",sig_bounds[1] + sig_bounds[0], "samples")
    

    #Define Common Time Axis
    common_time = np.linspace(
        avg_time_first,
        avg_time_last,
        sig_bounds[1] + sig_bounds[0]+1
    )

    #Interpolate segments_sig onto common time
    
    segments_common_time = np.vstack([
        np.interp(common_time, time_axis, waveform)
        for waveform, time_axis in zip(segments, segments_time)
    ])
    '''
    segments_common_time = np.vstack([
        CubicSpline(time_axis, waveform)(common_time)
        for waveform, time_axis in zip(segments, segments_time)
    ])
    '''
    return segments_common_time, common_time




    
        
            
def getSegments_maxAligned(datapath,
                           slopes,
                           ch_sig,
                           sig_tCalib,
                           sig_bounds,
                           dqMax = 0,
                           maxEvents = -1,
                           sampleOff_sig_time = 0,
                           sampleOff_sig_ped = 0,
                           printFreq = 1000):

    print("Loading data...")
    acq_file = Path(datapath)
    df = pd.read_csv(datapath)
    print("Loaded data")

    segments_sig = []
    segments_sig_time = []
    
    if maxEvents == -1:
        events = df["event"].unique()
    else:
        events = range(maxEvents)

    print(f"Begin event loop over {len(events)} events")
    for event_num in events:

        #Extract Event No. and Start Window
        
        event_df = df[df["event"] == event_num]
        start_window = event_df["start_window"].iloc[0]
        if event_num % printFreq == 0:
            print("Processing Event Number: ", event_num)

        #Calibration of Signal Channel
        
        raw_waveform_sig = event_df[f"ch{ch_sig}"].to_numpy()
        waveform_sig, times_sig, flags_sig, bad_sig = calib_waveform.process_waveform(raw_waveform_sig,
                                                                                      slopes,
                                                                                      start_window,
                                                                                      ch_sig,
                                                                                      baseline = [0,50],
                                                                                      badSample_range = [-1000, 2000],
                                                                                      waveform_bounds = sig_bounds,
                                                                                      tCalib = sig_tCalib,
                                                                                      invert = False,
                                                                                      flagMax = 0,
                                                                                      ievnt = event_num,
                                                                                      sampleOff_time = sampleOff_sig_time,
                                                                                      sampleOff_ped = sampleOff_sig_ped,
                                                                                      v=False
                                                                                      )
        if bad_sig:
            print(f"Bad signal event. Channel: {ch_sig}, Event: {event_num}")
            continue
        #align times with waveform argmax
        times_sig = times_sig - times_sig[np.argmax(waveform_sig)]
        segments_sig.append(waveform_sig)
        segments_sig_time.append(times_sig)

    #Done with loop...
    #Interpolate signal waveforms onto common time axis for analysis

    segments_time = np.asarray(segments_sig_time)
    segments = np.asarray(segments_sig)
    
    avg_time_first = np.mean(segments_time[:, 0])
    avg_time_last  = np.mean(segments_time[:, -1])
    
    print("Average time at index 0: ", avg_time_first)
    print("Average time at last index: ", avg_time_last)
    avg_time_first = int(round(avg_time_first))
    avg_time_last = int(round(avg_time_last))
    print("Start time for time axis: ", avg_time_first)
    print("Last time for time axis: ", avg_time_last)
    print("Length of output time axis: ",avg_time_last - avg_time_first, "samples")
    print("Length of waveforms used: ",sig_bounds[1] + sig_bounds[0], "samples")
    

    #Define Common Time Axis
    common_time = np.linspace(
        avg_time_first,
        avg_time_last,
        sig_bounds[1] + sig_bounds[0]+1
    )

    #Interpolate segments_sig onto common time
    
    segments_common_time = np.vstack([
        np.interp(common_time, time_axis, waveform)
        for waveform, time_axis in zip(segments, segments_time)
    ])
    '''
    segments_common_time = np.vstack([
        CubicSpline(time_axis, waveform)(common_time)
        for waveform, time_axis in zip(segments, segments_time)
    ])
    '''
    return segments_common_time, common_time
