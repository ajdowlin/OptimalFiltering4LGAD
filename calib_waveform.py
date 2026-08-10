"""
calib_waveform.py

Basic waveform preparation utilities:
- apply dynamic pedestals
- splice selected samples
- remove baseline
- return waveform and per-sample quality flags
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pedestals

def removeBadSamples(data,
                     thresh_low,
                     thresh_high,
                     ievnt = 0,
                     v = False):
    dq=np.zeros(len(data))
    for i in range(len(data)):
        if data[i]>thresh_high or data[i]<thresh_low:
            try:
                if data[i+1]<thresh_high and data[i+1]>thresh_low:
                    data[i] = (data[i-1]+data[i+1])/2
                    dq[i] = 1
                else:
                    if(v):
                        print("Event ", ievnt," has 2 or more consecutive saturated/corrupted samples")
                        print("Samples ", i, " to ", i+1, " are saturated/corrupted")
                    data[i] =  0    
                    dq[i] = 2

            except:
                data[i] = 0
                dq[i] = 2

        else:
            dq[i] = 0
        
    return data, dq

def fixTimeAxis(x, startWin, dtEven, dtOdd, sampleOffset=0):
    n_samples = len(x)

    if n_samples == 0:
        return np.array([])

    dtEven = np.asarray(dtEven, dtype=float)
    dtOdd = np.asarray(dtOdd, dtype=float)

    if len(dtEven) != 64 or len(dtOdd) != 64:
        raise ValueError("dtEven and dtOdd must each contain 64 values.")
    dt_pattern = np.concatenate((dtEven, dtOdd))

    start_index = (startWin * 64 + sampleOffset) % len(dt_pattern)

    # Interval following each sample.
    interval_indices = (
        start_index + np.arange(n_samples - 1)
    ) % len(dt_pattern)

    x_new = np.zeros(n_samples)
    x_new[1:] = np.cumsum(dt_pattern[interval_indices])
    
    return x_new / 100

def loadTimeCalibration(path):
    dt = pd.read_csv(Path(path)).iloc[:, -1].to_numpy(dtype=float)
    if len(dt) != 64:
        raise ValueError(f"{path} contains {len(dt)} values; expected 64.")
    return dt

def splice_waveform(
        waveform,
        times,
        flags,
        left,
        right,
        invert = False,
):

    waveform = np.asarray(waveform, dtype=float)
    maximum_index = int(np.nanargmax(waveform))
    if invert:
        maximum_index = int(np.nanargmin(waveform))

    start = maximum_index - left
    stop = maximum_index + right + 1

    if start < 0 or stop > waveform.size:
        raise ValueError(
            "Requested splice extends outside the waveform: "
            f"maximum index={maximum_index}, requested range=[{start}:{stop}], "
            f"waveform length={waveform.size}"
        )

    return waveform[start:stop].copy(), times[start:stop].copy(), flags[start:stop].copy()

def removeBaseline(data, startSample, stopSample):
    return data-np.average(data[startSample:stopSample])

def process_waveform(raw_waveform,
                     channelADC,
                     startWindow,
                     channel,
                     baseline = None,
                     badSample_range = None,
                     waveform_bounds = None,
                     tCalib = None,
                     invert = False,
                     flagMax = 0,
                     ievnt = 0,
                     sampleOff_time = 0,
                     sampleOff_ped = 0,
                     v=False,
                     returnIndices = False
                     ):

    # 1st Calibrate the ADC 
    # channelADC is either a
    #  - list of average slopes for channels 0-7: converts ADC to mV based on average ADC/mV response
    #    - use getPedestalSlopes(pedPath, plots=True, avgSlope=True)
    #  - 8x32640 array of individual pipeline slopes: converts ADC to mV based on per pipeline measured ADC [[[PREFERRED]]]
    #    - use - use getPedestalSlopes(pedPath, plots=True, avgSlope=False)
    
    # 2nd Remove Baseline (Optional)
    # baseline: 2 value array of [lower sample, upper sample]
    # average of this range is subtracted from samples

    # 3rd Flag Bad/Corrupted Samples (Optional)
    # badSample_range: 2 value array of [upper limit, lower limit]
    # samples outside of this range flagged as bad
    # - replace bad sample with interpolated value of neighboring samples (flag = 1)
    # - if neighbor also bad sample: replace bad sample with 0 (flag = 2)
    # - other samples are okay (flag = 0)
    # if badSample_range is None, no samples are flagged as bad (all flags = 0)

    # 4th Splice and Assign Time Axis 
    # waveform_bounds: 2 value array of [samples before max, samples after max]
    # only samples within this range of the maximum sample are kept
    # if waveform_bounds is None, no splicing occurs
    # timingCalib = 2x64 array of [even time bins, odd time bins]
    # gives waveform custom time axis

    #returns
    # waveform: waveform after calibrations have been applied
    # times: time axis of waveform relative to window that was triggered on
    # flags: array of flag values for corresponding waveform samples
    # badWaveform:
    #  - False if spliced waveform has number bad samples < flagMax
    
    try:
        if channelADC is not None:
            if isinstance(channelADC, pd.DataFrame):
                nPoints = len(raw_waveform)
                indices = (
                    startWindow * 64
                    + sampleOff_ped
                    + np.arange(nPoints, dtype=int)
                )
                
                waveform = pedestals.ADCmVbyPipelineSlope(
                    raw_waveform,
                    indices,
                    channelADC,
                    channel
                )

            else:
                channelADC = np.asarray(channelADC, dtype=float)

            if channelADC.ndim == 1:
                waveform = raw_waveform / channelADC[channel]
                
            elif channelADC.ndim > 1:
                nPoints = len(raw_waveform)
                indices  = np.array([startWindow*64 + i + sampleOff_ped for i in range(nPoints)])
                waveform = pedestals.ADCmVbyPipelineSlope(raw_waveform, indices, channelADC, channel)
        else:
            waveform = np.asarray(raw_waveform) / 4.0
            #print(f"Warning: ADC not calibrated. Samples returned in ADC/4 for channel {channel} event {ievnt}")
        if baseline is not None:
            waveform = removeBaseline(waveform, baseline[0], baseline[1])
        if badSample_range is not None:
            waveform, flags = removeBadSamples(waveform, badSample_range[0], badSample_range[1],ievnt,v)
        else:
            flags = np.zeros(waveform.size())
        if waveform_bounds is not None:
            times = np.asarray(range(len(waveform)))
            if tCalib is not None:
                times = fixTimeAxis(times, startWindow, tCalib[0], tCalib[1], sampleOffset = sampleOff_time)
            waveform, times,  flags = splice_waveform(waveform, times, flags, waveform_bounds[0], waveform_bounds[1], invert)
        badWaveform = False
        
    except Exception as e:
        badWaveform = True
        print(f"Error occurred: {e} at waveform ", ievnt)  
        return raw_waveform, range(len(raw_waveform)), np.zeros(len(raw_waveform)), badWaveform
                                   
    if np.max(flags) > flagMax:
        print("Waveform ", ievnt," flagged as bad for out of range samples")
        badWaveform = True

    return waveform, times, flags, badWaveform


