"""
pedestals.py

Analyzes set of pedestal captures
 - Maps ADC to mV values
 - Saves plots of ADC/mV relation
 - Saves plots of pipeline point variances
 - Function for Applying Pedestals
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getPedestalSlopes(pedPath, plots=True, avgSlope=False):
    """
    gets per-channel or per-pipeline-per-channel ADC response
    pedPath: path to pedestal scan file
    plots: whether or not to save some plots of ADC response
    avgSlope:
     - if True: returns 8 value array of slopes, 1 per channel
     - if False: returns 8x(510x64) array of slopes, 1 per channel per pipeline point

    """

    
    pedPath = Path(pedPath)
    df = pd.read_csv(pedPath)

    pattern = re.compile(
        r"^ch(?P<channel>[0-7])_ped_(?P<bias>[+-]?\d+(?:\.\d+)?)mV$",
        re.IGNORECASE
    )

    channel_columns = {ch: [] for ch in range(8)}
    for column in df.columns:
        match = pattern.fullmatch(str(column).strip())
        if match:
            ch = int(match.group("channel"))
            bias = float(match.group("bias"))
            channel_columns[ch].append((bias, column))

    result = df[["point"]].copy()
    average_slopes = np.full(8, np.nan)

    if plots:
        fig1, axes1 = plt.subplots(
            4, 4, figsize=(20, 14), constrained_layout=True
        )

    for ch in range(8):
        bias_columns = sorted(channel_columns[ch])
        bias_mV = np.array([bias for bias, _ in bias_columns])
        columns = [column for _, column in bias_columns]

        adc_values = (
            df[columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

        slopes = _fit_row_slopes(bias_mV, adc_values)
        result[f"ch{ch}_slope"] = slopes
        average_slopes[ch] = np.nanmean(slopes)

        if not plots:
            continue

        row = ch // 2
        col = 2 * (ch % 2)

        ax_response = axes1[row, col]
        ax_hist = axes1[row, col + 1]

        step = max(1, len(adc_values) // 100)
        ax_response.plot(
            bias_mV,
            adc_values[::step].T,
            linewidth=0.8,
            alpha=0.2
        )

        mean_adc = np.nanmean(adc_values, axis=0)
        valid = np.isfinite(mean_adc)
        fit = np.polyfit(bias_mV[valid], mean_adc[valid], 1)

        ax_response.plot(bias_mV, mean_adc, "o", label="Pipeline mean")
        ax_response.plot(
            bias_mV,
            np.polyval(fit, bias_mV),
            label=f"Fit: {fit[0]:.4f} ADC/mV"
        )
        ax_response.set(
            title=f"Channel {ch}: pedestal response",
            xlabel="Bias voltage [mV]",
            ylabel="ADC"
        )
        ax_response.legend(fontsize=8)

        for i, bias in enumerate(bias_mV):
            vals = adc_values[:, i]
            vals = vals[np.isfinite(vals)]
            ax_hist.hist(
                vals,
                bins=50,
                histtype="step",
                label=f"{bias:g} mV"
            )

        ax_hist.set(
            title=f"Channel {ch}: pipeline distributions",
            xlabel="ADC",
            ylabel="Pipeline points"
        )
        ax_hist.legend(fontsize=7)

    if plots:
        diag_png = pedPath.with_name(
            f"{pedPath.stem}_diagnostics.png"
        )
        fig1.savefig(diag_png, dpi=200)
        plt.close(fig1)

        fig2, axes2 = plt.subplots(
            4, 2, figsize=(12, 14), constrained_layout=True
        )

        for ch, ax in enumerate(axes2.flatten()):
            slopes = result[f"ch{ch}_slope"].to_numpy()
            slopes = slopes[np.isfinite(slopes)]

            ax.hist(
                slopes,
                bins=60,
                histtype="stepfilled",
                alpha=0.7
            )
            ax.axvline(
                average_slopes[ch],
                linestyle="--",
                linewidth=1.5,
                label=f"mean = {average_slopes[ch]:.4f}"
            )
            ax.set(
                title=f"Channel {ch}: slope distribution",
                xlabel="Slope [ADC/mV]",
                ylabel="Count"
            )
            ax.legend(fontsize=8)

        slope_png = pedPath.with_name(
            f"{pedPath.stem}_slopeDistributions.png"
        )
        fig2.savefig(slope_png, dpi=200)
        plt.close(fig2)

    return average_slopes if avgSlope else result


def _fit_row_slopes(x, y):
    valid = np.isfinite(y)
    x2 = np.broadcast_to(x, y.shape)

    n = valid.sum(axis=1)
    sum_x = np.where(valid, x2, 0.0).sum(axis=1)
    sum_y = np.where(valid, y, 0.0).sum(axis=1)
    sum_xx = np.where(valid, x2**2, 0.0).sum(axis=1)
    sum_xy = np.where(valid, x2 * y, 0.0).sum(axis=1)

    denom = n * sum_xx - sum_x**2

    return np.divide(
        n * sum_xy - sum_x * sum_y,
        denom,
        out=np.full(y.shape[0], np.nan),
        where=(n >= 2) & (denom != 0)
    )

def ADCmVbyPipelineSlope(data,indices,points, ch):
    """
    calculates mV using an array of per-channel-per-pipeline slopes
    data: raw, uncalibrated array of waveform values
    indices: corresponding pipeline positions of those waveform values
    points: slopes generated using "getPedestalSlopes(pedPath, plots, avgSlope = False)
    ch: DSA board input channel
    """
    data = np.asarray(data, dtype=float)
    indices = np.asarray(indices, dtype=int)

    if data.shape != indices.shape:
        raise ValueError("data and indices must have the same shape")

    slope_col = f"ch{ch}_slope"
    slopes = points[slope_col].to_numpy(dtype=float)

    sample_slopes = slopes[indices % len(slopes)]

    return np.divide(
        data,
        sample_slopes,
        out=np.full(data.shape, np.nan, dtype=float),
        where=np.isfinite(sample_slopes) & (sample_slopes != 0)
    )


