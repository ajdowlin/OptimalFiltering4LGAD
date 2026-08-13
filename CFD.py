import numpy as np

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
