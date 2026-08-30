"""Measure how wavy a microtubule is.

This is the measuring instrument the pH extrapolation is calibrated against. pH is not
something the model can be told about beyond the range it was trained on, but waviness IS
directly measurable in both the real crops and the generated ones - so waviness is what
links a requested pH to a guidance strength.

Measured on the dataset (crops >= 128x32, one dominant fibre traced per crop):

    pH    n   mean rms_dev(px)   tortuosity
    5.8   27       3.92            1.270
    6.4   19       3.14            1.171
    6.8   76       4.14            1.270
    7.2   46       4.31            1.246
    7.4   46       5.04            1.284
    7.8   27       6.74            1.444
    8.8   36       6.29            1.359

Pearson r between pH and rms_dev is +0.84, so the physical claim - higher pH means a wavier
filament - is supported by the data and can be extrapolated with a straight line.
"""
import numpy as np
import torch
import torch.nn.functional as F


def box_blur(img, k=3):
    """Small box blur - keeps per-pixel noise from dominating the per-column argmin."""
    pad = k // 2
    return F.avg_pool2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"), k, stride=1)


def trace_fibre(ref_image, min_confident_fraction=0.35, smooth=5):
    """Return (x, y) of the dominant fibre's centreline, detrended, or None if unreadable.

    The crop's bounding box is arbitrary and the filament may be tilted inside it, so a
    linear trend is removed: only the wiggle around the local axis is physical.
    """
    sm = box_blur(ref_image, 3)[0, 0]
    background = sm.median(dim=0).values
    darkest, path = sm.min(dim=0)
    depth = background - darkest

    strong = torch.quantile(depth, 0.75)
    if strong <= 0:
        return None
    confident = depth > 0.5 * strong
    if float(confident.float().mean()) < min_confident_fraction:
        return None

    xs = torch.nonzero(confident).flatten()
    if xs.numel() < 40:
        return None

    x = xs.detach().cpu().float().numpy()
    y = path[xs].detach().cpu().float().numpy()

    half = smooth // 2
    y = np.array([np.median(y[max(0, i - half):i + half + 1]) for i in range(len(y))])

    y = y - np.polyval(np.polyfit(x, y, 1), x)
    return x, y


def waviness(ref_image):
    """Scalar waviness of one image, or None. Higher = wavier.

    rms_dev is the RMS excursion of the centreline from its own axis, in pixels. It is the
    metric the calibration uses: it responds smoothly, it is in physical units, and unlike
    tortuosity it does not blow up when the trace is noisy.
    """
    traced = trace_fibre(ref_image)
    if traced is None:
        return None
    x, y = traced
    return float(np.sqrt(np.mean(y ** 2)))


def wave_period(ref_image, min_rms=1.5):
    """Dominant undulation period of the filament in pixels, or None. Lower = more waves.

    This is the SPECTRAL PEAK of the traced centreline - deliberately not a turning-point
    count and not a power-weighted mean period. The per-column trace is jittery, and both of
    those estimators are dominated by that jitter rather than by the undulation the eye
    follows: measured against the real crops they reported period RISING with pH and rated a
    single-arc generated image as wavier than a real many-waved one, both of which are wrong.
    The peak gives 311px at pH 5.8 falling to 136px at pH 8.8, agreeing with the independent
    measurement in ph_warp.py's header (304 -> 144).

    Why it is needed at all: waviness() is an RMS deviation, and a wave of amplitude A scores
    A/sqrt(2) whether it completes one arc across the crop or ten. The local slope and
    curvature, though, scale as A/L - so among all geometries meeting a requested rms, one
    long arc is by far the cheapest to draw, and a model told only the rms will produce
    exactly that. Conditioning on the period as well is what makes "many small waves"
    expressible at all.

    Returns None when the filament is too straight for a period to mean anything (rms below
    min_rms), so the label goes to the null embedding rather than carrying pure noise.

    The estimate is bounded by the crop width - a 400px crop cannot show a 600px period - so
    values near the frame width are truncated rather than resolved.
    """
    traced = trace_fibre(ref_image)
    if traced is None:
        return None
    x, y = traced
    xs = np.arange(int(x.min()), int(x.max()) + 1)
    n = len(xs)
    if n < 128:
        return None
    ys = np.interp(xs, x, y)
    if float(np.sqrt(np.mean(ys ** 2))) < min_rms:
        return None
    ys = ys - np.polyval(np.polyfit(np.arange(n), ys, 1), np.arange(n))
    spectrum = np.abs(np.fft.rfft(ys * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spectrum[0] = 0.0                       # the DC bin is the mean, already removed
    if spectrum.sum() <= 0:
        return None
    peak = freqs[int(np.argmax(spectrum))]
    return float(1.0 / peak) if peak > 0 else None


def waviness_stats(ref_image):
    """Both metrics, for analysis. Returns None when no fibre could be traced."""
    traced = trace_fibre(ref_image)
    if traced is None:
        return None
    x, y = traced
    dy = np.gradient(y, x)
    return {"rms_dev": float(np.sqrt(np.mean(y ** 2))),
            "tortuosity": float(np.mean(np.sqrt(1 + dy ** 2)))}


def orientation_spread(ref_image, smooth=5):
    """Angular spread of the local ridge orientation, in radians. Higher = wavier.

    trace_fibre follows a single dominant centreline, which is the right instrument for a
    real crop but the wrong one for a freshly generated sample: those contain several
    stacked filaments (an artefact of mirror-padded training crops), and a per-column argmin
    hops between them, producing excursions that have nothing to do with waviness.

    This metric needs no trace. A straight filament has essentially one ridge orientation
    everywhere; a wavy one sweeps through a range of angles. Measuring the circular spread
    of the structure-tensor orientation, weighted by edge energy, captures that regardless of
    how many filaments are present.
    """
    img = box_blur(ref_image, 3)
    gy, gx = torch.gradient(img[0, 0])

    def sm(a):
        return box_blur(a[None, None], smooth)[0, 0]

    jxx, jyy, jxy = sm(gx * gx), sm(gy * gy), sm(gx * gy)

    theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy)
    energy = jxx + jyy
    coherence = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / energy.clamp(min=1e-8)
    weight = energy * coherence  # ignore flat noise, trust strongly oriented structure

    total = weight.sum().clamp(min=1e-8)
    c = (weight * torch.cos(2 * theta)).sum() / total
    s = (weight * torch.sin(2 * theta)).sum() / total
    resultant = torch.sqrt(c ** 2 + s ** 2).clamp(1e-6, 1.0)
    return float(torch.sqrt(-2 * torch.log(resultant)))
