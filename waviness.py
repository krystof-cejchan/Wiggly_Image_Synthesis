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


# ---------------------------------------------------------------------------
# Spectral shape of the centreline: how the wiggle is distributed over scales.
#
# waviness() is a single RMS and wave_period() is a single spectral PEAK, and between
# them they still do not pin down what a filament looks like. Measured on this dataset,
# real crops spread their centreline energy almost evenly over every scale (per-band rms
# at pH 8.8: 3.62 drift / 2.75 at 96-192px / 2.52 at 48-96 / 2.83 at 24-48), whereas the
# period-conditioned checkpoint puts nearly all of it in one long wave and draws almost
# nothing fine (6.39 / 8.60 / 2.22 / 0.73 for the same request). Both have a similar peak
# and a similar-order total, so neither existing number can tell them apart - which is
# exactly why conditioning on them produced one huge wave where the data has many small
# ones.
#
# RIPPLE_{MIN,MAX}_WAVELENGTH bound the band that is BOTH physically real and reliably
# measurable, verified directly by drawing synthetic fibres on real background:
#   * a dead-straight fibre traces to total 0.99px, of which 0.71 sits below 24px - that
#     band is dominated by per-column tracer jitter, so including it would put a large
#     pH-independent constant into the label and dilute the signal.
#   * a pure 64px sinusoid of rms 3.0 reports 2.44px in 48-96 and 2.77px total, i.e. the
#     band lands the energy where it belongs at close to the right magnitude.
#   * a pure 160px sinusoid leaks most of its energy into the >192px bin, so the upper
#     edge is kept at 96px rather than pushed toward the drift scale.
RIPPLE_MIN_WAVELENGTH = 24.0
RIPPLE_MAX_WAVELENGTH = 96.0


def _detrended_trace(ref_image, min_len=128):
    """The traced centreline resampled onto a contiguous integer grid, linearly detrended.

    Shared by every spectral measurement below so they cannot disagree about which signal
    they are describing. Returns (ys, n) or None when the fibre is too short/unreadable to
    resolve the ripple band at all.
    """
    traced = trace_fibre(ref_image)
    if traced is None:
        return None
    x, y = traced
    xs = np.arange(int(x.min()), int(x.max()) + 1)
    n = len(xs)
    if n < min_len:
        return None
    ys = np.interp(xs, x, y)
    grid = np.arange(n)
    return ys - np.polyval(np.polyfit(grid, ys, 1), grid), n


def band_rms(ref_image, min_wavelength, max_wavelength):
    """RMS of the centreline restricted to one wavelength band, in pixels, or None.

    Parseval: filtering in the Fourier domain and taking the RMS of the inverse transform
    gives the part of the total excursion that lives at these scales, in the same physical
    units as waviness(), so the bands of a trace add up in quadrature to its total.
    """
    got = _detrended_trace(ref_image)
    if got is None:
        return None
    ys, n = got
    spec = np.fft.rfft(ys)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # freq 0 is the (already removed) mean; guard the division rather than warn on it
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    keep = (periods >= min_wavelength) & (periods <= max_wavelength)
    filtered = np.zeros_like(spec)
    filtered[keep] = spec[keep]
    return float(np.sqrt(np.mean(np.fft.irfft(filtered, n=n) ** 2)))


def band_profile(ref_image, bands):
    """Every band's rms from a SINGLE trace, plus the total and the ripple.

    band_rms() re-traces the fibre on each call, which is the expensive part (the FFT is
    free by comparison), so asking it for five bands plus waviness plus ripple traces the
    same image seven times. `bands` is a sequence of (min_wavelength, max_wavelength, name).
    Returns {name: rms, ..., "total": rms, "ripple": rms} or None.

    The caller's bands are treated as HALF-OPEN, [min, max), so adjacent bands sharing an
    edge do not both claim the bin sitting exactly on it - with band_rms()'s inclusive bounds
    a contiguous set of bands summed in quadrature to 1.08x the total rather than to it.
    "ripple" keeps band_rms()'s inclusive convention, because it is the training LABEL and
    has to stay bit-identical to ripple_rms().

    "total" is the rms of the same resampled, detrended trace the bands come from, so the two
    are consistent by construction. It agrees with waviness() on the median crop but can
    differ sharply on one whose trace has long unconfident gaps, since _detrended_trace
    interpolates across them and waviness() simply skips them.
    """
    got = _detrended_trace(ref_image)
    if got is None:
        return None
    ys, n = got
    spec = np.fft.rfft(ys)
    freqs = np.fft.rfftfreq(n, d=1.0)
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)

    def rms_between(lo, hi, inclusive=False):
        keep = ((periods >= lo) & (periods <= hi) if inclusive
                else (periods >= lo) & (periods < hi))
        filtered = np.zeros_like(spec)
        filtered[keep] = spec[keep]
        return float(np.sqrt(np.mean(np.fft.irfft(filtered, n=n) ** 2)))

    out = {name: rms_between(lo, hi) for lo, hi, name in bands}
    out["total"] = float(np.sqrt(np.mean(ys ** 2)))
    out["ripple"] = rms_between(RIPPLE_MIN_WAVELENGTH, RIPPLE_MAX_WAVELENGTH,
                                inclusive=True)
    return out


def ripple_rms(ref_image):
    """How much of the filament's excursion is FINE undulation, in pixels, or None.

    The second geometry scalar the model is conditioned on, and the one that makes "many
    small waves" expressible: waviness() alone is satisfied just as well by a single
    frame-wide arc, which is the cheaper thing to draw and therefore what a model told only
    the total produces. Conditioning on the total AND this together pins the balance,
    because everything outside the ripple band is then whatever is left over in quadrature.

    Replaces wave_period() in that role. The period was a spectral PEAK - one bin, no
    magnitude - and measured on the trained checkpoint it turned out to carry no usable
    gradient at all: sweeping the requested period across its full 50-400px range moved the
    velocity field by 0.05-0.15%, against 1-5.7% for the waviness channel. An rms in a fixed
    band is a magnitude in physical units, on the same footing as the waviness label that
    does work.
    """
    return band_rms(ref_image, RIPPLE_MIN_WAVELENGTH, RIPPLE_MAX_WAVELENGTH)
