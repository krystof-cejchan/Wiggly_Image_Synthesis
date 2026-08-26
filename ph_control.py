"""Conditioning on any pH, including values the model was never trained on.

THE PROBLEM
The dataset covers pH 5.8-8.8 and the model is conditioned through a Fourier feature
embedding, which is periodic. Feeding it a pH outside the trained range does not
extrapolate - it wraps. Measured on the actual embedding used by the v2 checkpoint:

    requested pH   ||emb - emb(8.8)||   nearest in-range pH it resembles
        8.8              0.00            8.8
        9.4              6.26            7.42
        9.8              7.59            7.82
       10.5              5.14            8.52
       11.8              5.41            6.84     <- same distance as pH 5.8

So `--target_pH 9.8` does not ask for "wavier than 8.8"; it lands somewhere arbitrary, and
by pH 11.8 the embedding is as far from 8.8 as pH 5.8 is. Asking for a more alkaline
filament could hand back a straighter one. Clamping the input would be honest but would cap
the tool at 8.8, which is the thing we are trying to get past.

THE APPROACH
Extrapolate in velocity space instead of in embedding space. The model knows the two ends
of the range well, so the difference between them is a direction meaning "become wavier":

    v = v(8.8) + lambda * ( v(8.8) - v(5.8) )

At lambda = 0 this is exactly the ordinary pH 8.8 velocity, so nothing about in-range
behaviour changes and the response stays continuous across the boundary. This is the same
manoeuvre as classifier-free guidance, and the contrastive_scale sweep already demonstrated
empirically that pushing along this direction increases waviness monotonically.

Extrapolation is not new knowledge - the model never saw a pH 11 filament and cannot know
what one looks like. What it produces is "the pH 5.8 -> 8.8 trend, continued". Whether that
matches real chemistry beyond 8.8 is a question only new microscopy can answer.

CALIBRATION
lambda is not chosen by feel. waviness.py measures the RMS centreline excursion of real
crops, which rises roughly linearly with pH; calibrate_ph.py measures how the generator's
output waviness responds to lambda, and inverts the two curves so that a requested pH maps
to the lambda that reproduces the physically extrapolated waviness. Constants live in
ph_calibration.json so a new checkpoint can be recalibrated without touching this file.
"""
import json
import os

import torch

from config import PH_MIN, PH_MAX

# The anchors are the dataset's endpoints: the two pH values the model knows best.
PH_ANCHOR_LO, PH_ANCHOR_HI = PH_MIN, PH_MAX
ANCHOR_SPAN = PH_ANCHOR_HI - PH_ANCHOR_LO

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ph_calibration.json")

# Fallbacks used only when ph_calibration.json is absent. WAVINESS_* describe the real
# data; LAMBDA_PER_PX is how much lambda buys one pixel of extra RMS excursion.
_DEFAULTS = {
    "waviness_slope": 1.093,      # rms_dev = slope*pH + intercept, fitted over 277 crops
    "waviness_intercept": -3.108,
    "lambda_gain": 1.0,           # 1.0 = the natural "range-widths past the anchor" scale
    "max_lambda": 3.0,
}


def _calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as fh:
            return {**_DEFAULTS, **json.load(fh)}
    return dict(_DEFAULTS)


def normalize_pH(pH):
    """Map the trained pH range onto [-1, 1]. Out-of-range values are NOT meaningful here;
    they are handled by extrapolation, never by feeding them through the embedding."""
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1


def predicted_waviness(pH):
    """RMS centreline excursion in pixels that a real filament at this pH would have,
    from the linear fit over the measured dataset, continued past the ends.

    Floored at 0.5px: the straight-line fit crosses zero near pH 2.8 and would go negative
    below that, but even a perfectly straight filament traces with some excursion.
    """
    cal = _calibration()
    return max(0.5, cal["waviness_slope"] * pH + cal["waviness_intercept"])


def ph_to_lambda(pH_query):
    """Signed extrapolation strength for a requested pH. 0 inside the trained range.

    The mapping is the natural one - lambda counts range-widths past the anchor, so
    lambda = 1 means "as far below pH 5.8 as pH 5.8 is below pH 8.8", i.e. pH 2.8. An
    earlier version derived lambda from a measured waviness response instead; that
    measurement turned out to be saturated by the img2img anchor and is not trustworthy,
    and the natural mapping has the advantage of meaning something without calibration.

    Only the negative (acidic, straighter) branch is actually used by the pipeline. Pushing
    positive is available but does NOT produce wavier filaments - see ph_warp.py, which
    handles that direction geometrically instead.
    """
    if PH_ANCHOR_LO <= pH_query <= PH_ANCHOR_HI:
        return 0.0
    cal = _calibration()
    anchor = PH_ANCHOR_HI if pH_query > PH_ANCHOR_HI else PH_ANCHOR_LO
    lam = (pH_query - anchor) / ANCHOR_SPAN * cal["lambda_gain"]
    limit = cal["max_lambda"]
    return float(max(-limit, min(limit, lam)))


def anchor_velocities(model, x, t):
    """The velocity fields at the two ends of the trained range."""
    lo = torch.full((x.shape[0],), normalize_pH(PH_ANCHOR_LO), device=x.device)
    hi = torch.full((x.shape[0],), normalize_pH(PH_ANCHOR_HI), device=x.device)
    return model(x, t, lo), model(x, t, hi)


def extrapolate(v_lo, v_hi, lam, rescale=True):
    """Push past an anchor along the acidic->alkaline direction.

    lam > 0 continues past the alkaline anchor, lam < 0 past the acidic one. The rescale
    restores the base anchor's standard deviation: extrapolation inflates the velocity
    norm, and left unchecked that compounds over the whole trajectory into the same
    blown-out texture that an over-aggressive guidance scale produces.
    """
    direction = v_hi - v_lo
    base = v_hi if lam >= 0 else v_lo
    v = base + abs(lam) * (direction if lam >= 0 else -direction)
    if rescale:
        v = v * (base.std() / v.std().clamp(min=1e-8))
    return v


def velocity_for_pH(model, x, t, pH_query, rescale=True, lam_override=None, waviness=None):
    """Velocity field at any pH, extrapolating beyond the trained range when needed.

    Inside [5.8, 8.8] this is a single ordinary conditional model call and behaves exactly
    as before. Outside, it costs two calls and blends them (embedding-space extrapolation) -
    UNLESS `waviness` is given, which takes over instead: pH is pinned to the nearest
    trained anchor (texture/chemistry fidelity) and the requested geometry goes straight
    into the model's own waviness conditioning (see model.py's WavinessEmbedding), which -
    unlike this function's embedding-space extrapolation - has no periodic component to
    alias and was built specifically to extrapolate past its training range without
    wrapping. Requires a checkpoint trained with the waviness-conditioned ConditionalUNet;
    a checkpoint predating that silently ignores `waviness` via forward()'s default (None),
    so this argument is always safe to pass, it just does nothing on an old checkpoint.
    """
    if waviness is not None:
        anchor = min(max(pH_query, PH_MIN), PH_MAX)
        ph = torch.full((x.shape[0],), normalize_pH(anchor), device=x.device)
        wav = torch.full((x.shape[0],), float(waviness), device=x.device)
        return model(x, t, ph, wav)
    lam = ph_to_lambda(pH_query) if lam_override is None else lam_override
    if lam == 0.0:
        ph = torch.full((x.shape[0],), normalize_pH(pH_query), device=x.device)
        return model(x, t, ph)
    v_lo, v_hi = anchor_velocities(model, x, t)
    return extrapolate(v_lo, v_hi, lam, rescale=rescale)


def describe(pH_query, geometry_mode="warp"):
    """One-line summary for CLI output, so an extrapolated request is never silent.

    geometry_mode must match whatever the caller is actually about to run (img2img.py's and
    sample.py's own --geometry_mode) - this function has no way to know which mechanism was
    selected otherwise, and describing the wrong one is worse than describing neither. Which
    mechanism applies inside "warp"/"embedding" mode depends on the direction, because those
    were measured to work asymmetrically - see ph_warp.py for the numbers behind that split.
    "native" is symmetric across direction (it's the same conditioning pathway either way)
    but is NOT YET VALIDATED - flagged as such every time it's named here on purpose.
    """
    if PH_ANCHOR_LO <= pH_query <= PH_ANCHOR_HI:
        return f"pH {pH_query:g} is inside the trained range - direct conditioning"
    direction = "BELOW" if pH_query < PH_ANCHOR_LO else "ABOVE"
    if geometry_mode == "native":
        anchor = min(max(pH_query, PH_ANCHOR_LO), PH_ANCHOR_HI)
        return (f"pH {pH_query:g} is {direction} the trained range [{PH_ANCHOR_LO:g}, "
                f"{PH_ANCHOR_HI:g}] - NATIVE waviness conditioning: anchoring pH at "
                f"{anchor:g}, targeting {predicted_waviness(pH_query):.1f}px directly "
                f"through the model's own conditioning (mechanism not yet validated)")
    if pH_query < PH_ANCHOR_LO:
        return (f"pH {pH_query:g} is BELOW the trained range [{PH_ANCHOR_LO:g}, "
                f"{PH_ANCHOR_HI:g}] - extrapolating the velocity field past the acidic "
                f"anchor with lambda={ph_to_lambda(pH_query):.2f} (straighter)")
    return (f"pH {pH_query:g} is ABOVE the trained range [{PH_ANCHOR_LO:g}, "
            f"{PH_ANCHOR_HI:g}] - conditioning at pH {PH_ANCHOR_HI:g}, then imposing "
            f"waviness {predicted_waviness(pH_query):.1f}px geometrically "
            f"(velocity extrapolation does not work in this direction)")
