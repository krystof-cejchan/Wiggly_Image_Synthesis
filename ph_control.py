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
import math
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
#
# Two separate waviness fits, at two different scales, because predicted_waviness() and
# predicted_waviness_native() feed two mechanisms that measure waviness completely
# differently:
#   - ph_warp.py's geometric warp compares its target against waviness() measured on the
#     WHOLE real reference image - waviness_slope/intercept, unchanged, whole-image scale.
#   - native conditioning (model.py's WavinessEmbedding) is trained on per-crop labels
#     (train.py's dynamic_collate_fn: a random TRAIN_SIZES crop, flips, jitter) -
#     native_waviness_slope/intercept, fit the same way, on that same scale.
# Measured directly, the two scales differ by roughly 1.4x, so feeding a whole-image-scale
# number to native conditioning systematically undershoots what the model was trained to
# respond to. The native fit's R^2 (~0.09) is low in absolute terms because an individual
# small crop is noisy around the pH trend the way any small random window is - but its
# per-bucket means climb monotonically across the real range (4.2 -> 7.6px from pH 5.8 to
# 8.8 over n>1000 crops), which is what the slope is actually being fit to.
#
# NOTE ON RANGE. The training distribution these feed reaches ~20px of waviness (see
# train.py's WARP_AUG_MAX_WAVINESS and the frame-height limits above TRAIN_SIZES), so
# native conditioning has real support up to about pH 16. predicted_waviness_native(20)
# asks for ~25px, which is past anything the model has seen; above roughly pH 16-17 the
# geometric warp (geometry_mode="warp") remains the mechanism with actual coverage.
#
# Re-run calibrate_ph.py after training a new checkpoint for the current fit of both;
# these are the same-methodology numbers to fall back on until then.
_DEFAULTS = {
    "waviness_slope": 1.093,        # rms_dev = slope*pH + intercept, whole-image scale
    "waviness_intercept": -3.108,
    "native_waviness_slope": 1.5258,   # same relationship, per-crop training scale
    "native_waviness_intercept": -5.4576,
    "lambda_gain": 1.0,           # 1.0 = the natural "range-widths past the anchor" scale
    "max_lambda": 3.0,
}


def _calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as fh:
            return {**_DEFAULTS, **json.load(fh)}
    return dict(_DEFAULTS)


# ---------------------------------------------------------------------------------------
# The pH -> waviness law. Its FUNCTIONAL FORM is chosen from the data rather than assumed,
# because the choice is unconstrained inside the trained range and dominates everything
# outside it: linear and exponential fits of the real crops are statistically
# indistinguishable over 5.8-8.8 (R^2 0.812 vs 0.798 on bucket means) yet differ by 1.67x at
# pH 12.8 and 6x at pH 20. Assuming one would be picking the answer rather than measuring it.
#
# Selection is by HELD-OUT EXTRAPOLATION error, not by in-sample fit: each form is refitted
# with an end bucket removed and scored on how well it predicts that bucket's mean. That
# tests the only thing we actually use the law for. Measured on the current dataset:
#
#     form           R^2 (all crops)   hold-out 8.8   hold-out 5.8   mean |err|
#     linear             0.087            0.865          0.880          0.872   <- winner
#     exponential       -0.041            0.348          1.922          1.135
#     quadratic          0.087            6.051          2.499          4.275
#
# Quadratic fits in-sample as well as linear and extrapolates disastrously, which is the
# whole reason the criterion is held-out rather than R^2.
WAVINESS_FORMS = ("linear", "exponential", "quadratic")

# The training distribution reaches WARP_AUG_MAX_WAVINESS (26px, see train.py), so a request
# past that asks for geometry the model has never been shown. Extrapolated laws are clamped
# here rather than allowed to run away - an exponential law would ask for 152px at pH 20.
MAX_SUPPORTED_WAVINESS = 26.0


def _fit_form(form, ph, w):
    """Least-squares coefficients for one candidate form. numpy is imported lazily so this
    module stays cheap to import for the inference path, which only ever evaluates."""
    import numpy as np
    ph = np.asarray(ph, dtype=float)
    w = np.asarray(w, dtype=float)
    if form == "linear":
        return np.polyfit(ph, w, 1).tolist()
    if form == "exponential":
        # fitted in log space, so it can never predict a negative waviness
        return np.polyfit(ph, np.log(np.clip(w, 1e-3, None)), 1).tolist()
    if form == "quadratic":
        return np.polyfit(ph, w, 2).tolist()
    raise ValueError(f"Unknown waviness law form: {form!r} (expected one of {WAVINESS_FORMS})")


def eval_law(law, pH):
    """Evaluate a stored {"form": ..., "coeffs": [...]} law at one pH."""
    form, c = law["form"], law["coeffs"]
    if form == "linear":
        return c[0] * pH + c[1]
    if form == "exponential":
        return math.exp(c[0] * pH + c[1])
    if form == "quadratic":
        return c[0] * pH * pH + c[1] * pH + c[2]
    raise ValueError(f"Unknown waviness law form: {form!r} (expected one of {WAVINESS_FORMS})")


def fit_waviness_law(ph, w, forms=WAVINESS_FORMS):
    """Pick the functional form the data supports, and return (law, diagnostics).

    ph/w are per-measurement (NOT per-bucket) so well-sampled pH levels carry more weight,
    matching how the old single-form fit worked. Scoring is the mean absolute error on the
    two end buckets when each is held out of the fit - see this section's header.
    """
    import numpy as np
    ph = np.asarray(ph, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = w > 0
    ph, w = ph[keep], w[keep]
    buckets = np.unique(ph)
    means = {float(b): float(w[ph == b].mean()) for b in buckets}

    diagnostics = {}
    for form in forms:
        coeffs = _fit_form(form, ph, w)
        law = {"form": form, "coeffs": coeffs}
        pred = np.array([eval_law(law, float(p)) for p in ph])
        r2 = float(1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum())
        errors = []
        for held in (float(buckets[-1]), float(buckets[0])):
            mask = ph != held
            if len(np.unique(ph[mask])) < 3:      # not enough levels left to refit
                continue
            held_law = {"form": form, "coeffs": _fit_form(form, ph[mask], w[mask])}
            errors.append(abs(eval_law(held_law, held) - means[held]))
        diagnostics[form] = {"coeffs": coeffs, "r2": r2,
                             "holdout_error": float(np.mean(errors)) if errors else float("inf")}

    best = min(diagnostics, key=lambda f: diagnostics[f]["holdout_error"])
    return {"form": best, "coeffs": diagnostics[best]["coeffs"]}, diagnostics


def _law(cal, key, legacy_slope, legacy_intercept):
    """The stored law, falling back to the pre-law slope/intercept pair for older
    ph_calibration.json files (and, through _DEFAULTS, to the hardcoded constants)."""
    law = cal.get(key)
    if isinstance(law, dict) and "form" in law:
        return law
    return {"form": "linear", "coeffs": [cal[legacy_slope], cal[legacy_intercept]]}


def normalize_pH(pH):
    """Map the trained pH range onto [-1, 1]. Out-of-range values are NOT meaningful here;
    they are handled by extrapolation, never by feeding them through the embedding."""
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1


def predicted_waviness(pH):
    """RMS centreline excursion in pixels that a real filament at this pH would have,
    from the fitted pH->waviness law over the measured dataset, continued past the ends.
    The law's functional form is chosen from the data - see WAVINESS_FORMS.

    Whole-image scale - use this for ph_warp.py's geometric warp, which measures its
    `current` waviness the same way. For the model's native waviness conditioning, use
    predicted_waviness_native() instead; the two are NOT interchangeable, see _DEFAULTS.

    Floored at 0.5px: the straight-line fit crosses zero near pH 2.8 and would go negative
    below that, but even a perfectly straight filament traces with some excursion.
    """
    cal = _calibration()
    law = _law(cal, "waviness_law", "waviness_slope", "waviness_intercept")
    return min(MAX_SUPPORTED_WAVINESS, max(0.5, eval_law(law, pH)))


def predicted_waviness_native(pH):
    """Same relationship as predicted_waviness(), fit on the per-crop training scale
    instead of the whole-image one - see _DEFAULTS for why the two must stay separate.
    This is the one to feed model.py's WavinessEmbedding (via velocity_for_pH's/
    edit_image's `waviness` argument); predicted_waviness() would systematically
    undershoot what the model was actually trained to respond to.
    """
    cal = _calibration()
    law = _law(cal, "native_waviness_law", "native_waviness_slope", "native_waviness_intercept")
    return min(MAX_SUPPORTED_WAVINESS, max(0.5, eval_law(law, pH)))


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


def anchor_velocities(model, x, t, source=None):
    """The velocity fields at the two ends of the trained range."""
    lo = torch.full((x.shape[0],), normalize_pH(PH_ANCHOR_LO), device=x.device)
    hi = torch.full((x.shape[0],), normalize_pH(PH_ANCHOR_HI), device=x.device)
    return model(x, t, lo, source=source), model(x, t, hi, source=source)


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


def velocity_for_pH(model, x, t, pH_query, rescale=True, lam_override=None, waviness=None,
                    source=None):
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

    `source` is the image being edited, for a source-conditioned checkpoint (model.py's
    in_channels == 2). A 1-channel model ignores it, so it is always safe to pass too.
    """
    if waviness is not None:
        anchor = min(max(pH_query, PH_MIN), PH_MAX)
        ph = torch.full((x.shape[0],), normalize_pH(anchor), device=x.device)
        wav = torch.full((x.shape[0],), float(waviness), device=x.device)
        return model(x, t, ph, wav, source=source)
    lam = ph_to_lambda(pH_query) if lam_override is None else lam_override
    if lam == 0.0:
        ph = torch.full((x.shape[0],), normalize_pH(pH_query), device=x.device)
        return model(x, t, ph, source=source)
    v_lo, v_hi = anchor_velocities(model, x, t, source=source)
    return extrapolate(v_lo, v_hi, lam, rescale=rescale)


def describe(pH_query, geometry_mode="warp"):
    """One-line summary for CLI output, so an extrapolated request is never silent.

    geometry_mode must match whatever the caller is actually about to run (img2img.py's and
    sample.py's own --geometry_mode) - this function has no way to know which mechanism was
    selected otherwise, and describing the wrong one is worse than describing neither. Which
    mechanism applies inside "warp"/"embedding" mode depends on the direction, because those
    were measured to work asymmetrically - see ph_warp.py for the numbers behind that split.
    "native" is symmetric across direction (it's the same conditioning pathway either way)
    and its response has been measured monotonic in the request; it has real training
    support to about pH 16 (see _DEFAULTS), above which "warp" is the mechanism with
    actual data behind it.
    """
    if PH_ANCHOR_LO <= pH_query <= PH_ANCHOR_HI:
        return f"pH {pH_query:g} is inside the trained range - direct conditioning"
    direction = "BELOW" if pH_query < PH_ANCHOR_LO else "ABOVE"
    if geometry_mode == "native":
        anchor = min(max(pH_query, PH_ANCHOR_LO), PH_ANCHOR_HI)
        return (f"pH {pH_query:g} is {direction} the trained range [{PH_ANCHOR_LO:g}, "
                f"{PH_ANCHOR_HI:g}] - NATIVE waviness conditioning: anchoring pH at "
                f"{anchor:g}, targeting {predicted_waviness_native(pH_query):.1f}px "
                f"directly through the model's own conditioning")
    if pH_query < PH_ANCHOR_LO:
        return (f"pH {pH_query:g} is BELOW the trained range [{PH_ANCHOR_LO:g}, "
                f"{PH_ANCHOR_HI:g}] - extrapolating the velocity field past the acidic "
                f"anchor with lambda={ph_to_lambda(pH_query):.2f} (straighter)")
    return (f"pH {pH_query:g} is ABOVE the trained range [{PH_ANCHOR_LO:g}, "
            f"{PH_ANCHOR_HI:g}] - conditioning at pH {PH_ANCHOR_HI:g}, then imposing "
            f"waviness {predicted_waviness(pH_query):.1f}px geometrically "
            f"(velocity extrapolation does not work in this direction)")
