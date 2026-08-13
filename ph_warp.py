"""Impose the waviness of an out-of-range pH geometrically.

WHY THIS EXISTS
The obvious approach - extrapolate the conditioning past the trained range - was tried
first and is implemented in ph_control.py. Measured on the v2 checkpoint it works in one
direction only:

    lambda   pH~      orientation spread of generated samples
     -2.0    -0.2         0.968      straighter  <- correct
     -1.0     2.8         1.213      straighter  <- correct
      0.0     8.8         1.939      (= plain pH 8.8)
     +1.0    11.8         1.699      STRAIGHTER  <- wrong direction
     +3.0    17.8         1.564      STRAIGHTER  <- wrong direction

Going acidic works. Going alkaline does not: the filaments stay wavy but become smoother
and more regular instead of buckling harder. The direction v(8.8) - v(5.8) encodes the
whole difference between the two ends - texture, fibre thickness, contrast - not just
geometry, and pushing along it amplifies all of that at once. The model has no way to
invent buckling it was never shown.

WHAT THIS DOES INSTEAD
Take the geometry from the measured physical law and the texture from the model. Both
properties of the centreline trend cleanly with pH across the dataset:

    pH     rms deviation (px)    dominant wavelength (px)
    5.8          3.92                    304
    6.8          4.14                    300
    7.4          5.04                    276
    7.8          6.74                    205
    8.8          6.29                    144

so a higher pH means larger AND tighter undulations. Extrapolating both and applying the
difference as a smooth vertical shear produces the geometry a more alkaline filament should
have, on top of texture the model actually knows how to draw. Amplitude extrapolates
linearly; wavelength is fitted in log space, because a linear fit reaches zero at pH 11.5
and then goes negative.

The honest caveat: this asserts that the trend measured over 5.8-8.8 continues. It is an
extrapolation of a fitted law, not evidence about real chemistry at pH 12, and the shear
adds contour length rather than conserving it as a buckling filament would.
"""
import math

import torch
import torch.nn.functional as F

from ph_control import predicted_waviness
from waviness import trace_fibre, waviness

# log-linear fit of dominant undulation wavelength against pH, over the measured buckets
_WL_LOG_SLOPE = -0.2477
_WL_LOG_INTERCEPT = 7.1420


def predicted_wavelength(pH):
    """Dominant undulation wavelength in pixels. Fitted in log space so it stays positive."""
    return float(math.exp(_WL_LOG_SLOPE * pH + _WL_LOG_INTERCEPT))


def synth_displacement(width, rms, wavelength, device, generator=None, num_modes=5):
    """A smooth random transverse displacement d(x) with the requested RMS.

    Several modes spread around the target wavelength rather than a single sinusoid: one
    pure sine reads as a manufactured wave, whereas a small band looks like the irregular
    buckling the real traces show.
    """
    x = torch.arange(width, device=device, dtype=torch.float32)
    d = torch.zeros(width, device=device)
    for _ in range(num_modes):
        scale = 0.7 + 0.7 * torch.rand(1, generator=generator, device=device).item()
        phase = 2 * math.pi * torch.rand(1, generator=generator, device=device).item()
        d = d + torch.sin(2 * math.pi * x / (wavelength * scale) + phase)
    d = d - d.mean()
    std = d.std().clamp(min=1e-6)
    return d * (rms / std)


def shear_rows(img, d, pad, generator=None):
    """Resample so that out(x, y) = in(x, y - d(x)), padding vertically first.

    A wavier filament genuinely needs more room than its original tight bounding box, so
    the output is taller. The new space is filled with noise matched to the image's own
    background: mirror padding would fold a copy of the filament into it, and replicating
    the edge rows smears whatever happens to sit on them into long vertical streaks -
    these crops are tight bounding boxes, so their edge rows often clip real structure.
    """
    values = img.flatten()
    background = values[values > values.median()]
    canvas = (torch.randn(img.shape[0], img.shape[1], img.shape[2] + 2 * pad, img.shape[3],
                          device=img.device, generator=generator)
              * background.std() + background.mean())
    canvas[:, :, pad:pad + img.shape[2], :] = img
    padded = canvas
    _, _, h, w = padded.shape
    ys = torch.arange(h, device=img.device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, -1)
    src_y = ys - d.view(1, -1)
    grid = torch.stack([2 * xs.expand(h, w) / max(w - 1, 1) - 1,
                        2 * src_y / max(h - 1, 1) - 1], dim=-1).unsqueeze(0)
    return F.grid_sample(padded, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def apply_ph_waviness(img, target_pH, measured=None, seed=None, wavelength=None):
    """Re-shape a filament image to the waviness a given pH calls for.

    img is (1, 1, H, W) in [-1, 1]. Returns (warped, info). Waviness adds roughly in
    quadrature, so the displacement needed on top of what the image already has is
    sqrt(target^2 - current^2); when the image is already wavier than the target this
    returns it untouched, since a shear can only add undulation, never remove it.
    """
    current = waviness(img) if measured is None else measured
    target = predicted_waviness(target_pH)
    info = {"current": current, "target": target, "applied_rms": 0.0,
            "wavelength": None, "warped": False}
    if current is None:
        return img, info
    if target < current:
        return _straighten(img, current, target, info)

    needed = math.sqrt(max(0.0, target ** 2 - current ** 2))
    lam = predicted_wavelength(target_pH) if wavelength is None else wavelength

    # Adding undulation in quadrature is only an approximation - the synthesised modes
    # partially cancel against the filament's existing shape, which undershot the target by
    # 10-15% in testing. So measure what was actually achieved and correct the amplitude,
    # always re-warping the ORIGINAL rather than the previous attempt: repeated resampling
    # would soften the filament a little more each round.
    best, best_err, applied = None, float("inf"), needed
    for attempt in range(3):
        gen = None
        if seed is not None:
            gen = torch.Generator(device=img.device).manual_seed(seed + attempt * 1000)
        d = synth_displacement(img.shape[3], applied, lam, img.device, generator=gen)
        pad = int(math.ceil(3 * applied))
        candidate = shear_rows(img, d, pad, generator=gen)

        achieved = waviness(candidate)
        if achieved is None:
            best = candidate if best is None else best
            break
        err = abs(achieved - target)
        if err < best_err:
            best, best_err, info["applied_rms"] = candidate, err, applied
            info["achieved"] = achieved
        if err <= 0.05 * target:
            break
        applied *= max(0.4, min(2.5, target / max(achieved, 1e-3)))

    info.update({"wavelength": lam, "warped": True})
    return best, info


def _straighten(img, current, target, info):
    """Damp an existing filament's undulation down to the target waviness.

    The mirror of the additive case, and it needs the actual centreline rather than a
    synthetic displacement: to remove a wiggle you have to know where it is. Scaling the
    traced deviation by k = target/current and shearing by the difference lands the
    centreline at k times its original excursion.

    This is what handles requests BELOW the trained range during editing. Velocity
    extrapolation does straighten during free generation, but not here - at strength 0.7
    the source anchor pins the centreline, and measured waviness stayed flat at ~6.5px
    across pH 3.0 to 7.3 no matter how hard the conditioning was pushed.
    """
    traced = trace_fibre(img)
    if traced is None:
        return img, info
    import numpy as np

    xs, ys = traced
    width = img.shape[3]
    # the trace only covers confident columns; hold the end values across the rest
    full = np.interp(np.arange(width), xs, ys)
    # Low-pass the displacement before shearing. A raw trace is jagged column to column,
    # and shearing neighbouring columns by very different amounts tears the texture into
    # vertical streaks. Real undulation is hundreds of pixels long, so smoothing well below
    # that removes the tearing without touching the shape being corrected.
    kernel = max(5, int(width * 0.02) | 1)
    pad_n = kernel // 2
    full = np.convolve(np.pad(full, pad_n, mode="edge"),
                       np.ones(kernel) / kernel, mode="valid")[:width]
    d = torch.from_numpy(full).float().to(img.device) * (target / max(current, 1e-3) - 1.0)
    out = shear_rows(img, d, pad=max(2, int(abs(d).max().item()) + 2))
    info.update({"applied_rms": float(d.std()), "warped": True, "mode_detail": "straighten"})
    return out, info


def edit_to_pH(model, ref_image, source_pH, target_pH, seed=None, **kw):
    """Edit a real crop to ANY pH, dispatching to whichever mechanism works there.

        target_pH < 5.8   velocity extrapolation past the acidic anchor (validated:
                          orientation spread falls 1.55 -> 1.21 -> 0.97 as it is pushed)
        5.8 <= pH <= 8.8  ordinary conditioning, exactly as before
        target_pH > 8.8   edit to the alkaline anchor, then impose the extra undulation
                          geometrically - velocity extrapolation fails in this direction

    Returns (image in [0,1], info dict). Everything in kw is forwarded to edit_image.
    """
    from img2img import edit_image
    from config import PH_MAX

    from config import PH_MIN

    anchor = min(max(target_pH, PH_MIN), PH_MAX)
    out = edit_image(model=model, ref_image=ref_image, source_pH=source_pH,
                     target_pH=anchor, seed=seed, **kw)
    if PH_MIN <= target_pH <= PH_MAX:
        return out, {"mode": "conditioned", "warped": False}

    # Outside the range in EITHER direction the geometry is imposed, because the model's
    # own response is unreliable there: it cannot buckle harder than pH 8.8, and under an
    # img2img anchor it cannot straighten either.
    warped, info = apply_ph_waviness(out * 2 - 1, target_pH, seed=seed)
    info["mode"] = "warp"
    return ((warped + 1) / 2).clamp(0, 1), info


def refine_texture(model, img, anchor_pH, strength=0.4, num_steps=60, seed=None):
    """Re-render texture over warped geometry, without disturbing the geometry.

    The shear resamples pixels, which softens the filament and leaves the synthesised
    background looking flatter than real microscopy noise. Running a short img2img pass
    fixes that, and it is safe precisely because of the property that made the anchor
    useless for CHANGING waviness: at a modest strength the source pins the centreline, so
    the model repaints texture around geometry it cannot move. Source and target pH are
    both the anchor, so no pH editing happens here - only resynthesis.
    """
    from img2img import edit_image  # imported lazily; img2img imports this module's peers
    out = edit_image(model=model, ref_image=img, source_pH=anchor_pH, target_pH=anchor_pH,
                     denoising_strength=strength, num_steps=num_steps,
                     contrastive_scale=1.0, seed=seed, contrast=1.0, solver="heun")
    return out * 2 - 1  # edit_image returns [0,1]; keep this module's [-1,1] convention
