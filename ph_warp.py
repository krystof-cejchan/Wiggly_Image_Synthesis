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
from waviness import box_blur, trace_fibre, waviness

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


def centreline(img, smooth_frac=0.02):
    """Absolute row position of the filament for every column, or None.

    trace_fibre removes the linear trend, which is right for measuring waviness and wrong
    here - the warp needs to know where the filament actually sits in the frame.
    """
    import numpy as np

    smoothed = box_blur(img, 3)[0, 0]
    background = smoothed.median(dim=0).values
    darkest, path = smoothed.min(dim=0)
    depth = background - darkest
    strong = torch.quantile(depth, 0.75)
    if strong <= 0:
        return None
    confident = torch.nonzero(depth > 0.5 * strong).flatten()
    if confident.numel() < 10:
        return None

    width = img.shape[3]
    xs = confident.detach().cpu().numpy()
    ys = path[confident].detach().cpu().float().numpy()
    full = np.interp(np.arange(width), xs, ys)
    kernel = max(5, int(width * smooth_frac) | 1)
    half = kernel // 2
    full = np.convolve(np.pad(full, half, mode="edge"),
                       np.ones(kernel) / kernel, mode="valid")[:width]
    return torch.from_numpy(full).float().to(img.device)


def _synth_background(img, rows, at_top, generator=None):
    """Synthesise `rows` of background that blends with the edge it is attached to.

    Built as illumination + grain, because those two need different treatment. The
    illumination profile is copied per column from the adjacent real row, so brightness
    continues smoothly across the seam and the crop's left-right falloff is preserved. The
    grain is fresh 2D-correlated noise scaled to the crop's ACTUAL high-frequency standard
    deviation, measured after subtracting a local blur.

    Everything simpler was tried and left visible artefacts. The crops are tight bounding
    boxes whose filament usually spans the full height, so there is no clean band to copy
    or tile - tiling two rows just prints a repeating pattern. Measuring the grain from the
    upper half of the histogram (an earlier attempt) underestimated it roughly twofold and
    the extension read as conspicuously smooth.
    """
    plane = img[0, 0]
    smooth = box_blur(img, 9)[0, 0]
    grain = plane - smooth

    line = centreline(img)
    if line is None:
        strength = float(grain.std())
    else:
        ys = torch.arange(plane.shape[0], device=plane.device).view(-1, 1)
        away = (ys - line.view(1, -1)).abs() > 6  # ignore the filament and its shoulders
        strength = float(grain[away].std()) if bool(away.any()) else float(grain.std())

    level = smooth[0] if at_top else smooth[-1]
    noise = torch.randn(1, 1, rows, plane.shape[1], device=plane.device, generator=generator)
    noise = F.avg_pool2d(F.pad(noise, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    noise = noise / noise.std().clamp(min=1e-6) * strength
    return (level.view(1, 1, 1, -1) + noise)


def _extend_frame(img, pad, generator=None):
    """Add `pad` synthesised background rows above and below; original rows untouched."""
    return torch.cat([_synth_background(img, pad, at_top=True, generator=generator),
                      img,
                      _synth_background(img, pad, at_top=False, generator=generator)], dim=2)


def warp_filament(img, d, margin=3, extend=True):
    """Displace the filament by d(x), with the displacement fading out away from it.

    Earlier attempts warped the whole frame, which drags the background along so the noise
    grain acquires the filament's undulation, and then tried splitting the image into
    background and filament layers - which failed differently: these crops are tight
    bounding boxes with ZERO filament-free rows, so the synthetic background fill had half
    the real contrast, and a greyscale closing puts every dark speckle into the "filament"
    layer, so warping it sheared the noise into vertical combs.

    Warping locally avoids both. A Gaussian envelope centred on the filament means it moves
    fully, background more than a few sigma away does not move at all, and the transition
    is a gentle stretch through pure noise where it cannot be seen. The frame is extended by
    reflection, so the new rows are real background grain rather than anything synthetic.

    Falls back to the plain global shear if the filament cannot be located.
    """
    line = centreline(img)
    if line is None:
        return img, 0.0

    # These crops are tight bounding boxes - a filament being made WAVIER often has no room
    # at all inside the original frame (fit scale 0), so the frame has to grow. A filament
    # being STRAIGHTENED moves inward and needs none, and extending it there is actively
    # harmful: the synthesised bands carry noise that the per-column trace can latch onto
    # instead of the now-flat filament, which reads back as more waviness, not less.
    if extend:
        pad = int(math.ceil(float(d.abs().max()))) + margin
        img = _extend_frame(img, pad)
        line = line + pad

    _, _, height, width = img.shape
    lo, hi = float(margin), float(height - 1 - margin)

    # Scale the displacement so the filament always lands inside the EXISTING frame. No
    # rows are invented at all, which is what finally removed the background artefacts:
    # synthetic fill had the wrong contrast on these tight crops (they contain zero
    # filament-free rows to sample from), and reflecting to make room duplicated the
    # filament itself whenever it sat near an edge, so the trace locked onto the mirror
    # copy. The cost is that a very tight crop cannot express the largest waviness; the
    # caller is told the scale so it can report the shortfall rather than hide it.
    scale = 1.0
    positive, negative = d > 0, d < 0
    if positive.any():
        scale = min(scale, float(((hi - line)[positive] / d[positive]).clamp(min=0).min()))
    if negative.any():
        scale = min(scale, float(((lo - line)[negative] / d[negative]).clamp(min=0).min()))
    scale = max(0.0, min(1.0, scale))
    d = d * scale

    peak = max(1.0, float(d.abs().max()))
    # wide enough that the stretch between moved and stationary rows never folds
    # (|dD/dy| < 1 needs sigma above about 0.6 * the peak displacement)
    sigma = max(8.0, 1.3 * peak)

    ys = torch.arange(height, device=img.device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(width, device=img.device, dtype=torch.float32).view(1, -1)
    envelope = torch.exp(-0.5 * ((ys - line.view(1, -1)) / sigma) ** 2)
    src_y = ys - d.view(1, -1) * envelope

    grid = torch.stack([2 * xs.expand(height, width) / max(width - 1, 1) - 1,
                        2 * src_y / max(height - 1, 1) - 1], dim=-1).unsqueeze(0)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection",
                           align_corners=True)
    return warped, scale


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
        candidate, scale = warp_filament(img, d)
        info["fit_scale"] = scale

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
    out, scale = warp_filament(img, d, extend=False)
    info["fit_scale"] = scale
    info.update({"applied_rms": float(d.std()), "warped": True, "mode_detail": "straighten"})
    return out, info


def refine_texture(model, img, anchor_pH, strength=0.4, num_steps=60, seed=None):
    """Re-render texture over warped geometry, without disturbing the geometry.

    The shear resamples pixels, which softens the filament and leaves the synthesised
    background looking flatter than real microscopy noise. Running a short img2img pass
    fixes that, and it is safe precisely because of the property that made the anchor
    useless for CHANGING waviness: at a modest strength the source pins the centreline, so
    the model repaints texture around geometry it cannot move. Source and target pH are
    both the anchor, so no pH editing happens here - only resynthesis.
    """
    from img2img import edit_image  # imported lazily; img2img itself now imports FROM this
                                    # module (apply_ph_waviness etc. for edit_to_pH), so this
                                    # import must stay deferred to avoid a circular top-level import
    out = edit_image(model=model, ref_image=img, source_pH=anchor_pH, target_pH=anchor_pH,
                     denoising_strength=strength, num_steps=num_steps,
                     contrastive_scale=1.0, seed=seed, contrast=1.0, solver="heun")
    return out * 2 - 1  # edit_image returns [0,1]; keep this module's [-1,1] convention
