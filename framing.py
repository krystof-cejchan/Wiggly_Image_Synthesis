"""Fit a crop to a target frame size without ever duplicating the fibre.

WHY THIS EXISTS
The pipeline used to reach a target height by mirror-padding (safe_mirror_pad), and for
this dataset that is actively destructive. The crops are tight bounding boxes around a
single fibre with a median height of 43px, while TRAIN_SIZES asked for 48-256px - so
reflection stacked between 1.5 and 6 mirrored copies of the fibre into one frame. Measured
consequences, on the real training set:

  - waviness.trace_fibre takes a per-column argmin, which then hops between the stacked
    copies instead of following one fibre. Median column-to-column jump was 45px on padded
    crops versus 15px on untiled ones, and corr(measured waviness, hop size) = +0.94.
    The "waviness label" was therefore ~94% a measurement of the tiling artefact.
  - fed that label, the model learned exactly what it was shown: high waviness means a
    tiled, banded frame. Asking a trained checkpoint for 30px of waviness produced vertical
    banding, not a wavy fibre. This is also the long-standing "hall of mirrors" generation
    artefact already noted in CLAUDE.md.

Padding with synthesised background instead keeps exactly one fibre in the frame, so the
label measures the fibre and the model learns the fibre.

HORIZONTAL padding is deliberately still mirroring (mirror_pad_width). Reflecting along the
fibre's own long axis maps y(W+d) -> y(W-d): the centreline stays continuous, nothing new
appears at a different height in the same column, and the argmin has nothing to hop to. It
is only the vertical direction that has to change.
"""
import torch
import torch.nn.functional as F


def _background_residual(img):
    """The crop's own high-frequency grain, restricted to rows that are background.

    The residual carries the fibre's signature too, and recycling a row that crosses the
    fibre paints a dark streak into what should be empty background - so rows near the
    fibre are excluded. These crops are tight bounding boxes, so what remains is the band
    above and below the fibre, on both sides (grain is homogeneous across the frame, so
    both edges are equally valid donors and using both doubles the pool).
    """
    blur = F.avg_pool2d(F.pad(img, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
    residual = img - blur

    smooth = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), 3, stride=1)[0, 0]
    fibre_row = float(smooth.min(dim=0).indices.float().median())
    rows = torch.arange(img.shape[2], device=img.device, dtype=torch.float32)
    donor = residual[:, :, (rows - fibre_row).abs() > 6, :]
    if donor.shape[2] < 4:                       # fibre fills the frame - no clean band
        donor = residual
    return donor


def synth_background(img, rows, at_top, generator=None):
    """Synthesise `rows` of background that blends with the edge it is attached to.

    Illumination and grain are handled separately. The illumination profile is copied per
    column from the adjacent real rows, so brightness continues across the seam and the
    crop's left-right falloff is preserved. The grain is REAL grain, recycled from the
    crop's own background rows (see _background_residual).

    Two details matter and were both found by looking at the output. Synthesising grain as
    scaled white noise reproduces the right standard deviation but the wrong spectrum, and
    reads as coarse blotching against the fine grain of the rows it adjoins. And recycling
    donor rows verbatim makes the reuse visible as horizontal streaking whenever the pad is
    deep relative to the source (a 28px crop grown to 64px needs more synthetic rows than
    it has real ones), so every recycled row also gets an independent horizontal roll and a
    random sign flip - the grain is zero-mean, so negating it is statistically free and
    breaks up the repetition that the eye picks out.
    """
    smooth = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    level = smooth[:, :, :2, :].mean(dim=2, keepdim=True) if at_top \
        else smooth[:, :, -2:, :].mean(dim=2, keepdim=True)

    donor = _background_residual(img)
    n_donor, w = donor.shape[2], donor.shape[3]
    pick = torch.randint(0, n_donor, (rows,), generator=generator)
    shift = torch.randint(0, w, (rows,), generator=generator)
    sign = torch.randint(0, 2, (rows,), generator=generator) * 2 - 1
    grain = torch.stack([donor[0, 0, int(pick[i])].roll(int(shift[i]), dims=0) * int(sign[i])
                         for i in range(rows)], dim=0)
    return level + grain.view(1, 1, rows, w).to(img.device)


def pad_height_background(img, target_h, generator=None, jitter=0.0):
    """Grow a (1,1,H,W) crop to target_h by adding synthesised background above and below.

    jitter (0..1) offsets the fibre off dead-centre by up to that fraction of the padding,
    so the model does not learn that a fibre is always exactly centred.
    """
    h = img.shape[2]
    if h >= target_h:
        return img
    need = target_h - h
    top = need // 2
    if jitter > 0 and need > 1:
        span = int(need * jitter * 0.5)
        if span > 0:
            top += int(torch.randint(-span, span + 1, (1,), generator=generator).item())
            top = max(0, min(need, top))
    bottom = need - top
    parts = []
    if top:
        parts.append(synth_background(img, top, at_top=True, generator=generator))
    parts.append(img)
    if bottom:
        parts.append(synth_background(img, bottom, at_top=False, generator=generator))
    return torch.cat(parts, dim=2)


def fit_height(img, target_h, generator=None, jitter=0.35):
    """Make a (1,1,H,W) crop exactly target_h tall: random-crop if taller, background-pad
    if shorter. Never reflects vertically - see this module's docstring."""
    h = img.shape[2]
    if h == target_h:
        return img
    if h > target_h:
        top = int(torch.randint(0, h - target_h + 1, (1,), generator=generator).item())
        return img[:, :, top:top + target_h, :]
    return pad_height_background(img, target_h, generator=generator, jitter=jitter)


def mirror_pad_width(img, target_w):
    """Reach target_w by reflecting along the fibre's long axis, then trim.

    Safe for the waviness label in a way vertical reflection is not: a reflection in x
    keeps the centreline y(x) single-valued and continuous, so the per-column argmin has no
    second fibre to jump to (it only makes the traced shape symmetric about the seam).
    """
    while img.shape[3] < target_w:
        img = torch.cat([img, img.flip(dims=[3])], dim=3)
    return img[:, :, :, :target_w]


def fit_frame(img, target_h, target_w, generator=None, jitter=0.35):
    """Full fit: width by random window (reflecting only when the source is too narrow),
    height by crop-or-background-pad.

    The random window has to be taken from the SOURCE, before any reflection. This used to
    call mirror_pad_width first and then crop - but mirror_pad_width always returns exactly
    target_w (it trims), so `w > target_w` was never true and the crop was dead code: every
    sample contributed only its leftmost target_w columns, on every epoch, forever. Verified
    directly before the fix - 20 different generator seeds returned the identical window. For
    a dataset whose widths reach 958px against a 384px frame that threw away most of the
    longest fibres, and it meant each source's waviness label was measured on one fixed
    region of it rather than sampling the whole filament.
    """
    w = img.shape[3]
    if w > target_w:
        left = int(torch.randint(0, w - target_w + 1, (1,), generator=generator).item())
        img = img[:, :, :, left:left + target_w]
    else:
        img = mirror_pad_width(img, target_w)
    return fit_height(img, target_h, generator=generator, jitter=jitter)
