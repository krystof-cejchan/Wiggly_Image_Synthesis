import os
import argparse
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF
from config import (PH_MIN, PH_MAX, DEVICE, CHECKPOINT_PATH,
                    TRAIN_MIN_H, TRAIN_MIN_W, TRAIN_MAX_W)
from model import from_state_dict
from ph_control import normalize_pH, velocity_for_pH, describe as describe_pH
from ph_warp import edit_to_pH
from framing import pad_height_background, mirror_pad_width
from waviness import (box_blur, waviness as measure_waviness,
                      wave_period as measure_wave_period, ripple_rms as measure_ripple_rms)

def load_and_preprocess_image(image_path):
    """loads a reference image, converts it to grayscale, and normalizes it to [-1, 1]"""
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)
    
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return img_tensor, original_size

def create_blending_mask(win_h, win_w, device, taper_h=True, taper_w=True):
    """2D blending mask for overlapping patches, tapered only along axes that actually slide.

    The window is taken from the INTERIOR of a Hann window of length n + 2. A plain
    hann_window(n) is exactly 0 at both endpoints, and the canvas border is covered by only
    one patch, so weight_global there is 0; dividing by the clamped weight then yields
    velocity 0, freezing the outermost row/column at its initial noise value. Trimming the
    endpoints keeps the window strictly positive while staying symmetric.

    An axis with only one window position needs no taper at all (its factor would cancel in
    the weighted average anyway) - passing ones keeps that exact rather than relying on the
    division to undo it.
    """
    def axis(n, taper):
        if not taper:
            return torch.ones(n, device=device)
        return torch.hann_window(n + 2, periodic=False, device=device)[1:-1]
    mask_2d = axis(win_h, taper_h).unsqueeze(1) * axis(win_w, taper_w).unsqueeze(0)
    return mask_2d.unsqueeze(0).unsqueeze(0)


# How far a crop may be grown vertically, as a multiple of its own height. Matches the cap
# dynamic_collate_fn applies during training (~2.2x the batch's 25th-percentile source
# height): past it, more of the frame is synthesised background than real crop, and
# framing.synth_background has too few donor rows to recycle without visible streaking.
HEIGHT_PAD_LIMIT = 2.2


def frame_height(h):
    """Canvas height to edit a crop of height `h` in, as a multiple of 16.

    The U-Net needs a multiple of 16 (4 downsampling stages), and the model was only ever
    shown frames TRAIN_MIN_H..TRAIN_MAX_H tall (config.TRAIN_SIZES), so a shorter crop is grown toward
    TRAIN_MIN_H with synthesised background - but never by more than HEIGHT_PAD_LIMIT.
    A crop taller than the trained band keeps its own height; nothing is ever cropped away.

    It is tempting to go further and size the frame to the waviness being ASKED for, since an
    RMS excursion of r sweeps roughly +-sqrt(2)*r and a frame shorter than that cannot show
    it. That was tried and is wrong: padding this dataset's 55px crop out to 96px so an 11px
    request would fit gave the model a mostly-empty frame and it filled the new room with a
    SECOND fibre (measured waviness 15.2px, but of a stacked pair, not one wavier filament),
    on top of the vertical streaking the deep background synthesis leaves. The native path is
    therefore bounded by the crop's own height, roughly (h/2 - fibre half-thickness)/sqrt(2),
    ~18px of RMS excursion for a 64px frame. Growing the frame to get past that is
    ph_warp.py's warp path (`extend_frame`), which can do it safely because it moves pixels
    instead of asking the model to generate into the new rows.
    """
    natural = int(math.ceil(h / 16) * 16)
    want = min(max(h, TRAIN_MIN_H), HEIGHT_PAD_LIMIT * h)
    return max(natural, int(want // 16) * 16)


def plan_window_starts(total_w, window_w, stride):
    """Left edges of the sliding windows, spread evenly across the canvas.

    `stride` decides only HOW MANY windows there are; where they sit follows from the canvas,
    with the last one flush against the right edge. The old version stepped by a fixed stride
    and then mirror-padded the canvas until the arithmetic came out even - for a 478px crop
    that was ~100px (20%) of reflected fibre, which every window near the right edge then
    spent part of its receptive field on.
    """
    if total_w <= window_w:
        return [0]
    count = math.ceil((total_w - window_w) / stride) + 1
    span = total_w - window_w
    return [round(i * span / (count - 1)) for i in range(count)]


def apply_contrast(img01, contrast, mode="linear"):
    """Contrast boost for an image already mapped to [0, 1].

    mode="linear" scales around the image mean, so the overall brightness is preserved.
    mode="gamma" is the original img**contrast. That gamma only ever darkens (x**2 < x on
    [0,1]) - it dropped the mean from 0.64 to 0.43 in testing - and it stretches
    differences in the dark half of the range, which turns the illumination falloff that
    is already present in the source crops into a heavy black end. The gradient itself is
    real data and is left untouched either way; "linear" just stops amplifying it.
    """
    if contrast == 1.0:
        return img01
    if mode == "gamma":
        return torch.pow(img01, contrast)
    if mode != "linear":
        raise ValueError(f"Unknown contrast mode: {mode!r} (expected 'linear' or 'gamma')")
    mean = img01.mean()
    return ((img01 - mean) * contrast + mean).clamp(0, 1)

@torch.no_grad()
def repair_fibre_gaps(ref_image, max_gap=30, min_side=6, max_slope=1.5, blur=3):
    """Bridge short bright breaks in the fibre, returning (repaired_image, list_of_gaps).

    Why this exists: edit_image starts the ODE from (1-strength)*source + strength*noise,
    so the source stays pinned as an anchor for the whole trajectory. A bright break in the
    fibre is pinned along with everything else - the guidance wants a dark fibre through
    that spot while the anchor insists it is bright, and the result fades or breaks exactly
    there. Loosening the anchor is NOT a fix: at strength 0.85 ghost fibres appear and by
    0.95 the output degenerates into several stacked copies, because the 0.3 anchor is the
    only thing suppressing the mirror-padding tiling. So the repair edits the anchor
    instead, and leaves strength alone.

    Only short interior breaks bracketed by confident fibre on both sides are touched, and
    the painted line is clamped with a minimum against the original, so background texture
    survives and only the fibre core darkens.
    """
    device = ref_image.device
    smooth = box_blur(ref_image, blur)[0, 0]
    height, width = smooth.shape

    # per-column background: the fibre is thin relative to the strip, so the column median
    # sits in the background rather than on the fibre
    background = smooth.median(dim=0).values
    darkest, path_raw = smooth.min(dim=0)
    depth = background - darkest  # >0 wherever a dark fibre is present

    strong = torch.quantile(depth, 0.75)
    if strong <= 0:
        return ref_image, []
    confident = depth > 0.5 * strong
    if int(confident.sum()) < 2 * min_side:
        return ref_image, []

    # fibre thickness, from the half-maximum width of the confident columns
    widths = []
    for x in torch.nonzero(confident).flatten().tolist():
        widths.append(int((smooth[:, x] < background[x] - 0.5 * depth[x]).sum()))
    fwhm = torch.tensor(widths, dtype=torch.float32).median().item() if widths else 3.0
    sigma = float(min(6.0, max(0.8, fwhm / 2.355)))

    repaired = ref_image.clone()
    rows = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)
    flags = confident.tolist()
    gaps = []

    x = 0
    while x < width:
        if flags[x]:
            x += 1
            continue
        start = x
        while x < width and not flags[x]:
            x += 1
        end = x - 1
        left, right = start - 1, end + 1
        # interior breaks only, and only short ones - a long stretch is a fibre that really
        # does end there, not a defect to paint over
        if left < 0 or right >= width or (end - start + 1) > max_gap:
            continue
        # a large vertical jump between the two sides usually means the columns belong to
        # two different fibres; bridging them would invent a connection that is not there
        if abs(int(path_raw[right]) - int(path_raw[left])) / max(1, right - left) > max_slope:
            continue

        lo = max(0, left - min_side + 1)
        hi = min(width, right + min_side)
        amp_left = depth[lo:left + 1][confident[lo:left + 1]]
        amp_right = depth[right:hi][confident[right:hi]]
        if amp_left.numel() == 0 or amp_right.numel() == 0:
            continue
        amplitude = 0.5 * (amp_left.mean() + amp_right.mean())

        span = right - left
        for xi in range(start, end + 1):
            frac = (xi - left) / span
            y_centre = (1 - frac) * float(path_raw[left]) + frac * float(path_raw[right])
            base = (1 - frac) * background[left] + frac * background[right]
            # Blend toward the fibre core weighted by the Gaussian itself: full strength on
            # the centre line, untouched original a couple of sigma away. Compositing with a
            # minimum() instead would clamp every background pixel brighter than `base` down
            # to it, replacing the noise texture across the whole column with a flat band.
            weight = torch.exp(-0.5 * ((rows.squeeze(1) - y_centre) / sigma) ** 2)
            repaired[0, 0, :, xi] = (1 - weight) * repaired[0, 0, :, xi] + weight * (base - amplitude)

        gaps.append({"start": start, "end": end, "width": end - start + 1,
                     "y_left": int(path_raw[left]), "y_right": int(path_raw[right]),
                     "amplitude": float(amplitude)})

    return repaired, gaps


def save_repair_diagnostic(original, repaired, gaps, out_path):
    """Write a before/after of the anchor repair so the detection can be eyeballed."""
    orig = ((original[0, 0].detach().cpu() + 1) / 2)
    rep = ((repaired[0, 0].detach().cpu() + 1) / 2)
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.5))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("source (anchor before repair)", fontsize=9)
    axes[1].imshow(rep, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"anchor after repair - {len(gaps)} gap(s) bridged", fontsize=9)
    axes[2].imshow((rep - orig).abs(), cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("what the repair changed", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def measure_frame_waviness(ref_image):
    """The reference's own waviness, measured in the frames the model will actually see.

    Returns None if no window traces confidently. This is what the native path should feed
    the CONTRASTIVE pair's source branch: that branch is meant to say "here is what the
    filament is now", and it can be measured rather than guessed from ph_calibration.json's
    pH->waviness fit (R^2 ~ 0.09). On the reference this was developed against, the fit says
    3.46px where the crop actually measures 1.77px - so the fitted value understates the gap
    the guidance has to close by about a quarter.

    Measured per window and averaged, not on the whole image, because the model is
    conditioned per frame: a 384px window captures less of a long-wavelength undulation than
    the full 478px canvas does, so whole-image waviness is a different (larger) number than
    the one the conditioning is calibrated in.
    """
    _, _, h, w = ref_image.shape
    frame_h = frame_height(h)
    framed = pad_height_background(ref_image, frame_h, jitter=0.0) if frame_h > h else ref_image
    target_w = max(TRAIN_MIN_W, ((w + 15) // 16) * 16)
    framed = mirror_pad_width(framed, target_w)
    win_w = min(target_w, TRAIN_MAX_W)
    values = [measure_waviness(framed[:, :, :, x:x + win_w])
              for x in plan_window_starts(target_w, win_w, win_w // 2)]
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


@torch.no_grad()
def measure_frame_period(ref_image):
    """The reference's own undulation period, measured in the frames the model will see.

    Companion to measure_frame_waviness - the contrastive pair's source branch should state
    both halves of "what the filament is now", and both are measurable. Returns None when no
    window yields a period (waviness.wave_period declines to guess one for a filament too
    straight for it to mean anything), which the caller turns into the fitted value.
    """
    _, _, h, w = ref_image.shape
    frame_h = frame_height(h)
    framed = pad_height_background(ref_image, frame_h, jitter=0.0) if frame_h > h else ref_image
    target_w = max(TRAIN_MIN_W, ((w + 15) // 16) * 16)
    framed = mirror_pad_width(framed, target_w)
    win_w = min(target_w, TRAIN_MAX_W)
    values = [measure_wave_period(framed[:, :, :, x:x + win_w])
              for x in plan_window_starts(target_w, win_w, win_w // 2)]
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


@torch.no_grad()
def measure_frame_ripple(ref_image):
    """The reference's own FINE-undulation rms, in the frames the model will see.

    The ripple companion to measure_frame_waviness, and the source branch of the contrastive
    pair should state it for the same reason it states the waviness: that branch describes
    what the filament IS, and both halves of that are measurable rather than guessed. Returns
    None when no window traces.
    """
    _, _, h, w = ref_image.shape
    frame_h = frame_height(h)
    framed = pad_height_background(ref_image, frame_h, jitter=0.0) if frame_h > h else ref_image
    target_w = max(TRAIN_MIN_W, ((w + 15) // 16) * 16)
    framed = mirror_pad_width(framed, target_w)
    win_w = min(target_w, TRAIN_MAX_W)
    values = [measure_ripple_rms(framed[:, :, :, x:x + win_w])
              for x in plan_window_starts(target_w, win_w, win_w // 2)]
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


@torch.no_grad()
def edit_image(model, ref_image, source_pH, target_pH, denoising_strength=0.5,
               num_steps=100, contrastive_scale=3.0, seed=None, window_size=None, stride=None,
               contrast=1.2, solver="heun", contrast_mode="linear", ph_lambda=None,
               ph_rescale=False, source_waviness=None, target_waviness=None,
               source_period=None, target_period=None,
               source_ripple=None, target_ripple=None, geometry=None):
    """
    Edits the reference image to change its pH from source_pH to target_pH using a sliding window approach.

    window_size/stride default to the frame geometry the model was trained on
    (config.TRAIN_SIZES): one window over the whole crop when it fits inside the trained
    width band, a TRAIN_MAX_W-wide window slid across it when it does not. Passing them
    explicitly overrides that and is only useful for reproducing an older run - the previous
    fixed 128px-wide window sat outside that band and is exactly what stopped the waviness
    conditioning working (see config.TRAIN_SIZES for the measurement).

    solver: "euler" (1st order) or "heun" (2nd order predictor-corrector). Heun evaluates
    the velocity field twice per step (2x model calls) but has O(dt^2) local error instead
    of O(dt), which reduces integration drift over the num_steps trajectory.

    Either pH may lie outside the trained range: ph_control.velocity_for_pH extrapolates
    along the acidic->alkaline direction rather than pushing an out-of-range value through
    the periodic embedding, which would alias. ph_lambda forces a specific extrapolation
    strength instead of deriving one from target_pH, and exists for calibrate_ph.py.

    source_waviness/target_waviness, when given, are forwarded to velocity_for_pH and take
    over from the embedding-space extrapolation above: geometry is driven directly through
    the model's own waviness conditioning instead (needs a checkpoint trained with it - see
    model.py's WavinessEmbedding). source_ripple/target_ripple are the second half of that
    geometry request - HOW MUCH of the excursion is FINE undulation (waviness.ripple_rms,
    24-96px) rather than one long arc, where waviness says only how far the centreline
    strays in total. Asking for the total alone is degenerate: one frame-wide arc and ten
    small waves of the same amplitude score identical rms, the arc is cheaper to draw, and
    that is exactly what the model produced before this channel existed (8.60px of
    centreline rms in the 96-192px band against real crops' 2.75, and 0.73px in 24-48px
    against real crops' 2.83). source_period/target_period drive the LEGACY period channel
    on a _pair-generation checkpoint and are ignored by a ripple-conditioned one.
    Both are needed: rms alone is degenerate between one long arc and many short waves, and a
    model told only the rms produces the arc. ph_warp.edit_to_pH's geometry_mode="native" is what sets
    these; a plain call here still defaults to today's behaviour unchanged.

    `geometry` supersedes all of the scalars above on an in_channels == 3 checkpoint: it is
    the target centreline rendered as a full-canvas ridge (waviness.centreline_map), i.e. the
    actual curve rather than statistics about it, and it is windowed alongside the source. It
    goes to BOTH contrastive branches deliberately - the pair should differ in pH texture
    only, so the shape stays pinned by the channel instead of being pushed by
    contrastive_scale, which would reintroduce the very hedging this channel removes. Must be
    (1, 1, H, W) matching ref_image before any framing; it is framed here the same way.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, _, h, w = ref_image.shape

    # Frame the canvas inside the geometry the model was actually trained on (frame_height).
    # Height grows with SYNTHESISED BACKGROUND rather than reflection: reflecting to reach a
    # target height mirror-tiles the fibre - 3.5x over for a 55px crop, back when the height
    # was padded to fit the window rather than the other way round - and then asks the model
    # to edit a hall of mirrors. See framing.py.
    frame_h = frame_height(h)
    top_offset = (frame_h - h) // 2
    padded_ref = pad_height_background(ref_image, frame_h, jitter=0.0) if frame_h > h else ref_image
    # Width: reflection here is safe (it keeps the centreline single-valued - framing.py) and
    # is now used only to reach the next multiple of 16, or the trained minimum width for a
    # crop narrower than anything the model has seen.
    target_w = max(TRAIN_MIN_W, ((w + 15) // 16) * 16)
    padded_ref = mirror_pad_width(padded_ref, target_w)
    target_h = padded_ref.shape[2]

    # One window over the whole crop whenever it fits the trained width band; a wider crop
    # slides a trained-width window across it with 50% overlap.
    win_w = min(target_w, TRAIN_MAX_W if window_size is None else window_size)
    starts = plan_window_starts(target_w, win_w, win_w // 2 if stride is None else stride)

    # A SOURCE-CONDITIONED checkpoint (model.py, in_channels == 2) is trained on the editing
    # task itself: the reference goes in through its own input channel, clean, at every step.
    # So the trajectory starts from PURE NOISE and runs the whole way - there is no anchor to
    # set a level for and `denoising_strength` does not apply. That is the point of it: mixing
    # the source into the noise (SDEdit, below) made one knob govern both how much texture is
    # redrawn and how far the geometry may move, and no setting satisfied both - at strength
    # 0.8 the requested waviness was reached but the fibre was lost in 16% of columns, and at
    # 0.7 the fibre was perfect at 6px of a 10.7px request. Starting from noise also means the
    # old straight filament is never present to be half-erased, which is what produced the
    # gaps and ghosting.
    # The geometry canvas has to go through the SAME two framing operations the image does,
    # or the curve stops lining up with the pixels it describes. Height grows with zeros
    # rather than synthesised background: a blank row means "no fibre requested here", which
    # is exactly true of rows the request never covered.
    geometry_canvas = None
    if geometry is not None and getattr(model, "geometry_conditioned", False):
        geometry_canvas = geometry
        if frame_h > geometry_canvas.shape[2]:
            padded = torch.zeros(1, 1, frame_h, geometry_canvas.shape[3],
                                 device=geometry_canvas.device)
            padded[:, :, top_offset:top_offset + geometry_canvas.shape[2], :] = geometry_canvas
            geometry_canvas = padded
        geometry_canvas = mirror_pad_width(geometry_canvas, target_w)

    source_canvas = padded_ref if getattr(model, "source_conditioned", False) else None
    if source_canvas is not None:
        t_start = 0.0
        x = torch.randn_like(padded_ref)
    else:
        t_start = 1.0 - denoising_strength
        noise = torch.randn_like(padded_ref)
        x = (1 - t_start) * noise + t_start * padded_ref

    start_step = int(t_start * num_steps)
    # one full-height window, sliding horizontally only: no vertical taper needed
    mask = create_blending_mask(target_h, win_w, ref_image.device, taper_h=False,
                                taper_w=len(starts) > 1)

    def compute_v_dir(x_in, step_idx):
        """Velocity field at (x_in, t=step_idx/num_steps), blended across sliding windows."""
        t = torch.full((1,), step_idx / num_steps, device=DEVICE)

        progress = (step_idx - start_step) / max(1, num_steps - start_step)
        current_scale = contrastive_scale * (1.0 - progress) + 1.0 * progress

        v_source_global = torch.zeros_like(x_in)
        v_target_global = torch.zeros_like(x_in)
        weight_global = torch.zeros_like(x_in)

        for x_idx in starts:
            x_patch = x_in[:, :, :, x_idx:x_idx+win_w]
            src_patch = (None if source_canvas is None
                         else source_canvas[:, :, :, x_idx:x_idx+win_w])
            # The same curve goes to both branches - see the docstring.
            geo_patch = (None if geometry_canvas is None
                         else geometry_canvas[:, :, :, x_idx:x_idx+win_w])

            v_src_patch = velocity_for_pH(model, x_patch, t, source_pH,
                                          rescale=ph_rescale, waviness=source_waviness,
                                          source=src_patch, period=source_period,
                                          ripple=source_ripple, geometry=geo_patch)
            v_tgt_patch = velocity_for_pH(model, x_patch, t, target_pH,
                                          rescale=ph_rescale, lam_override=ph_lambda,
                                          waviness=target_waviness, source=src_patch,
                                          period=target_period, ripple=target_ripple,
                                          geometry=geo_patch)

            v_source_global[:, :, :, x_idx:x_idx+win_w] += v_src_patch * mask
            v_target_global[:, :, :, x_idx:x_idx+win_w] += v_tgt_patch * mask
            weight_global[:, :, :, x_idx:x_idx+win_w] += mask

        v_source = v_source_global / weight_global.clamp(min=1e-8)
        v_target = v_target_global / weight_global.clamp(min=1e-8)

        return v_source + current_scale * (v_target - v_source)

    dt = 1.0 / num_steps
    for i in range(start_step, num_steps):
        v1 = compute_v_dir(x, i)

        if solver == "euler":
            x = x + v1 * dt
        elif solver == "heun":
            x_pred = x + v1 * dt
            v2 = compute_v_dir(x_pred, i + 1)
            x = x + 0.5 * (v1 + v2) * dt
        else:
            raise ValueError(f"Unknown solver: {solver!r} (expected 'euler' or 'heun')")

    x_cropped = x[:, :, top_offset:top_offset + h, :w]

    out = (x_cropped.clamp(-1, 1) + 1) / 2

    return apply_contrast(out, contrast, contrast_mode)

def visualize_difference(original_tensor, edited_tensor, original_size, source_pH, target_pH):
    """Visualizes the difference between the original and edited images.

    An above-range target pH can grow the edited canvas taller than the source (ph_warp
    extends the frame to fit waviness a thin crop has no room for) - the diff map only
    makes sense over pixels the two share, so it's taken from a centered crop of the taller
    image rather than assuming matching shapes.
    """
    orig_w, orig_h = original_size
    orig_crop = original_tensor[:, :, :orig_h, :orig_w]
    edit_crop = edited_tensor[:, :, :, :orig_w]

    orig_img = (orig_crop.squeeze().cpu() + 1) / 2
    edit_img = edit_crop.squeeze().cpu()

    if edit_img.shape[0] == orig_img.shape[0]:
        diff_map = torch.abs(orig_img - edit_img)
    else:
        top = (edit_img.shape[0] - orig_img.shape[0]) // 2
        diff_map = torch.abs(orig_img - edit_img[top:top + orig_img.shape[0]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # original
    axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original Image (pH: {source_pH:.2f})")
    axes[0].axis('off')
    
    # result
    axes[1].imshow(edit_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Edited Image (pH: {target_pH:.2f})")
    axes[1].axis('off')

    im_diff = axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title("Difference Map (Absolute Change)")
    axes[2].axis('off')
    
    fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Image-to-Image pH Editing using Conditional UNet")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference image (e.g., 'data/ref_image.png')")
    parser.add_argument("--source_pH", type=float, required=True, help=f"Initial pH of the reference image (trained range {PH_MIN}-{PH_MAX})")
    parser.add_argument("--target_pH", type=float, required=True,
                        help=f"Target pH. Values outside the trained range {PH_MIN}-{PH_MAX} "
                             "are reached by extrapolating along the acidic->alkaline "
                             "direction; see ph_control.py")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to the model checkpoint")
    parser.add_argument("--strength", type=float, default=0.65, help="Editing strength [0.0 - 1.0] (corresponds to noise level)")
    parser.add_argument("--contrastive_scale", type=float, default=3.0, help="Contrastive scale for editing")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for editing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--contrast", type=float, default=2.0, help="Contrast strength (optional)")
    parser.add_argument("--contrast_mode", type=str, default="linear", choices=["linear", "gamma"],
                        help="'linear' scales around the mean (preserves brightness); "
                             "'gamma' is the original img**contrast, kept to reproduce older runs")
    parser.add_argument("--repair_gaps", action="store_true",
                        help="Bridge short bright breaks in the fibre before editing. Off by "
                             "default - it modifies the source anchor. Writes a before/after "
                             "diagnostic next to the result.")
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"], help="ODE solver for the editing trajectory")
    parser.add_argument("--waviness_mode", type=str, default="relative",
                        choices=["relative", "absolute"],
                        help="Whose waviness the target pH sets. 'relative' (default) scales "
                             "THIS filament's own excursion by the ratio the fitted law "
                             "predicts between source and target pH - so an X->X edit is the "
                             "identity and a flat crop stays relatively flat. 'absolute' "
                             "targets the population average at the requested pH, which "
                             "discards the individual filament: real crops at one pH span "
                             "0.69-9.33px around a 4.12px mean, so roughly half of them get "
                             "BENT even when the request lowers the pH.")
    parser.add_argument("--geometry_mode", type=str, default="auto",
                        choices=["auto", "warp", "native"],
                        help="'auto' (default) follows the checkpoint: a geometry-conditioned "
                             "one (in_channels==3) gets 'native', anything older gets 'warp'. "
                             "'native' hands the model the target CURVE as a third input "
                             "channel and lets it draw every pixel - no pixel warping at all, "
                             "and pH stays inside its trained range because the shape arrives "
                             "as geometry rather than as an out-of-range pH. "
                             "'warp' holds the fibre still while the model "
                             "re-renders pH texture, then moves REAL PIXELS into the "
                             "requested shape - a broadband displacement carrying the "
                             "requested waviness and ripple. Use it: the fibre keeps the "
                             "source's contrast and continuity because it is the source's "
                             "own fibre, but the shape is imposed outside the network. On a "
                             "pre-geometry checkpoint 'native' falls back to driving shape "
                             "through the waviness/ripple SCALARS, which state how much wave "
                             "but never where - keep that for comparison only.")

    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist.")
        return
        
    if not os.path.exists(args.ref_image):
        print(f"Reference image {args.ref_image} not found. Please check the provided path.")
        return

    model = from_state_dict(torch.load(args.checkpoint, map_location=DEVICE), DEVICE)
    model.eval()
    
    os.makedirs("outputs_img2img", exist_ok=True)
    
    ref_image, original_size = load_and_preprocess_image(args.ref_image)
    print(f"Loaded image with original resolution: {original_size[0]}x{original_size[1]}")
    if getattr(model, "source_conditioned", False):
        print(f"Source-conditioned checkpoint: the reference goes in through its own input "
              f"channel and the trajectory starts from pure noise, so --strength "
              f"({args.strength:g}) is ignored.")
    # Resolve "auto" the same way ph_warp.edit_to_pH will, so the description matches the
    # mechanism that is actually about to run - describing the wrong one is worse than
    # describing neither (see ph_control.describe).
    resolved_mode = args.geometry_mode
    if resolved_mode == "auto":
        resolved_mode = ("native" if getattr(model, "geometry_conditioned", False)
                         else "warp")
    print(describe_pH(args.target_pH, geometry_mode=resolved_mode,
                      geometry_channel=getattr(model, "geometry_conditioned", False)))

    display_ref = ref_image
    if args.repair_gaps:
        repaired, gaps = repair_fibre_gaps(ref_image)
        if gaps:
            spans = ", ".join(f"x={g['start']}-{g['end']} ({g['width']}px)" for g in gaps)
            print(f"Repaired {len(gaps)} fibre gap(s): {spans}")
            diag_path = "outputs_img2img/repair_diagnostic.png"
            save_repair_diagnostic(ref_image, repaired, gaps, diag_path)
            print(f"Repair diagnostic saved to: {diag_path}")
            ref_image = repaired
        else:
            print("No repairable fibre gaps detected - anchor left unchanged.")

    edited_img, edit_info = edit_to_pH(
        model=model,
        ref_image=ref_image,
        source_pH=args.source_pH,
        target_pH=args.target_pH,
        denoising_strength=args.strength,
        num_steps=args.num_steps,
        contrastive_scale=args.contrastive_scale,
        seed=args.seed,
        contrast=args.contrast,
        solver=args.solver,
        contrast_mode=args.contrast_mode,
        extend_frame=True,
        geometry_mode=args.geometry_mode,
        waviness_mode=args.waviness_mode,
    )
    if edit_info.get("mode") == "warp":
        if edit_info.get("warped"):
            grown = edited_img.shape[2] - ref_image.shape[2]
            achieved = edit_info.get("achieved")
            detail = f"achieved rms {achieved:.1f}px" if achieved is not None else \
                     f"applied rms {edit_info.get('applied_rms', 0.0):.1f}px"
            extra = f", canvas grew {grown}px to fit it" if grown > 0 else ""
            print(f"Geometric pH warp applied: target waviness {edit_info.get('target', 0.0):.1f}px "
                  f"({detail}{extra})")
            # The ripple half of the request. Reported next to the total because the total
            # alone cannot distinguish one frame-wide arc from many small waves, and telling
            # them apart is the whole reason the warp is broadband.
            tgt_r, got_r = edit_info.get("target_ripple"), edit_info.get("achieved_ripple")
            if tgt_r is not None:
                got = f" -> {got_r:.1f}px" if got_r is not None else ""
                print(f"  Fine undulation (24-96px band): {edit_info.get('current_ripple') or 0.0:.1f}px "
                      f"on the source, {tgt_r:.1f}px requested{got}")
            if edit_info.get("fit_scale", 1.0) < 0.999:
                print(f"  Note: source crop limited the warp to "
                      f"{edit_info['fit_scale'] * 100:.0f}% of the requested displacement.")
            target_rms = edit_info.get("target") or 0.0
            if achieved is not None and target_rms and achieved < 0.7 * target_rms:
                # Not a bug and not a bad seed: a displacement steep enough to reach a very
                # large request tears the texture under grid_sample, so ph_warp backs the
                # amplitude off against WARP_MAX_SLOPE. Far above pH 8.8 the fitted law asks
                # for more excursion than a crop this tall can carry cleanly, and what comes
                # out is the steepest clean warp rather than the requested one.
                print(f"  Note: only {achieved / target_rms * 100:.0f}% of the requested "
                      f"waviness was reachable without over-steepening the warp - the law "
                      f"asks for more than a {ref_image.shape[2]}px-tall crop can carry.")
        else:
            print("No geometric pH warp applied: the fibre could not be confidently traced "
                  "in this crop, so the result is left at the pH 5.8/8.8 anchor edit.")
    elif edit_info.get("geometry") == "curve":
        # The geometry-channel path: the model drew every pixel, following the curve it was
        # handed. Three numbers matter and they are different things - what the fitted law
        # asked for, what curve was actually synthesised for it, and what the model then drew.
        # Reporting the middle one separately is what distinguishes "the plan fell short" from
        # "the model did not follow the plan".
        rendered = edit_info.get("rendered")
        grown = edit_info.get("canvas_grew", 0)
        extra = f", canvas grew {grown}px to fit it" if grown else ""
        basis = ("scaled from this crop" if args.waviness_mode == "relative"
                 else "population average")
        print(f"Geometry channel: pH {args.target_pH:g} calls for "
              f"{edit_info.get('target', 0.0):.1f}px of centreline rms ({basis}); "
              f"curve synthesised at "
              f"{edit_info.get('achieved') or 0.0:.1f}px"
              + (f", model drew {rendered:.1f}px" if rendered is not None else "")
              + extra)
        r_t, r_g = edit_info.get("target_ripple"), edit_info.get("rendered_ripple")
        if r_t is not None:
            print(f"  Fine undulation (24-96px band): {r_t:.1f}px requested"
                  + (f" -> {r_g:.1f}px drawn" if r_g is not None else ""))
        planned = edit_info.get("achieved")
        # The 2px floor keeps this quiet on straightening requests: at a planned 0.5px the
        # tracer's own noise is larger than the shortfall being complained about (a
        # dead-straight fibre already measures ~0.7px in the sub-24px band), so a relative
        # threshold alone cries wolf on every request below the trained range.
        if rendered is not None and planned and planned > 2.0 and rendered < 0.7 * planned:
            print("  Note: the model fell well short of the curve it was given - that is a "
                  "model failure, not a planning one. Re-run with a different --seed, and "
                  "check the geometry conditioning gap in outputs/training_loss.csv.")
    elif edit_info.get("mode") == "native":
        achieved = edit_info.get("achieved")
        got = f", achieved {achieved:.1f}px" if achieved is not None else ""
        print(f"Native waviness conditioning applied (SCALAR path - pre-geometry checkpoint): "
              f"{edit_info.get('source_waviness', 0.0):.1f}px "
              f"measured on the reference -> {edit_info.get('target_waviness', 0.0):.1f}px requested"
              f"{got}")
        if achieved is not None and achieved < 0.6 * edit_info.get("target_waviness", 0.0):
            print("  Note: well short of the request. The anchored edit varies a lot with the "
                  "noise draw - re-run with a different --seed before concluding anything.")

    # compare against the unrepaired source - the repair is an input-side aid, not a result
    visualize_difference(display_ref, edited_img, original_size, source_pH=args.source_pH, target_pH=args.target_pH)

    orig_w, orig_h = original_size
    edited_crop_for_save = edited_img[:, :, :, :orig_w]

    save_path = f"outputs_img2img/edited_pH_{args.target_pH}_str_{args.strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()