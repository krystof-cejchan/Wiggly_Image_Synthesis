import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF
from config import PH_MIN, PH_MAX, DEVICE
from model import ConditionalUNet
from ph_control import normalize_pH, velocity_for_pH, describe as describe_pH
from ph_warp import edit_to_pH
from framing import pad_height_background, mirror_pad_width
from waviness import box_blur

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
def edit_image(model, ref_image, source_pH, target_pH, denoising_strength=0.5,
               num_steps=100, contrastive_scale=3.0, seed=None, window_size=128, stride=64,
               contrast=1.2, solver="heun", contrast_mode="linear", ph_lambda=None,
               ph_rescale=False, source_waviness=None, target_waviness=None):
    """
    Edits the reference image to change its pH from source_pH to target_pH using a sliding window approach.

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
    model.py's WavinessEmbedding). ph_warp.edit_to_pH's geometry_mode="native" is what sets
    these; a plain call here still defaults to today's behaviour unchanged.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, _, h, w = ref_image.shape

    # Height: grow only to the next multiple of 16 (the U-Net's 4 downsampling stages), with
    # SYNTHESISED BACKGROUND rather than reflection. The old code padded to
    # max(window_size, ...) + stride - 192px for a 55px crop - which mirror-tiled the fibre
    # 3.5x and then asked the model to edit a hall of mirrors. See framing.py.
    pad_h = ((h + 15) // 16) * 16
    top_offset = (pad_h - h) // 2
    padded_ref = pad_height_background(ref_image, pad_h, jitter=0.0) if pad_h > h else ref_image
    # Width: reflection here is safe (it keeps the centreline single-valued - framing.py).
    target_w = max(window_size, ((w + stride - 1) // stride) * stride) + stride
    padded_ref = mirror_pad_width(padded_ref, target_w)
    target_h = padded_ref.shape[2]

    t_start = 1.0 - denoising_strength
    noise = torch.randn_like(padded_ref)
    x = (1 - t_start) * noise + t_start * padded_ref

    start_step = int(t_start * num_steps)
    # one full-height window, sliding horizontally only: no vertical taper needed
    mask = create_blending_mask(target_h, window_size, ref_image.device, taper_h=False)

    def compute_v_dir(x_in, step_idx):
        """Velocity field at (x_in, t=step_idx/num_steps), blended across sliding windows."""
        t = torch.full((1,), step_idx / num_steps, device=DEVICE)

        progress = (step_idx - start_step) / max(1, num_steps - start_step)
        current_scale = contrastive_scale * (1.0 - progress) + 1.0 * progress

        v_source_global = torch.zeros_like(x_in)
        v_target_global = torch.zeros_like(x_in)
        weight_global = torch.zeros_like(x_in)

        for x_idx in range(0, target_w - window_size + 1, stride):
            x_patch = x_in[:, :, :, x_idx:x_idx+window_size]

            v_src_patch = velocity_for_pH(model, x_patch, t, source_pH,
                                          rescale=ph_rescale, waviness=source_waviness)
            v_tgt_patch = velocity_for_pH(model, x_patch, t, target_pH,
                                          rescale=ph_rescale, lam_override=ph_lambda,
                                          waviness=target_waviness)

            v_source_global[:, :, :, x_idx:x_idx+window_size] += v_src_patch * mask
            v_target_global[:, :, :, x_idx:x_idx+window_size] += v_tgt_patch * mask
            weight_global[:, :, :, x_idx:x_idx+window_size] += mask

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
    #plt.show()

def main():
    parser = argparse.ArgumentParser(description="Image-to-Image pH Editing using Conditional UNet")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference image (e.g., 'data/ref_image.png')")
    parser.add_argument("--source_pH", type=float, required=True, help=f"Initial pH of the reference image (trained range {PH_MIN}-{PH_MAX})")
    parser.add_argument("--target_pH", type=float, required=True,
                        help=f"Target pH. Values outside the trained range {PH_MIN}-{PH_MAX} "
                             "are reached by extrapolating along the acidic->alkaline "
                             "direction; see ph_control.py")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cfm_best_ema.pt", help="Path to the model checkpoint")
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
    parser.add_argument("--geometry_mode", type=str, default="warp", choices=["warp", "native"],
                        help="'warp' (default) is the validated geometric pixel-warp mechanism "
                             "this CLI has always used outside the trained range. 'native' drives "
                             "geometry through the model's own waviness conditioning instead - "
                             "requires a checkpoint trained with it (see model.py's "
                             "WavinessEmbedding); training support to ~pH 16.")

    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist.")
        return
        
    if not os.path.exists(args.ref_image):
        print(f"Reference image {args.ref_image} not found. Please check the provided path.")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs_img2img", exist_ok=True)
    
    ref_image, original_size = load_and_preprocess_image(args.ref_image)
    print(f"Loaded image with original resolution: {original_size[0]}x{original_size[1]}")
    print(describe_pH(args.target_pH, geometry_mode=args.geometry_mode))

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
            if edit_info.get("fit_scale", 1.0) < 0.999:
                print(f"  Note: source crop limited the warp to "
                      f"{edit_info['fit_scale'] * 100:.0f}% of the requested displacement.")
        else:
            print("No geometric pH warp applied: the fibre could not be confidently traced "
                  "in this crop, so the result is left at the pH 5.8/8.8 anchor edit.")
    elif edit_info.get("mode") == "native":
        print(f"Native waviness conditioning applied: target waviness "
              f"{edit_info.get('target_waviness', 0.0):.1f}px (supported by training data to "
              f"~pH 16; above that compare against --geometry_mode warp)")

    # compare against the unrepaired source - the repair is an input-side aid, not a result
    visualize_difference(display_ref, edited_img, original_size, source_pH=args.source_pH, target_pH=args.target_pH)

    orig_w, orig_h = original_size
    edited_crop_for_save = edited_img[:, :, :, :orig_w]

    save_path = f"outputs_img2img/edited_pH_{args.target_pH}_str_{args.strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()