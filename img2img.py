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
from waviness import box_blur
# ph_warp does NOT import img2img at its own module level (only lazily, inside function
# bodies that need edit_image) specifically so this import stays one-directional - see
# ph_warp.refine_texture's docstring comment for the reasoning.
from ph_warp import apply_ph_waviness, synth_displacement, warp_filament, predicted_wavelength

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

def create_blending_mask(window_size, device):
    """create a 2D blending mask using a Hann window for smooth transitions between overlapping patches

    The window is taken from the INTERIOR of a Hann window of length window_size + 2.
    A plain hann_window(window_size) is exactly 0 at both endpoints, and the canvas border
    is covered by only one patch, so weight_global there is 0. Dividing by the clamped
    weight then yields velocity 0, freezing row 0 and column 0 at their initial noise
    value - the speckled 1px edge that appeared on the left/top of every result. Trimming
    the endpoints keeps the window strictly positive while staying symmetric.
    """
    window_1d = torch.hann_window(window_size + 2, periodic=False, device=device)[1:-1]
    mask_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    return mask_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)


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

def safe_mirror_pad_4d(img_tensor, target_h, target_w):
    """
    Pads a 4D tensor (N, C, H, W) to at least target_h and target_w using mirror padding.
    If the tensor is already larger than the target dimensions, it will be returned unchanged.
    """
    # height mirror padding (2nd dimension)
    while img_tensor.shape[2] < target_h:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[2])], dim=2)
        
    # width mirror padding (3rd dimension)
    while img_tensor.shape[3] < target_w:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[3])], dim=3)
        
    # Crop to the exact target size if it exceeds
    return img_tensor[:, :, :target_h, :target_w]

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
               ph_rescale=False):
    """
    Edits the reference image to change its pH from source_pH to target_pH using a sliding window approach.

    solver: "euler" (1st order) or "heun" (2nd order predictor-corrector). Heun evaluates
    the velocity field twice per step (2x model calls) but has O(dt^2) local error instead
    of O(dt), which reduces integration drift over the num_steps trajectory.

    Either pH may lie outside the trained range: ph_control.velocity_for_pH extrapolates
    along the acidic->alkaline direction rather than pushing an out-of-range value through
    the periodic embedding, which would alias. ph_lambda forces a specific extrapolation
    strength instead of deriving one from target_pH, and exists for calibrate_ph.py.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, _, h, w = ref_image.shape

    target_h = max(window_size, ((h + stride - 1) // stride) * stride) + stride
    target_w = max(window_size, ((w + stride - 1) // stride) * stride) + stride

    padded_ref = safe_mirror_pad_4d(ref_image, target_h, target_w)

    t_start = 1.0 - denoising_strength
    noise = torch.randn_like(padded_ref)
    x = (1 - t_start) * noise + t_start * padded_ref

    start_step = int(t_start * num_steps)
    mask = create_blending_mask(window_size, ref_image.device)

    def compute_v_dir(x_in, step_idx):
        """Velocity field at (x_in, t=step_idx/num_steps), blended across sliding windows."""
        t = torch.full((1,), step_idx / num_steps, device=DEVICE)

        progress = (step_idx - start_step) / max(1, num_steps - start_step)
        current_scale = contrastive_scale * (1.0 - progress) + 1.0 * progress

        v_source_global = torch.zeros_like(x_in)
        v_target_global = torch.zeros_like(x_in)
        weight_global = torch.zeros_like(x_in)

        for y in range(0, target_h - window_size + 1, stride):
            for x_idx in range(0, target_w - window_size + 1, stride):
                x_patch = x_in[:, :, y:y+window_size, x_idx:x_idx+window_size]

                v_src_patch = velocity_for_pH(model, x_patch, t, source_pH,
                                              rescale=ph_rescale)
                v_tgt_patch = velocity_for_pH(model, x_patch, t, target_pH,
                                              rescale=ph_rescale, lam_override=ph_lambda)

                v_source_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += v_src_patch * mask
                v_target_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += v_tgt_patch * mask
                weight_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += mask

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

    x_cropped = x[:, :, :h, :w]

    out = (x_cropped.clamp(-1, 1) + 1) / 2

    return apply_contrast(out, contrast, contrast_mode)


@torch.no_grad()
def edit_to_pH(model, ref_image, source_pH, target_pH, seed=None, **kw):
    """Edit a real crop to ANY pH, dispatching to whichever mechanism actually works there.

    This is the entry point `main()` uses below - `edit_image` alone is not enough once
    target_pH is outside [PH_MIN, PH_MAX]. velocity_for_pH's extrapolation (used inside
    edit_image, and directly by sample.py) is only validated for free generation from pure
    noise; under an img2img anchor it was measured to barely move the output in either
    direction (waviness stayed flat at ~6.5px across pH 3.0-7.3 during editing "no matter how
    hard the conditioning was pushed" - see ph_warp.py). So for editing, both out-of-range
    directions instead: (1) edit normally to the nearer trained anchor via edit_image, then
    (2) geometrically reshape the result's pixels to hit a physically-extrapolated waviness
    target (ph_warp.apply_ph_waviness) - this is genuine postprocessing, not something the
    network is doing on its own.

    Returns (image in [0,1], info dict). `contrast`/`contrast_mode` in kw apply only to the
    anchor-conditioned edit_image call; the geometric step is contrast-neutral (it moves
    pixels, not their values). Everything else in kw is forwarded to edit_image unchanged.
    """
    anchor = min(max(target_pH, PH_MIN), PH_MAX)
    out = edit_image(model=model, ref_image=ref_image, source_pH=source_pH,
                     target_pH=anchor, seed=seed, **kw)
    if PH_MIN <= target_pH <= PH_MAX:
        return out, {"mode": "conditioned", "warped": False}

    # Outside the range in EITHER direction the geometry is imposed, because the model's own
    # response is unreliable there under an img2img anchor: it cannot buckle harder than pH
    # 8.8, and it cannot straighten either.
    # Size the displacement against the REAL source, then apply it to the model's output.
    # Measuring the output directly does not work: img2img results are softer and carry
    # faint ghost filaments, so the per-column trace wanders between them and reports ~9px of
    # waviness even for an in-range edit of an 8.4px source. Fed that, the closed loop
    # concludes the filament is already wavy enough and barely warps at all.
    canvas = out * 2 - 1
    if target_pH < PH_MIN:
        # Straightening scales the filament's OWN traced centreline, so it must read the
        # image it is straightening; a source-derived displacement would not line up.
        straightened, plan = apply_ph_waviness(canvas, target_pH, seed=seed)
        return ((straightened + 1) / 2).clamp(0, 1), {"mode": "warp", **plan}

    _, plan = apply_ph_waviness(ref_image, target_pH, seed=seed)
    info = {"mode": "warp", **plan}
    if not plan.get("warped") or not plan.get("applied_rms"):
        return out, info

    generator = torch.Generator(device=canvas.device).manual_seed(seed or 0)
    displacement = synth_displacement(canvas.shape[3], plan["applied_rms"],
                                      plan["wavelength"] or predicted_wavelength(target_pH),
                                      canvas.device, generator=generator)
    warped, info["fit_scale"] = warp_filament(canvas, displacement,
                                              extend=plan.get("mode_detail") != "straighten")
    return ((warped + 1) / 2).clamp(0, 1), info


def visualize_difference(original_tensor, edited_tensor, original_size, source_pH, target_pH):
    """Visualizes the difference between the original and edited images.

    A geometric pH warp (out-of-range target, see edit_to_pH) can grow the canvas taller to
    fit the requested waviness, so the edited image's height may exceed the original's - an
    elementwise diff is meaningless (and impossible) against pixels that don't exist in the
    original, so that case shows the two images side by side without a diff panel instead.
    """
    orig_w, orig_h = original_size
    edit_h = edited_tensor.shape[2]

    orig_img = ((original_tensor[:, :, :orig_h, :orig_w].squeeze().cpu() + 1) / 2)
    edit_img = edited_tensor[:, :, :edit_h, :orig_w].squeeze().cpu()

    if edit_h != orig_h:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"Original Image (pH: {source_pH:.2f})")
        axes[1].imshow(edit_img, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Edited Image (pH: {target_pH:.2f}) - canvas extended by the warp"
                          f" ({orig_h}px -> {edit_h}px), no diff map")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        return

    diff_map = torch.abs(orig_img - edit_img)

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
                             "are reached automatically via edit_to_pH: edit to the nearer "
                             "trained anchor, then geometrically reshape the result toward the "
                             "requested pH's physically-extrapolated waviness; see ph_warp.py")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cfm_best_ema.pt", help="Path to the model checkpoint")
    parser.add_argument("--strength", type=float, default=0.65, help="Editing strength [0.0 - 1.0] (corresponds to noise level)")
    parser.add_argument("--contrastive_scale", type=float, default=3.0, help="Contrastive scale for editing")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for editing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast strength (optional)")
    parser.add_argument("--contrast_mode", type=str, default="linear", choices=["linear", "gamma"],
                        help="'linear' scales around the mean (preserves brightness); "
                             "'gamma' is the original img**contrast, kept to reproduce older runs")
    parser.add_argument("--repair_gaps", action="store_true",
                        help="Bridge short bright breaks in the fibre before editing. Off by "
                             "default - it modifies the source anchor. Writes a before/after "
                             "diagnostic next to the result.")
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"], help="ODE solver for the editing trajectory")

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
    print(describe_pH(args.target_pH, editing=True))

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
        contrast_mode=args.contrast_mode
    )

    if edit_info["mode"] == "warp":
        current, target = edit_info.get("current"), edit_info.get("target")
        detail = f"target {target:.1f}px waviness" if target is not None else "no target"
        if current is not None:
            detail = f"current ~{current:.1f}px -> {detail}"
        achieved = edit_info.get("achieved")
        if achieved is not None:
            detail += f" (achieved ~{achieved:.1f}px)"
        print(f"Out-of-range pH: edited to the nearest trained anchor, then geometrically "
              f"{'damped' if edit_info.get('mode_detail') == 'straighten' else 'added'} "
              f"waviness ({detail}).")

    # compare against the unrepaired source - the repair is an input-side aid, not a result
    visualize_difference(display_ref, edited_img, original_size, source_pH=args.source_pH, target_pH=args.target_pH)

    orig_w, orig_h = original_size
    edit_h = edited_img.shape[2]
    if edit_h != orig_h:
        print(f"Note: the geometric pH warp extended the canvas ({orig_h}px -> {edit_h}px "
              f"tall) to fit the requested waviness; the saved image keeps the full height.")
    edited_crop_for_save = edited_img[:, :, :edit_h, :orig_w]

    save_path = f"outputs_img2img/edited_pH_{args.target_pH}_str_{args.strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()