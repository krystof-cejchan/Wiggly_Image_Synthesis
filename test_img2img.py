"""Sweep img2img over several source images and contrastive_scale values.

Runs the real edit_image() pipeline from img2img.py (nothing is reimplemented here) and
writes one comparison figure per source image: the original on top, then the generated
result at each contrastive_scale below it. Fully headless - no window is ever opened.

    python3 test_img2img.py
    python3 test_img2img.py --contrastive_scales 1 3 5 --num_steps 50

Every scale for a given source reuses the same seed, so the initial noise is identical
across a row and the only thing that varies is contrastive_scale.
"""

import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (img2img imports it) or a display
                       # backend gets selected and savefig can try to open a window

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from PIL import Image

from config import PH_MIN, PH_MAX, DEVICE
from img2img import (edit_image, load_and_preprocess_image, repair_fibre_gaps,
                     save_repair_diagnostic)
from model import ConditionalUNet

DATA_DIR = "data/cropped/cropped_output"
DEFAULT_SCALES = [1.0, 2.0, 3.0, 5.0, 7.0]


def discover_sources(data_dir, per_bucket, seed, min_width, min_height):
    """Pick `per_bucket` images from every pH folder. The folder name is the source pH.

    Crops smaller than the 128x128 sliding window are skipped. edit_image mirror-pads
    anything undersized up to the window, so a 44x37 crop becomes a 3x5 tiled grid of
    itself and the model faithfully generates that tiling instead of a single fiber -
    the result is unreadable noise. About a third of the dataset is this small.
    """
    rng = random.Random(seed)
    sources = []
    skipped = 0
    for ph_folder in sorted(os.listdir(data_dir), key=lambda p: float(p) if _is_ph(p) else 0.0):
        ph_dir = os.path.join(data_dir, ph_folder)
        if not (os.path.isdir(ph_dir) and _is_ph(ph_folder)):
            continue
        candidates = []
        for name in sorted(f for f in os.listdir(ph_dir) if f.endswith(".png")):
            width, height = Image.open(os.path.join(ph_dir, name)).size
            if width >= min_width and height >= min_height:
                candidates.append(name)
            else:
                skipped += 1
        if not candidates:
            print(f"  warning: pH {ph_folder} has no crop >= {min_width}x{min_height} - skipping")
            continue
        for name in rng.sample(candidates, min(per_bucket, len(candidates))):
            sources.append((os.path.join(ph_dir, name), float(ph_folder)))
    return sources, skipped


def _is_ph(folder_name):
    try:
        float(folder_name)
        return True
    except ValueError:
        return False


def pick_target_pH(source_pH):
    """Default target: the far end of the trained pH range, so the edit is clearly visible."""
    midpoint = (PH_MIN + PH_MAX) / 2
    return PH_MAX if source_pH <= midpoint else PH_MIN


def save_comparison(orig_img, results, out_path, source_pH, target_pH, src_name, args):
    """One figure: original on top, one row per contrastive_scale below it.

    Crops are wide, short strips, so rows stack vertically rather than side by side.
    """
    n_rows = 1 + len(results)
    img_h, img_w = orig_img.shape
    aspect = img_h / img_w
    row_h = max(1.1, 11.0 * aspect)

    fig, axes = plt.subplots(n_rows, 1, figsize=(11, row_h * n_rows + 0.6))
    axes = [axes] if n_rows == 1 else list(axes)

    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original - pH {source_pH:g}  ({src_name})", fontsize=9)

    for ax, (scale, gen_img, mad) in zip(axes[1:], results):
        ax.imshow(gen_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"pH {source_pH:g} -> {target_pH:g}   contrastive_scale={scale:g}   "
            f"mean|diff|={mad:.3f}",
            fontsize=9,
        )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"img2img sweep - strength={args.strength:g}, steps={args.num_steps}, "
        f"contrast={args.contrast:g} ({args.contrast_mode}), solver={args.solver}, "
        f"seed={args.seed}{', gaps repaired' if args.repair_gaps else ''}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)  # closing matters in a loop - open figures accumulate and leak memory


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cfm_best_emav2.pt")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--out_dir", type=str, default="outputs_img2img/sweep")
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--contrast", type=float, default=2.0)
    parser.add_argument("--contrastive_scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--target_pH", type=float, default=None,
                        help="Target pH for every source. Default: the far end of "
                             f"[{PH_MIN}, {PH_MAX}] relative to each source.")
    parser.add_argument("--per_bucket", type=int, default=1,
                        help="How many source images to draw from each pH folder")
    parser.add_argument("--min_width", type=int, default=128,
                        help="Skip crops narrower than this (they get mirror-tiled into noise)")
    parser.add_argument("--min_height", type=int, default=32,
                        help="Skip crops shorter than this (they get mirror-tiled into noise)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--contrast_mode", type=str, default="linear", choices=["linear", "gamma"],
                        help="'linear' preserves mean brightness; 'gamma' reproduces older runs")
    parser.add_argument("--repair_gaps", action="store_true",
                        help="Bridge short bright breaks in the fibre before editing (off by "
                             "default; writes a before/after diagnostic per source)")
    parser.add_argument("--save_raw", action="store_true",
                        help="Also write each generated image as a standalone PNG")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    sources, skipped = discover_sources(
        args.data_dir, args.per_bucket, args.seed, args.min_width, args.min_height)
    if not sources:
        raise SystemExit(f"No source images found under {args.data_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw")
    if args.save_raw:
        os.makedirs(raw_dir, exist_ok=True)

    total = len(sources) * len(args.contrastive_scales)
    print(f"device: {DEVICE} | checkpoint: {args.checkpoint}")
    print(f"skipped {skipped} crops below {args.min_width}x{args.min_height}")
    print(f"{len(sources)} source images x {len(args.contrastive_scales)} scales = {total} runs")

    run = 0
    for src_path, source_pH in sources:
        target_pH = args.target_pH if args.target_pH is not None else pick_target_pH(source_pH)
        ref_image, original_size = load_and_preprocess_image(src_path)
        orig_w, orig_h = original_size
        # the figure always shows the untouched source, even when the anchor was repaired
        orig_img = ((ref_image[0, 0, :orig_h, :orig_w].cpu() + 1) / 2)
        stem = os.path.splitext(os.path.basename(src_path))[0]

        if args.repair_gaps:
            repaired, gaps = repair_fibre_gaps(ref_image)
            if gaps:
                spans = ", ".join(f"{g['start']}-{g['end']}" for g in gaps)
                print(f"  repaired {len(gaps)} gap(s) in {stem}: cols {spans}")
                save_repair_diagnostic(ref_image, repaired, gaps,
                                       os.path.join(args.out_dir, f"repair_{stem}.png"))
                ref_image = repaired

        results = []
        for scale in args.contrastive_scales:
            run += 1
            edited = edit_image(
                model=model,
                ref_image=ref_image,
                source_pH=source_pH,
                target_pH=target_pH,
                denoising_strength=args.strength,
                num_steps=args.num_steps,
                contrastive_scale=scale,
                seed=args.seed,  # same noise for every scale -> differences are the scale alone
                contrast=args.contrast,
                solver=args.solver,
                contrast_mode=args.contrast_mode,
            )
            gen_img = edited[0, 0, :orig_h, :orig_w].cpu()
            mad = torch.abs(gen_img - orig_img).mean().item()
            results.append((scale, gen_img, mad))

            if args.save_raw:
                vutils.save_image(edited[:, :, :orig_h, :orig_w],
                                  os.path.join(raw_dir, f"{stem}_cs{scale:g}.png"))
            print(f"  [{run:3d}/{total}] {stem} | pH {source_pH:g}->{target_pH:g} | "
                  f"cs={scale:<4g} | mean|diff|={mad:.4f}")

        out_path = os.path.join(
            args.out_dir, f"cmp_pH{source_pH:g}to{target_pH:g}_{stem}.png")
        save_comparison(orig_img, results, out_path, source_pH, target_pH, stem, args)
        print(f"  -> {out_path}")

    print(f"\nDone. {total} runs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
