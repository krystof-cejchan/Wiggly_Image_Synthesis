"""Check that requesting a pH outside the trained range does something sensible.

Runs one source image across a span of target pH values reaching well past both ends of the
trained 5.8-8.8 range, measures the waviness of every result, and reports whether the
response is monotonic. A monotonic curve is the whole claim: the extrapolation is only
worth anything if "more alkaline" reliably means "wavier".

Writes a comparison strip and a waviness-vs-pH plot; opens no windows.

    python3 test_ph_extrapolation.py
    python3 test_ph_extrapolation.py --pH 3 4 5 6 7 8 9 10 11 12 --per_bucket 2
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import ph_control
from config import DEVICE, PH_MAX, PH_MIN
from img2img import load_and_preprocess_image
from ph_warp import edit_to_pH
from model import ConditionalUNet
from waviness import waviness

DATA_DIR = "data/cropped/cropped_output"


def pick_sources(data_dir, count, min_w=220, min_h=32):
    """Widest available crops, which trace most reliably."""
    candidates = []
    for folder in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, folder)
        try:
            ph = float(folder)
        except ValueError:
            continue
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".png"):
                continue
            w, h = Image.open(os.path.join(path, name)).size
            if w >= min_w and h >= min_h:
                candidates.append((w, os.path.join(path, name), ph))
    candidates.sort(reverse=True)
    return [(p, ph) for _, p, ph in candidates[:count]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/cfm_best_emav2.pt")
    ap.add_argument("--pH", type=float, nargs="+",
                    default=[3.0, 4.4, 5.8, 7.3, 8.8, 10.3, 11.8, 13.0])
    ap.add_argument("--sources", type=int, default=3)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--contrastive_scale", type=float, default=1.0,
                    help="Kept at 1.0 so target_pH is the only thing driving waviness")
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="outputs_img2img/ph_range")
    args = ap.parse_args()

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    sources = pick_sources(DATA_DIR, args.sources)
    print(f"{len(sources)} sources x {len(args.pH)} pH values\n")
    for ph in args.pH:
        print(" ", ph_control.describe(ph))
    print()

    per_ph = {ph: [] for ph in args.pH}
    for src, source_ph in sources:
        ref, size = load_and_preprocess_image(src)
        w, h = size
        stem = os.path.splitext(os.path.basename(src))[0]
        rows = [("source · pH %g" % source_ph, ((ref[0, 0, :h, :w].cpu() + 1) / 2))]

        for ph in args.pH:
            out, info = edit_to_pH(model=model, ref_image=ref, source_pH=source_ph,
                                   target_pH=ph, denoising_strength=args.strength,
                                   num_steps=args.num_steps,
                                   contrastive_scale=args.contrastive_scale,
                                   seed=args.seed, contrast=args.contrast, solver="heun")
            # a warp makes the image taller, so crop to whatever came back
            img = out[0, 0, :, :w].cpu()
            value = waviness(out * 2 - 1)
            if value is not None:
                per_ph[ph].append(value)
            label = f"pH {ph:g} · {info.get('mode', '?')}"
            if value is not None:
                label += f" · rms {value:.1f}px"
            rows.append((label, img))
            print(f"  {stem[:34]:34s} pH {ph:5.1f}  "
                  f"waviness {value if value is None else round(value, 2)}")

        fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.5 * len(rows)))
        for ax, (label, im) in zip(axes, rows):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"pH extrapolation · trained range {PH_MIN}-{PH_MAX} · "
                     f"strength {args.strength:g}, cs {args.contrastive_scale:g}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(args.out_dir, f"ph_range_{stem}.png"), dpi=125)
        plt.close(fig)

    xs = [p for p in args.pH if per_ph[p]]
    ys = [float(np.mean(per_ph[p])) for p in xs]
    print(f"\n{'pH':>7} {'n':>3} {'mean waviness (px)':>20} {'predicted':>10}")
    for p, y in zip(xs, ys):
        print(f"{p:7.1f} {len(per_ph[p]):3d} {y:20.2f} {ph_control.predicted_waviness(p):10.2f}")

    diffs = np.diff(ys)
    monotonic = bool(np.all(diffs > -0.15))
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
    print(f"\nmonotonic in pH: {monotonic}   Pearson r = {corr:+.3f}")
    if not monotonic:
        worst = int(np.argmin(diffs))
        print(f"  largest drop between pH {xs[worst]:g} and {xs[worst+1]:g} "
              f"({diffs[worst]:+.2f} px)")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(PH_MIN, PH_MAX, color="#0f8177", alpha=.09, label="trained range")
    ax.plot(xs, ys, "o-", color="#6f4aa0", lw=2, label="generated")
    grid = np.linspace(min(xs), max(xs), 100)
    ax.plot(grid, [ph_control.predicted_waviness(g) for g in grid], "--",
            color="#b06a12", label="physical trend, extrapolated")
    ax.set_xlabel("requested pH"); ax.set_ylabel("rms centreline deviation (px)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "waviness_vs_pH.png"), dpi=130)
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
