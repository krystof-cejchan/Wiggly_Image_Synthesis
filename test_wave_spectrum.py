"""Is the generated filament wavy in the same WAY a real one is, or one long arc?

This is the diagnostic the ripple conditioning exists to satisfy, and the one to run after
retraining. It decomposes the traced centreline into wavelength bands and prints generated
against real, per band, in pixels of rms excursion.

Why bands rather than a single number: waviness() is one rms and wave_period() is one
spectral peak, and a filament that is entirely one frame-wide S-curve can score the same on
both as a filament covered in small ripples. Measured on the period-conditioned checkpoint,
a pH 5.8->8.8 edit put 8.60px of rms into the 96-192px band where real crops have 2.75, and
0.73px into 24-48px where real crops have 2.83 - 3x too much long wave, 4x too little fine
wiggle - while its total rms (11.03 vs 7.16) looked merely somewhat high. Only the band
breakdown says WHICH of those two shapes came out.

Real crops spread their centreline energy almost evenly over every band; that flatness is
the target, and RIPPLE (the 24-96px total, waviness.ripple_rms) is the single number to
watch, together with its share of the total - real data holds it near 0.61 at every pH.

    python3 test_wave_spectrum.py
    python3 test_wave_spectrum.py --source_pH 5.8 --target_pH 8.8 --num_steps 100
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import os

import numpy as np
import torch

from config import DEVICE, CHECKPOINT_PATH, TRAIN_MIN_W, TRAIN_MAX_W
from framing import mirror_pad_width, pad_height_background
from img2img import (edit_image, frame_height, load_and_preprocess_image,
                     measure_frame_ripple, measure_frame_waviness, plan_window_starts)
from model import from_state_dict
from ph_control import predicted_ripple, predicted_waviness_native
from ph_warp import edit_to_pH
from waviness import (RIPPLE_MAX_WAVELENGTH, RIPPLE_MIN_WAVELENGTH, band_profile, box_blur)

DATA_DIR = "data/cropped/cropped_output"
# Boundaries chosen so the ripple band (24-96px) is two of them and the rest split what is
# left: >192 is the frame-scale drift, <24 is at the tracer's own noise floor (a dead-straight
# fibre already measures 0.71px there) and is reported only so it is visibly not signal.
BANDS = [(192.0, 1e9, ">192"), (96.0, 192.0, "96-192"), (48.0, 96.0, "48-96"),
         (24.0, 48.0, "24-48"), (0.0, 24.0, "<24")]


def _windows(img4):
    """The frames the model actually sees, so the measurement matches the conditioning."""
    frame_h = frame_height(img4.shape[2])
    framed = (pad_height_background(img4, frame_h, jitter=0.0)
              if frame_h > img4.shape[2] else img4)
    target_w = max(TRAIN_MIN_W, ((framed.shape[3] + 15) // 16) * 16)
    framed = mirror_pad_width(framed, target_w)
    win_w = min(target_w, TRAIN_MAX_W)
    return [framed[:, :, :, x:x + win_w]
            for x in plan_window_starts(target_w, win_w, win_w // 2)]


def profile(img4):
    """Median per-band rms over every window of one image, or None if nothing traces."""
    rows = [row for row in (band_profile(win, BANDS) for win in _windows(img4))
            if row is not None]
    if not rows:
        return None
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def real_profile(data_dir, pH, min_w=200, min_h=24):
    """The same profile over every real crop in one pH bucket - the target to hit."""
    folder = os.path.join(data_dir, f"{pH:g}")
    if not os.path.isdir(folder):
        return None
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".png"):
            continue
        img, (w, h) = load_and_preprocess_image(os.path.join(folder, name))
        if w < min_w or h < min_h:
            continue
        got = profile(img)
        if got:
            rows.append(got)
    if not rows:
        return None
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def _render_row(label, img01):
    """Fibre darkness and continuity - the half of the result a band profile cannot see.

    depth: median over columns of (background level - darkest pixel), on a [-1,1] image, so
    a real crop's near-black filament on mid-grey noise scores ~0.9 and a faint smear ~0.35.
    cont:  fraction of columns whose depth reaches half the strong-column depth, i.e. how
           much of the fibre's length is actually drawn rather than broken.
    """
    if img01 is None:
        return f"{label:>16}  (no example available)"
    x = img01 * 2 - 1
    smoothed = box_blur(x, 3)[0, 0]
    background = smoothed.median(dim=0).values
    darkest, _ = smoothed.min(dim=0)
    depth = background - darkest
    strong = torch.quantile(depth, 0.75)
    continuity = float((depth > 0.5 * strong).float().mean())
    return f"{label:>16}  depth {float(depth.median()):.2f}   continuity {continuity:.2f}"


def _real_example(data_dir, pH, min_w=200, min_h=24):
    """One real crop at this pH, for the rendering comparison - the widest that qualifies."""
    folder = os.path.join(data_dir, f"{pH:g}")
    if not os.path.isdir(folder):
        return None
    best = None
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".png"):
            continue
        img, (w, h) = load_and_preprocess_image(os.path.join(folder, name))
        if w >= min_w and h >= min_h and (best is None or w > best[0]):
            best = (w, img)
    return None if best is None else (best[1] + 1) / 2


def _row(label, prof, keys):
    return f"{label:>22} " + " ".join(f"{prof[k]:>8.2f}" for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--ref_image", default=None,
                    help="Source crop. Default: the widest crop in the --source_pH bucket.")
    ap.add_argument("--source_pH", type=float, default=5.8)
    ap.add_argument("--target_pH", type=float, default=8.8)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--contrastive_scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--geometry_mode", default="warp", choices=["warp", "native"],
                    help="Which mechanism to measure - the same flag img2img.py takes. "
                         "'warp' (default) is the product path: the model re-renders "
                         "texture with the fibre held still and a broadband pixel warp "
                         "imposes the shape. 'native' measures the model's own geometry "
                         "conditioning in isolation, which is what this script was "
                         "originally written to check.")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    ref_path = args.ref_image
    if ref_path is None:
        folder = os.path.join(DATA_DIR, f"{args.source_pH:g}")
        best = None
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".png"):
                continue
            _, (w, h) = load_and_preprocess_image(os.path.join(folder, name))
            if h >= 24 and (best is None or w > best[0]):
                best = (w, os.path.join(folder, name))
        ref_path = best[1]
    ref, (w, h) = load_and_preprocess_image(ref_path)
    print(f"source: {os.path.basename(ref_path)} ({w}x{h}) at pH {args.source_pH:g}")

    model = from_state_dict(torch.load(args.checkpoint, map_location=DEVICE), DEVICE)
    model.eval()
    print(f"checkpoint: {args.checkpoint}  (ripple channel: "
          f"{'present' if getattr(model, 'ripple_conditioned', False) else 'ABSENT'})")

    target_waviness = predicted_waviness_native(args.target_pH)
    target_ripple = predicted_ripple(args.target_pH)
    source_waviness = measure_frame_waviness(ref) or predicted_waviness_native(args.source_pH)
    source_ripple = measure_frame_ripple(ref) or predicted_ripple(args.source_pH)
    if args.geometry_mode == "native":
        print(f"requested: waviness {source_waviness:.2f} -> {target_waviness:.2f}px, "
              f"ripple {source_ripple:.2f} -> {target_ripple:.2f}px "
              f"(share {target_ripple / max(target_waviness, 1e-6):.2f})")

    with torch.no_grad():
        if args.geometry_mode == "native":
            out = edit_image(model=model, ref_image=ref, source_pH=args.source_pH,
                             target_pH=args.target_pH, num_steps=args.num_steps,
                             contrastive_scale=args.contrastive_scale, seed=args.seed,
                             contrast=1.0, solver="heun",
                             source_waviness=source_waviness, target_waviness=target_waviness,
                             source_ripple=source_ripple, target_ripple=target_ripple)
        else:
            # edit_to_pH, not edit_image: the warp lives there, and measuring edit_image
            # alone would score a mechanism the CLI no longer uses on its own.
            out, info = edit_to_pH(model=model, ref_image=ref, source_pH=args.source_pH,
                                   target_pH=args.target_pH, num_steps=args.num_steps,
                                   contrastive_scale=args.contrastive_scale, seed=args.seed,
                                   contrast=1.0, solver="heun", geometry_mode="warp")
            if not info.get("warped"):
                print("warp: NOT APPLIED - the fibre could not be traced in this crop")
            elif info.get("mode_detail") == "straighten":
                # The source is already wavier than the target pH calls for, so the warp
                # shears its own traced centreline down instead of adding to it; there is
                # no synthesised displacement and so no ripple fraction to report.
                print(f"warp: STRAIGHTEN {info.get('current', 0.0):.2f} -> "
                      f"{info.get('target', 0.0):.2f}px total, by shearing the traced "
                      f"centreline (rms of the shear {info.get('applied_rms', 0.0):.2f}px)")
            else:
                print(f"warp: {info.get('current', 0.0):.2f} -> {info.get('target', 0.0):.2f}px "
                      f"total and {info.get('current_ripple') or 0.0:.2f} -> "
                      f"{info.get('target_ripple') or 0.0:.2f}px ripple, via a "
                      f"{info.get('applied_rms', 0.0):.2f}px displacement that is "
                      f"{(info.get('ripple_fraction') or 0.0) * 100:.0f}% ripple")

    # How DARK and how CONTINUOUS the fibre came out, alongside its shape. Both matter and
    # they fail independently: the geometry can land exactly on the real per-band profile
    # while the fibre itself renders as a faint broken smear, which is precisely what the
    # model's own conditioning does when it has to draw the filament somewhere it must
    # invent (measured 0.35-0.39 depth against 0.86 for a real crop at the same pH).
    print(f"\nfibre rendering:  {_render_row('generated', out)}")
    print(f"                  {_render_row(f'real pH {args.target_pH:g}', _real_example(DATA_DIR, args.target_pH))}")
    print(f"                  {_render_row('source', (ref + 1) / 2)}")

    keys = [name for _, _, name in BANDS] + ["ripple", "total"]
    print(f"\nrms of the traced centreline, per wavelength band (px). "
          f"ripple = {RIPPLE_MIN_WAVELENGTH:g}-{RIPPLE_MAX_WAVELENGTH:g}px")
    print(f"{'':>22} " + " ".join(f"{k:>8}" for k in keys))
    gen = profile(out * 2 - 1)
    if gen is None:
        raise SystemExit("no fibre could be traced in the generated image")
    print(_row("generated", gen, keys))
    real = real_profile(DATA_DIR, args.target_pH)
    if real:
        print(_row(f"real pH {args.target_pH:g}", real, keys))
        print(_row("ratio gen/real", {k: gen[k] / max(real[k], 1e-6) for k in keys}, keys))
        print(f"\nripple share of total:  generated {gen['ripple'] / max(gen['total'], 1e-6):.2f}"
              f"   real {real['ripple'] / max(real['total'], 1e-6):.2f}")
        print("A ratio near 1.0 across every band is the goal. The failure this diagnostic "
              "was written for\nshows up as >2 in 96-192 together with <0.5 in 24-48: one "
              "long arc instead of many waves.")


if __name__ == "__main__":
    main()
