"""Calibrate the pH extrapolation, and write ph_calibration.json.

Two curves get fitted and then composed:

  1. PHYSICS   waviness of REAL crops vs pH  ->  how wavy a filament at pH q should be
  2. RESPONSE  waviness of GENERATED crops vs lambda  ->  what lambda delivers that

Composing them turns lambda, an arbitrary guidance knob, into a physical pH request.
Re-run this after training a new checkpoint; nothing else needs to change.

    python3 calibrate_ph.py --checkpoint checkpoints/cfm_best_ema.pt
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import ph_control
from config import DEVICE, PH_MAX, PH_MIN
from img2img import edit_image, load_and_preprocess_image
from model import ConditionalUNet
from waviness import waviness

DATA_DIR = "data/cropped/cropped_output"


def measure_real(data_dir, min_w=128, min_h=32):
    """Waviness of every readable real crop, grouped by its pH folder."""
    per_ph = {}
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
            if w < min_w or h < min_h:
                continue
            ref, _ = load_and_preprocess_image(os.path.join(path, name))
            value = waviness(ref)
            if value is not None:
                per_ph.setdefault(ph, []).append(value)
    return per_ph


def pick_sources(data_dir, per_bucket, min_w=200, min_h=32):
    """A few clean, wide crops to run the lambda sweep on."""
    picked = []
    for folder in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, folder)
        try:
            ph = float(folder)
        except ValueError:
            continue
        if not os.path.isdir(path):
            continue
        good = []
        for name in sorted(os.listdir(path)):
            if not name.endswith(".png"):
                continue
            w, h = Image.open(os.path.join(path, name)).size
            if w >= min_w and h >= min_h:
                good.append(os.path.join(path, name))
        picked += [(p, ph) for p in good[:per_bucket]]
    return picked


def measure_response(model, sources, lambdas, steps, strength, seed):
    """Waviness of generated output as a function of extrapolation strength.

    Each source is edited toward the alkaline end at a sequence of lambdas. lambda = 0 is
    plain pH 8.8 conditioning, so it doubles as the anchor point both curves must agree on.
    """
    out = {lam: [] for lam in lambdas}
    for idx, (src, source_ph) in enumerate(sources, 1):
        ref, _ = load_and_preprocess_image(src)
        for lam in lambdas:
            edited = edit_image(
                model=model, ref_image=ref, source_pH=source_ph, target_pH=PH_MAX,
                denoising_strength=strength, num_steps=steps, contrastive_scale=1.0,
                seed=seed, contrast=1.0, solver="heun", ph_lambda=lam)
            value = waviness(edited * 2 - 1)
            if value is not None:
                out[lam].append(value)
        print(f"  [{idx}/{len(sources)}] {os.path.basename(src)[:44]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/cfm_best_ema.pt")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.5, 4.0])
    ap.add_argument("--per_bucket", type=int, default=1)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=ph_control.CALIBRATION_PATH)
    args = ap.parse_args()

    print("[1/3] measuring waviness of the real crops")
    per_ph = measure_real(DATA_DIR)
    phs, means = [], []
    print(f"  {'pH':>5} {'n':>4} {'mean rms_dev':>13}")
    for ph in sorted(per_ph):
        vals = np.array(per_ph[ph])
        phs.append(ph); means.append(vals.mean())
        print(f"  {ph:5.1f} {len(vals):4d} {vals.mean():13.2f}")

    flat_ph = np.concatenate([[ph] * len(v) for ph, v in sorted(per_ph.items())])
    flat_w = np.concatenate([v for _, v in sorted(per_ph.items())])
    slope, intercept = np.polyfit(flat_ph, flat_w, 1)
    resid = flat_w - (slope * flat_ph + intercept)
    r2 = 1 - resid.var() / flat_w.var()
    print(f"  fit over {len(flat_w)} crops: rms_dev = {slope:.3f}*pH {intercept:+.3f}"
          f"   (R^2 = {r2:.3f})")

    print("\n[2/3] measuring the generator's response to lambda")
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    sources = pick_sources(DATA_DIR, args.per_bucket)
    print(f"  {len(sources)} sources x {len(args.lambdas)} lambdas")
    response = measure_response(model, sources, args.lambdas, args.num_steps,
                                args.strength, args.seed)

    lam_x = np.array([l for l in args.lambdas if response[l]])
    lam_y = np.array([np.mean(response[l]) for l in lam_x])
    print(f"\n  {'lambda':>7} {'n':>4} {'mean rms_dev':>13}")
    for l, y in zip(lam_x, lam_y):
        print(f"  {l:7.2f} {len(response[l]):4d} {y:13.2f}")

    # lambda per extra pixel of waviness, measured against the lambda = 0 baseline
    gain_num, gain_den = lam_x - lam_x[0], lam_y - lam_y[0]
    usable = gain_den > 1e-6
    lambda_per_px = (float(np.polyfit(gain_den[usable], gain_num[usable], 1)[0])
                     if usable.sum() >= 2 else _DEFAULT_GAIN)
    monotonic = bool(np.all(np.diff(lam_y) > -0.15))

    cal = {
        "checkpoint": os.path.basename(args.checkpoint),
        "waviness_slope": float(slope),
        "waviness_intercept": float(intercept),
        "waviness_r2": float(r2),
        "lambda_per_px": lambda_per_px,
        "max_lambda": float(max(args.lambdas)),
        "response_lambdas": [float(v) for v in lam_x],
        "response_waviness": [float(v) for v in lam_y],
        "response_monotonic": monotonic,
    }
    with open(args.out, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"\n[3/3] wrote {args.out}")
    print(f"  lambda_per_px = {lambda_per_px:.3f}   response monotonic: {monotonic}")
    if not monotonic:
        print("  WARNING: waviness did not rise monotonically with lambda - the "
              "extrapolation is unreliable past the point where it turns over.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(flat_ph + np.random.uniform(-.04, .04, len(flat_ph)), flat_w,
                    s=7, alpha=.25, color="#0f8177", linewidths=0)
    axes[0].plot(phs, means, "o-", color="#0f8177", lw=2, label="bucket mean")
    grid = np.linspace(min(phs) - 2, max(phs) + 3, 100)
    axes[0].plot(grid, slope * grid + intercept, "--", color="#b06a12",
                 label=f"fit (R²={r2:.2f}), extrapolated")
    axes[0].axvspan(PH_MIN, PH_MAX, color="#0f8177", alpha=.07)
    axes[0].set_xlabel("pH"); axes[0].set_ylabel("rms centreline deviation (px)")
    axes[0].set_title("physics: real crops", fontsize=10); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)

    axes[1].plot(lam_x, lam_y, "o-", color="#6f4aa0", lw=2)
    axes[1].set_xlabel("lambda (extrapolation strength)")
    axes[1].set_ylabel("rms centreline deviation (px)")
    axes[1].set_title("response: generated output", fontsize=10); axes[1].grid(alpha=.25)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/ph_calibration.png", dpi=130)
    print("  curve saved to outputs/ph_calibration.png")


_DEFAULT_GAIN = 0.35

if __name__ == "__main__":
    main()
