"""Calibrate the pH extrapolation, and write ph_calibration.json.

Four curves get fitted:

  1. PHYSICS (whole-image)  waviness of REAL crops vs pH, measured on the whole image  ->
     feeds ph_warp.py's geometric warp, which measures its `current` the same way.
  2. PHYSICS (per-crop)     the same real-data relationship, but measured through
     train.py's own dynamic_collate_fn (random TRAIN_SIZES crop/flip/jitter)  ->  feeds
     model.py's native waviness conditioning, which is trained on that same per-crop
     scale. These two are NOT interchangeable: measured directly, whole-image waviness has
     std ~3.4px while the per-crop training distribution has std ~8.5px (crops reaching up
     to 82px) - calibrating native conditioning against the whole-image number compressed
     almost its entire learned range into under one training-distribution standard
     deviation of extrapolation request.
  3. PHYSICS (ripple)  how much of that per-crop excursion is FINE undulation (24-96px,
     waviness.ripple_rms), on the same crops and the same scale  ->  feeds the model's
     second geometry channel. Without it the total is degenerate: one frame-wide arc and
     ten small waves score the same rms, and the model draws the arc because it is cheaper.
  4. RESPONSE  waviness of GENERATED crops vs lambda  ->  what lambda delivers that, for
     the older embedding-space extrapolation mechanism (ph_control.ph_to_lambda).

Re-run this after training a new checkpoint; nothing else needs to change.

    python3 calibrate_ph.py        # defaults to config.CHECKPOINT_PATH
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

import math

import ph_control
import train as T
from config import DEVICE, PH_MAX, PH_MIN, CHECKPOINT_PATH
from img2img import edit_image, load_and_preprocess_image
from model import from_state_dict
from waviness import waviness

DATA_DIR = "data/cropped/cropped_output"


def measure_real(data_dir, min_w=128, min_h=32):
    """Waviness of every readable real crop, grouped by its pH folder - whole-image scale.

    This is what ph_warp.py's geometric warp needs: it compares its target against
    waviness() measured on the whole reference image it's about to edit. For the scale
    the model's own native waviness conditioning was actually trained on, see
    measure_real_native instead - the two are NOT interchangeable, see ph_control.py's
    _DEFAULTS for why.
    """
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


def measure_real_native(data_dir, min_w=128, min_h=32, crops_per_image=8, seed=42):
    """Waviness of real crops measured exactly as training measures it - through train.py's
    own dynamic_collate_fn (frame fitted to a TRAIN_SIZES aspect ratio, flips, jitter).

    This is the scale the model's native waviness conditioning is trained on, and it is not
    the same as measure_real's whole-image scale, so the two fits are kept separate - see
    ph_control.py's _DEFAULTS. Returns (waviness, ripple) dicts: the geometry request has two
    halves and both are measured here, on the same crops and the same scale.

    The warp augmentation is switched OFF here. It exists to populate the conditioning axis
    during training, but this function is fitting the PHYSICS - how wavy a real filament at
    a given pH actually is - and that has to come from real geometry only. Leaving it on
    would fold a random synthetic displacement into the pH->waviness law itself.
    """
    saved_prob, T.WARP_AUG_PROB = T.WARP_AUG_PROB, 0.0
    T.set_seed(seed)  # dynamic_collate_fn draws from both random's and torch's global RNG
    try:
        per_ph, per_ph_ripple = {}, {}
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
                for _ in range(crops_per_image):
                    _, _, _, wav, ripple = T.dynamic_collate_fn(
                        [(ref.squeeze(0).cpu(), torch.tensor(ph))])
                    value = wav.item()
                    if not math.isnan(value):
                        per_ph.setdefault(ph, []).append(value)
                    ripple_value = ripple.item()
                    if not math.isnan(ripple_value):
                        per_ph_ripple.setdefault(ph, []).append(ripple_value)
        return per_ph, per_ph_ripple
    finally:
        T.WARP_AUG_PROB = saved_prob


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


def _fit_physics(per_ph, label):
    """Fit the pH->waviness law over a measure_real*-style {ph: [values]} dict.

    The FUNCTIONAL FORM is selected from the data by held-out extrapolation error rather
    than assumed - see ph_control.fit_waviness_law. Returns (law, diagnostics, slope,
    intercept, r2), where the last three are the plain linear fit, kept so the written
    calibration stays readable by older code that expects slope/intercept.
    """
    flat_ph = np.concatenate([[ph] * len(v) for ph, v in sorted(per_ph.items())])
    flat_w = np.concatenate([v for _, v in sorted(per_ph.items())])
    law, diagnostics = ph_control.fit_waviness_law(flat_ph, flat_w)
    print(f"  candidate laws for {label} (selection is by hold-out extrapolation, NOT R^2 - "
          f"a quadratic fits in-sample as well as a line and extrapolates disastrously):")
    print(f"    {'form':12s} {'R^2':>9} {'hold-out err':>13}")
    for form, d in diagnostics.items():
        mark = "  <- selected" if form == law["form"] else ""
        print(f"    {form:12s} {d['r2']:9.4f} {d['holdout_error']:13.3f}{mark}")
    lin = diagnostics["linear"]
    return law, diagnostics, float(lin["coeffs"][0]), float(lin["coeffs"][1]), float(lin["r2"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.5, 4.0])
    ap.add_argument("--per_bucket", type=int, default=1)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=ph_control.CALIBRATION_PATH)
    args = ap.parse_args()

    print("[1/4] measuring whole-image waviness of the real crops (feeds ph_warp.py's "
          "geometric warp)")
    per_ph = measure_real(DATA_DIR)
    print(f"  {'pH':>5} {'n':>4} {'mean rms_dev':>13}")
    for ph in sorted(per_ph):
        vals = np.array(per_ph[ph])
        print(f"  {ph:5.1f} {len(vals):4d} {vals.mean():13.2f}")
    law, law_diag, slope, intercept, r2 = _fit_physics(per_ph, "whole-image waviness")
    print(f"  selected law: {law['form']} {np.round(law['coeffs'], 4).tolist()}")

    print("\n[2/4] measuring per-crop waviness the same way training sees it (feeds "
          "native waviness conditioning)")
    per_ph_native, per_ph_ripple = measure_real_native(DATA_DIR)
    print(f"  {'pH':>5} {'n':>4} {'mean rms_dev':>13}")
    for ph in sorted(per_ph_native):
        vals = np.array(per_ph_native[ph])
        print(f"  {ph:5.1f} {len(vals):4d} {vals.mean():13.2f}")
    (native_law, native_diag, native_slope, native_intercept,
     native_r2) = _fit_physics(per_ph_native, "per-crop waviness")
    print(f"  selected law: {native_law['form']} {np.round(native_law['coeffs'], 4).tolist()}")
    print(f"  (R^2 is much lower than [1/4]'s: an individual small crop is noisy around the "
          f"pH trend the way any small random window is)")

    print("\n[2b/4] fine-undulation RIPPLE rms vs pH (the other half of the geometry "
          "request - waviness says how far the centreline strays in total, ripple how much "
          "of that is fine wiggle rather than one long arc)")
    print(f"  {'pH':>5} {'n':>4} {'median ripple':>14} {'share of total':>15}")
    for ph in sorted(per_ph_ripple):
        vals = np.array(per_ph_ripple[ph])
        tot = np.median(per_ph_native.get(ph, [float("nan")]))
        print(f"  {ph:5.1f} {len(vals):4d} {np.median(vals):13.2f}px {np.median(vals) / tot:15.2f}")
    ripple_law, ripple_diag, *_ = _fit_physics(per_ph_ripple, "ripple rms")
    print(f"  selected law: {ripple_law['form']} {np.round(ripple_law['coeffs'], 4).tolist()}")
    print("  extrapolated: " + "  ".join(
        f"pH{p}:{ph_control.eval_law(ripple_law, p):.1f}px" for p in (8.8, 10.8, 12.8)))

    print("\n[3/4] measuring the generator's response to lambda")
    model = from_state_dict(torch.load(args.checkpoint, map_location=DEVICE), DEVICE)
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
        "waviness_law": law,
        "native_waviness_law": native_law,
        "ripple_law": ripple_law,
        "ripple_law_diagnostics": ripple_diag,
        "waviness_law_diagnostics": law_diag,
        "native_waviness_law_diagnostics": native_diag,
        # the plain linear fit is still written so an older checkout can read this file
        "waviness_slope": slope,
        "waviness_intercept": intercept,
        "waviness_r2": r2,
        "native_waviness_slope": native_slope,
        "native_waviness_intercept": native_intercept,
        "native_waviness_r2": native_r2,
        "lambda_per_px": lambda_per_px,
        "max_lambda": float(max(args.lambdas)),
        "response_lambdas": [float(v) for v in lam_x],
        "response_waviness": [float(v) for v in lam_y],
        "response_monotonic": monotonic,
    }
    with open(args.out, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"\n[4/4] wrote {args.out}")
    print(f"  lambda_per_px = {lambda_per_px:.3f}   response monotonic: {monotonic}")
    if not monotonic:
        print("  WARNING: waviness did not rise monotonically with lambda - the "
              "extrapolation is unreliable past the point where it turns over.")

    def _physics_panel(ax, per_ph_dict, law_v, r2_v, title):
        flat_ph = np.concatenate([[ph] * len(v) for ph, v in sorted(per_ph_dict.items())])
        flat_w = np.concatenate([v for _, v in sorted(per_ph_dict.items())])
        phs = sorted(per_ph_dict)
        means = [np.mean(per_ph_dict[ph]) for ph in phs]
        ax.scatter(flat_ph + np.random.uniform(-.04, .04, len(flat_ph)), flat_w,
                  s=7, alpha=.2, color="#0f8177", linewidths=0)
        ax.plot(phs, means, "o-", color="#0f8177", lw=2, label="bucket mean")
        grid = np.linspace(min(phs) - 2, max(phs) + 3, 100)
        ax.plot(grid, [ph_control.eval_law(law_v, float(g)) for g in grid], "--",
               color="#b06a12",
               label=f"{law_v['form']} fit (R²={r2_v:.2f}), extrapolated")
        ax.axvspan(PH_MIN, PH_MAX, color="#0f8177", alpha=.07)
        ax.set_xlabel("pH"); ax.set_ylabel("rms centreline deviation (px)")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
        ax.grid(alpha=.25)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    _physics_panel(axes[0], per_ph, law, r2,
                  "physics: whole-image (-> ph_warp.py warp)")
    _physics_panel(axes[1], per_ph_native, native_law, native_r2,
                  "physics: per-crop (-> native conditioning)")

    axes[2].plot(lam_x, lam_y, "o-", color="#6f4aa0", lw=2)
    axes[2].set_xlabel("lambda (extrapolation strength)")
    axes[2].set_ylabel("rms centreline deviation (px)")
    axes[2].set_title("response: generated output", fontsize=10); axes[2].grid(alpha=.25)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/ph_calibration.png", dpi=130)
    print("  curve saved to outputs/ph_calibration.png")


_DEFAULT_GAIN = 0.35

if __name__ == "__main__":
    main()
