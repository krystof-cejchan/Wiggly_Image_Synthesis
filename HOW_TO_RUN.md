# How to run this project

A practical, task-by-task guide. For *why* the pipeline is built the way it is, see
`PROJECT_OVERVIEW.md` (conceptual explainer) or `CLAUDE.md` (dense, file-by-file reference).

## 1. Setup

```bash
pip install -r requirements.txt
```

Requires Python with a working PyTorch install (CUDA optional — everything falls back to CPU
automatically via `config.DEVICE`, just much slower: a single `img2img.py` edit at the default
100 steps takes seconds on a GPU and can take several minutes on CPU).

## 2. Get a trained checkpoint

Nothing in this repo works without one — download it per the instructions in
`checkpoints/download_trained_model.txt` and save it as:

```
checkpoints/cfm_best_ema.pt
```

Every script defaults `--checkpoint` (or `CHECKPOINT_PATH`) to that exact path, so if you save
it there you never need to pass the flag. If you keep multiple checkpoints around (e.g. before
and after a retrain), pass `--checkpoint path/to/other.pt` explicitly rather than renaming files
back and forth.

## 3. Smoke test: edit one image

```bash
python3 img2img.py \
    --ref_image data/cropped/cropped_output/5.8/20260219_005_Ch3_pos2_MES_pH5_frame0000_crop00.png \
    --source_pH 5.8 --target_pH 8.8 --num_steps 100 --strength 0.65 --contrastive_scale 3.0
```

This opens a comparison plot (original / edited / difference map) and saves the result to
`outputs_img2img/edited_pH_8.8_str_0.65.png`. If this runs and produces a plausible-looking
image, the checkpoint and environment are both working.

## 4. Editing a real image to a target pH — `img2img.py`

This is the main deliverable: take one real crop with a known pH, and synthesize how it would
look at a different pH.

```bash
python3 img2img.py --ref_image <path/to/crop.png> \
    --source_pH <known pH of the input> --target_pH <pH you want> \
    --strength 0.65 --contrastive_scale 3.0 --num_steps 100
```

### The knobs that matter, and why

- **`--strength`** (default `0.65`, range 0–1) — how much of the input image survives. It sets
  how far the reverse ODE integration starts from "pure noise" (`t_start = 1 - strength`): at
  `strength=0` you'd get the input back unchanged, at `strength=1` you'd get an unconditioned
  sample with no memory of the input at all. **For these thin fiber crops, high strength breaks
  the fiber into disconnected segments** — 0.35–0.45 is recommended over the 0.65 default if you
  see fragmentation.
- **`--contrastive_scale`** (default `3.0`) — how hard the edit pushes toward the target pH's
  morphology. At every step the model is run twice (once conditioned on `source_pH`, once on
  `target_pH`), and the actual step taken is `v_source + scale * (v_target - v_source)`. Higher
  values push harder toward the target's waviness; the scale anneals down to `1.0` by the end of
  the trajectory regardless of what you set, so it mainly affects early-trajectory structure.
- **`--num_steps`** (default `100`) — ODE integration steps. More is smoother/more accurate but
  slower; each step costs 2 model calls with the default Heun solver (see `--solver` below).
- **`--solver`** (`heun` default, or `euler`) — Heun is a 2nd-order predictor-corrector: it costs
  2x the model calls per step but tracks the true trajectory noticeably closer at the same step
  count (measured ~2.6x lower error than Euler at matched steps, and still ~1.9x lower even at
  matched *compute*, i.e. Euler at 2x the steps). Use `euler` only if you need the raw speed.
- **`--contrast`** (default `1.0`) and **`--contrast_mode`** (`linear` default, or `gamma`) —
  post-processing only, applied after generation. Flow matching can wash out deep blacks, so
  `--contrast` re-stretches pixel values; `1.0` is a no-op. `linear` mode scales pixels *around
  the image's own mean*, which preserves overall brightness while still restoring contrast.
  `gamma` mode (`pixel ** contrast`, the original implementation) always darkens the image and
  disproportionately crushes the dark end — it exists only to reproduce old runs, not as a
  general recommendation. **When comparing real vs. generated images (e.g. for FID/KID), use the
  same contrast on both sides** — this is a post-processing step, not something the real images
  went through, so mismatched contrast is a systematic bias, not a real quality difference.
- **`--repair_gaps`** (off by default) — if the input fiber has a short bright break in it
  (imaging artifact, not a real discontinuity), the edit will fade or break at that exact spot,
  because the input image anchors every step of the ODE and a bright gap stays pinned bright the
  whole way through. `--repair_gaps` repaints short interior breaks in the *input* before
  editing (not the output), and writes `outputs_img2img/repair_diagnostic.png` so you can check
  by eye whether the repair did something reasonable. It will not — and should not — bridge a
  fiber that genuinely ends there.
- **`--seed`** — set for reproducible output; omit for a fresh random draw each run.

## 5. Requesting a pH outside the trained range (5.8–8.8)

The dataset only covers 5.8–8.8, but `img2img.py` handles requests outside it automatically —
no different from any other edit:

```bash
python3 img2img.py --ref_image <crop.png> --source_pH 5.8 --target_pH 11.0 --strength 0.65
```

`main()` routes every edit through `edit_to_pH()` (in `img2img.py`), which picks the right
mechanism for wherever `target_pH` falls and prints a one-line summary of what it's doing before
it runs. You don't need to think about which direction you're asking for or call anything else —
this section is about *why* it works this way, not something you need to act on.

### Why this needs two different mechanisms internally

**In-range (5.8–8.8)**: ordinary conditioning, nothing special — `edit_to_pH` is a thin pass-through
to `edit_image` here.

**Outside the range, in either direction**: the obvious approach — extrapolate the model's own
conditioning past the trained range in *velocity* space (the same maneuver as classifier-free
guidance) — was tried first, and it's still what `sample.py` uses for free generation (step 6
below), where it's validated to work for the below-5.8 direction. But it does **not** carry over
to editing a real image: a reference image anchors the ODE trajectory strongly enough that
pushing the velocity field harder barely changes the output (measured waviness stayed flat at
~6.5px across pH 3.0–7.3 during editing, "no matter how hard the conditioning was pushed"). And
above 8.8 it doesn't work even for free generation — filaments got *smoother*, not wavier, because
the velocity direction from 5.8→8.8 encodes texture/thickness/contrast together, not just
geometry, and the model can't invent buckling it never saw.

So for editing, `edit_to_pH` does something different for out-of-range requests: it edits
normally to whichever trained anchor (5.8 or 8.8) is nearer, then explicitly reshapes the
*resulting image's pixels* — geometrically warping the traced fiber centreline wavier, or
geometrically shearing it straighter, sized against a physically-fit pH-vs-waviness law
(`waviness.py`, `ph_control.predicted_waviness`). This is genuine postprocessing, not the network
extrapolating on its own, and it's the only mechanism that was actually measured to work for
editing in both directions.

A geometric wavier-edit can grow the canvas taller to fit the extra undulation — `img2img.py`
prints a note when this happens, and the diff-map plot switches to a side-by-side view (no
elementwise diff) when the edited image is taller than the input, since a pixel-by-pixel
difference isn't meaningful against rows that don't exist in the original.

`test_ph_extrapolation.py --pH 3 4.4 5.8 7.3 8.8 10.3 11.8 13.0` runs a whole sweep and plots
waviness vs. pH if you want to sanity-check a checkpoint's extrapolation behavior in bulk rather
than one image at a time.

### Calibrating the extrapolation

The mapping from "requested pH" to "how hard to push" is fit from real measurements, not
guessed, and it's specific to one checkpoint's response curve:

```bash
python3 calibrate_ph.py --checkpoint checkpoints/cfm_best_ema.pt
```

This writes `ph_calibration.json` (fitted constants) plus `outputs/ph_calibration.png` (the two
curves it fit). **Re-run this every time you train a new checkpoint** — `ph_control.py`/
`ph_warp.py` read `ph_calibration.json` at runtime and silently fall back to hardcoded defaults
(fit against an older checkpoint) if it's missing or stale, which is easy to not notice.

## 6. Unconditional sampling — `sample.py`

Generates fresh samples from pure noise at a requested pH, with no input image. This is mainly a
sanity check that the base model learned the pH→morphology relationship at all — it is not the
product deliverable (`img2img.py` is).

```bash
python3 sample.py --pH 5.8 6.4 7.0 7.4 8.2 8.8 --num_samples 4 --num_steps 1000
```

Saves a grid PNG per requested pH to `outputs/sample_pH_<value>.png`. Out-of-range `--pH` values
extrapolate via `ph_control.py`'s velocity-space mechanism directly — unlike `img2img.py`, there's
no reference image here for that mechanism to fail against, and it's validated for the below-5.8
(straighter) direction. It's still wrong for above-8.8 (wavier) even here; that direction has no
correct path through `sample.py` at all, since the geometric fix in `ph_warp.py` requires editing
an existing image.

## 7. Training from scratch

```bash
python3 train.py
```

Runs up to 100,000 iterations with early stopping (stops after 10 evaluations, i.e. 5000 steps,
with no validation improvement). Writes:

- `checkpoints/cfm_best_ema.pt` — updated every time validation loss improves; this is the file
  every other script loads by default.
- `checkpoints/cfm_final_ema.pt` — written only if training runs to completion without
  early-stopping.
- `outputs/training_loss.png` and `outputs/training_loss.csv` — written when training ends,
  either way. **Always check the plot after a run**: the log-scale panel makes late-stage
  movement visible (losses plateau around 0.25–0.6, where a linear axis hides almost everything),
  and a dashed line marks which step's checkpoint was actually kept.

No arguments are exposed on the CLI — hyperparameters (`BATCH_SIZE`, `LR`, `PH_JITTER_STD`,
`TRAIN_SIZES`, etc.) are constants at the top of `train.py`; edit the file directly to change
them. **After training a new checkpoint, re-run `calibrate_ph.py`** (step 5 above) before relying
on pH extrapolation — the old calibration is fit against the old model's response curve.

## 8. Diagnostics and sweeps

These call the real pipeline (nothing is reimplemented) and save comparison figures for a human
to look at — they are not assertion-based tests, and nothing here fails a build.

```bash
# Sweep contrastive_scale across one source image per pH bucket - the fastest way to
# eyeball whether a freshly trained checkpoint behaves sensibly
python3 test_img2img.py --contrastive_scales 1 3 5 --num_steps 50

# Check that requesting an out-of-range pH does something monotonic and sensible
# (writes a waviness-vs-pH plot; the whole point is checking this curve is monotonic)
python3 test_ph_extrapolation.py --pH 3 4.4 5.8 7.3 8.8 10.3 11.8 13.0

# Sweep target pH / strength / contrastive_scale over one fixed source image
python3 experiments.py
```

## 9. Evaluation (FID / KID)

```bash
python3 eval_fid.py             # unconditional FID vs one pH (sample.py output vs real images)
python3 eval_fid_img2img.py     # FID for pH-to-pH translation (img2img.py output vs real images)
python3 eval_metrics_dino.py    # recommended: KID (primary) + FID (secondary), DINOv2 backbone
```

Read the printed KID number as primary and FID as directional-only — with only 36–136 real
images per pH bucket, FID is a biased estimator and not trustworthy as an absolute number;
KID is unbiased at small sample sizes and is what these scripts treat as authoritative.
`eval_metrics_dino.py` additionally swaps the feature extractor for a self-supervised DINOv2
backbone (better domain transfer to microscopy than an ImageNet-supervised one) — its first run
needs internet access to download the backbone weights via `torch.hub`.

## Troubleshooting

- **"Checkpoint not found"** — see step 2. Every script's default path is
  `checkpoints/cfm_best_ema.pt`; if you're using a different file, pass `--checkpoint` explicitly.
- **Output looks like a tiled/repeating pattern, not one fiber** — the input crop is smaller than
  the sliding-window size (128px by default) in one or both dimensions; `img2img.py`
  mirror-pads it up before editing, and a small enough crop mostly generates its own mirrored
  reflection. Check the crop's actual pixel dimensions.
- **Fiber breaks into disconnected pieces** — lower `--strength` (try 0.35–0.45); see the
  `--strength` explanation in step 4.
- **A gap/hole in the source fiber doesn't get filled in by the edit** — try `--repair_gaps`
  (step 4); this is a known limitation of anchoring every ODE step to the input image.
- **Saved/displayed image is taller than the input after an out-of-range edit** — expected; see
  step 5. A wavier geometric edit can need more vertical room than the original crop had, so the
  canvas gets extended (a printed note says by how much). The comparison plot drops the diff-map
  panel in this case, since an elementwise diff against rows that don't exist in the original
  isn't meaningful.
- **Achieved waviness (printed after an out-of-range edit) is noticeably off from the target** —
  the geometric warp re-measures what it actually achieved and retries up to 3 times, but a very
  tight/thin crop may not have room to reach a large target even after extending the canvas (see
  `fit_scale` in the returned info dict, or `ph_warp.warp_filament`'s docstring); this is a real
  physical limit of the crop, not a bug.
- **Training produces near-white / washed-out samples** — this specific failure mode was caused
  in the past by an EMA bug interacting with gradient accumulation (see `CLAUDE.md`'s training
  section); if you see it again after modifying `train.py`, check that the EMA update still runs
  once per *optimizer* step at a decay that's actually converged by the time training ends, not
  once per iteration at a naively fixed decay.
- **DINOv2 backbone fails to load** (`TypeError: unsupported operand type(s) for |`) — you're on
  Python < 3.10 and something bypassed the commit pin in `dino_features.py`; it should already
  load `facebookresearch/dinov2:81b2b64` rather than `main`. If the local hub cache has a stale
  copy from `main`, delete `~/.cache/torch/hub/facebookresearch_dinov2_main` and re-run.
