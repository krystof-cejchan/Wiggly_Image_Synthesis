# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Wiggly Image Synthesis trains a **pH-conditioned conditional flow matching (CFM) model** over grayscale microtubule microscopy crops, then uses it to synthesize how a microtubule image would look at a different pH. The core deliverable is `img2img.py`, which takes a real reference image plus its known pH and edits it toward a target pH. Higher pH makes the microtubule more curvy and wavy; lower pH makes it flatter. The dataset only covers pH 5.8-8.8, but the pipeline also supports requesting pH values outside that range (see "pH extrapolation" below) — this is not just a training detail, it's a deliberate second capability of `img2img.py`/`ph_warp.py`.

## Setup

```bash
pip install -r requirements.txt
```

A pretrained checkpoint is required for inference/eval — it is not committed to git. Download it per `checkpoints/download_trained_model.txt` and place it at `checkpoints/cfm_best_ema.pt`. All scripts (`train.py`'s save path, and every script's `--checkpoint`/`CHECKPOINT_PATH` default) agree on that one filename — there is no separate "v2" checkpoint name to worry about.

There is no test suite, linter, or type-checker configured in this repo. `test_img2img.py` and `test_ph_extrapolation.py` are headless diagnostic sweeps (they call the real pipeline and save comparison figures/plots for a human to eyeball), not assertion-based tests — nothing in them fails a build.

## Common commands

```bash
# Train from scratch (writes checkpoints/cfm_best_ema.pt + cfm_final_ema.pt,
# plus outputs/training_loss.png + .csv when training ends)
python3 train.py

# pH-edit a single reference image (main deliverable), in-range
python3 img2img.py --ref_image data/cropped/cropped_output/5.8/<file>.png \
    --source_pH 5.8 --target_pH 8.8 --num_steps 100 --strength 0.65 --contrastive_scale 3.0

# pH-edit OUTSIDE the trained range (5.8-8.8), either direction - the plain img2img.py CLI
# will run without erroring here but is NOT reliable for editing in either direction (see "pH
# extrapolation" below); use ph_warp.edit_to_pH instead, e.g. via the diagnostic script:
python3 test_ph_extrapolation.py --pH 3 4 5 6 7 8 9 10 11 12

# Unconditional-style CFG sampling at fixed 128x128 (sanity-checking the base model);
# pH_query may also be below the trained range (correctly extrapolates here via raw velocity
# extrapolation, since there's no reference image anchoring the trajectory) - above-range is
# still wrong here, and has no correct path through sample.py at all (needs a real image to warp)
python3 sample.py --pH 5.8 6.4 7.0 7.4 8.2 8.8 --num_steps 1000

# Sweep img2img over a grid of target pH / strength / scale (shells out to img2img.py)
python3 experiments.py

# Sweep contrastive_scale across one source image per pH bucket (headless, saves comparison
# figures under outputs_img2img/sweep/) - the recommended way to eyeball a new checkpoint
python3 test_img2img.py --contrastive_scales 1 3 5 --num_steps 50

# Re-fit the pH<->waviness/lambda calibration after training a new checkpoint;
# writes ph_calibration.json, which ph_control.py/ph_warp.py read at runtime
# (falls back to hardcoded defaults if this file is absent - it is not committed to git
# and does not exist until this is run at least once)
python3 calibrate_ph.py --checkpoint checkpoints/cfm_best_ema.pt

# Evaluation (require the checkpoint; DINOv2 variant needs internet for first torch.hub download)
python3 eval_fid.py             # unconditional FID vs one pH
python3 eval_fid_img2img.py     # FID for pH-to-pH translation
python3 eval_metrics_dino.py    # recommended: KID + DINOv2 backbone, native per-image resolution
```

There's no CLI entry point for a single unit test — verify inference-path changes by running `img2img.py` on one image and inspecting the saved plot/PNG in `outputs_img2img/`, or run `test_img2img.py` / `test_ph_extrapolation.py` for a broader sweep.

## Architecture

**Sampling (`sample.py`)**: unconditional-style CFG generation at fixed 128x128, defaults to the Heun solver, and uses CFG-rescale (rescale the CFG velocity toward the conditional branch's std, per Lin et al.) instead of a hard `clamp(v_cfg, -5, 5)` guardrail — avoids clipping artifacts from a fixed bound while still taming divergence from aggressive guidance. `pH_query` is routed through `ph_control.velocity_for_pH` directly (raw velocity extrapolation, no geometric fallback) — this is validated for the below-range direction here specifically because there's no reference image anchoring the trajectory (see "pH extrapolation" below); above-range is still wrong even in `sample.py`.

**Model (`model.py`)**: `ConditionalUNet` (~60.7M parameters) — a U-Net with FiLM-conditioned residual blocks (`FiLMResBlock`) and self-attention at the bottleneck (`SelfAttention2d`). Time `t` and `pH` are each mapped through Fourier features + MLP (`ScalarEmbedding`) and summed into one embedding that FiLM-modulates every res block. `pH = NaN` selects a learned null embedding — this is what makes classifier-free guidance (CFG) possible: the same network can run conditioned or unconditioned. `t_embed` and `pH_embed` deliberately use *different* Fourier frequency ranges (`t_embed`: 64 freqs up to `max_freq=10`; `pH_embed`: 16 freqs up to `max_freq=2`) — `t` is sampled densely and continuously during training so a high-frequency embedding is safe, but `pH` only takes 7 sparse, unevenly-spaced values (0.2-1.0 apart), and the old shared high-frequency config let the embedding oscillate arbitrarily between buckets with nothing in training to constrain it, which measurably broke interpolation to any untrained pH.

**Training objective (`train.py`)**: standard conditional flow matching / rectified flow — sample `x0 ~ N(0,1)`, interpolate `xt = (1-t)x0 + t*x1`, regress the model's output onto the constant velocity `target = x1 - x0` (MSE). `pH` is normalized to `[-1, 1]` via `PH_MIN`/`PH_MAX` in `config.py` (currently 5.8-8.8, the real pH range covered by the dataset), and jittered by `PH_JITTER_STD = 0.08` before normalization since the real pH buckets are sparse and unevenly spaced — this discourages the model from memorizing discrete conditioning values and is the actual interpolation signal (kept small relative to the tightest real gap, 0.2 pH, so it doesn't blur adjacent buckets together). During training, `pH` is randomly replaced with NaN at rate `CFG_DROPOUT` so the null embedding gets trained for CFG. Batch size is 16 with 4-step gradient accumulation (effective batch 64); the optimizer/scheduler/EMA all step once per *accumulated* batch, not per iteration — `CosineAnnealingLR`'s `T_max` is set in optimizer-steps (`ITERATIONS // ACCUMULATION_STEPS`), not raw iterations, or the LR never decays properly. An EMA copy of the weights (`ema_model`) is what actually gets checkpointed and evaluated; it uses a warmup ramp (`decay = min(0.9999, (1+n)/(10+n))`) rather than a flat 0.9999 from step 1 — with gradient accumulation cutting the number of EMA updates, a flat decay left a measurable fraction of the checkpoint as pure random initialization (produced washed-out, near-white samples). Validation (`val_collate_fn`) uses a fixed, **center**-cropped 128x128 (not `RandomCrop`) plus a fixed RNG seed, so `evaluate()` is fully deterministic — a random per-call crop made the val loss noisy enough to trip early stopping on crop variance rather than a real plateau. Early stopping (`PATIENCE = 10` evaluations, i.e. 5000 steps, `MIN_DELTA = 1e-5`) is driven by this flow-matching MSE, not any perceptual metric. When training ends (normally or via early stop), `plot_loss_history` saves `outputs/training_loss.png` (linear + log scale, train/val curves, best-checkpoint marker) and `outputs/training_loss.csv` (raw values) — check this after every run.

Training uses **dynamic aspect-ratio batching**: `dynamic_collate_fn` picks a random size from `TRAIN_SIZES = [(128,128), (64,256), (256,64), (48,384), (80,192)]` per batch and mirror-pads (`safe_mirror_pad`) + random-crops each sample to it, plus random horizontal/vertical flip and mild color jitter. `TRAIN_SIZES` must stay close to the data's actual aspect ratio (median crop is ~292x42, a thin strip): `safe_mirror_pad` reaches a target size by repeatedly *reflecting* the image, so a large square target (e.g. 384x384) turns one thin fiber into over a dozen stacked mirror copies, and the model learns to generate that tiled hall-of-mirrors instead of a single fiber — this was tried and produced visibly broken output.

**Data (`dataset.py`)**: `MicrotubuleDataset` expects `data/cropped/cropped_output/<pH>/*.png` — the pH value comes from the **folder name**, not the filename. Train/val split is deterministic and content-based: an MD5 hash of the filename prefix (before `_crop`) decides which split each image belongs to, so the same physical crop group never leaks across train/val even if the dataset is re-listed. `train.py` also uses a `WeightedRandomSampler` (inverse pH-bucket frequency) on top of this split, since bucket sizes range from 36 to 136 images — without it the largest bucket would dominate gradient signal ~4x over the smallest.

**Inference / editing (`img2img.py`)**: this is the actual product path, and it's more than a simple SDEdit:
- Partial denoising from the reference image: noise level is set by `--strength` (`t_start = 1 - strength`), then integrated forward to `t=1` via `--solver` (`heun` by default — a 2nd-order predictor-corrector; `euler` is available for comparison/speed). Heun costs 2x the model calls per step but measurably tracks the true ODE trajectory closer at the same step count (validated empirically: ~2.6x lower RMSE than Euler against a fine-grained reference trajectory at matched step count, ~1.9x lower even at matched *compute*).
- At every step it evaluates the model **twice** — once conditioned on `source_pH`, once on `target_pH` — and steers with `v_dir = v_source + scale * (v_target - v_source)`, where `scale` anneals from `--contrastive_scale` down to `1.0` over the trajectory. This is what actually pushes the image's morphology toward the target pH rather than just denoising it. Each per-step model call is routed through `ph_control.velocity_for_pH` rather than a plain conditioned call — inside `[PH_MIN, PH_MAX]` this is a single ordinary call; outside it, it transparently extrapolates (see below).
- Arbitrary image sizes are handled via a **sliding window** (`window_size`/`stride`, default 128/64) with Hann-window blending (`create_blending_mask`) to avoid seams, plus mirror-padding to a multiple of the stride. `create_blending_mask` trims the Hann window's zero endpoints (uses the interior of a length-`window_size+2` window) — a plain `hann_window(window_size)` is exactly 0 at the border, which froze the outermost row/column of the canvas at its initial noise value. Crops smaller than the window get mirror-tiled *before* editing, which produces a tiled/repeating result — `test_img2img.py` skips sources below `min_width`/`min_height` (128/32 by default) for this reason; about a third of the dataset is this small.
- `--repair_gaps` (off by default) runs `repair_fibre_gaps` before editing: it bridges short bright breaks in the fibre by repainting the anchor image itself. This exists because the img2img anchor pins the source at every step, so a bright gap in the source fibre stays pinned and the edit fades or breaks there; loosening `--strength` to compensate causes ghosting/tiling instead. Writes a before/after diagnostic to `outputs_img2img/repair_diagnostic.png` when it fires.
- `--strength` is the most sensitive knob: thin fiber structures break into disconnected segments at high strength; the README recommends 0.35-0.45 for that case even though the CLI default is 0.65.
- `--contrast_mode` controls how `--contrast` is applied post-generation: `linear` (default) scales pixels around the image mean, preserving overall brightness; `gamma` is the legacy `img**contrast` behavior, kept only to reproduce older runs (it always darkens and disproportionately crushes the dark end).

**pH extrapolation beyond the trained range (`ph_control.py`, `ph_warp.py`, `waviness.py`, `calibrate_ph.py`)**: the dataset and conditioning embedding only cover pH 5.8-8.8, but the tooling supports requesting any pH. Feeding an out-of-range value straight through the (periodic) Fourier pH embedding does not extrapolate — it aliases; measured on the trained embedding, pH 11.8 sits exactly as close to pH 8.8 as pH 5.8 does. Two different, independently-validated mechanisms are used instead, and they are **asymmetric by direction**:
- **Below pH 5.8 (straighter), during free generation only**: `ph_control.py` extrapolates in *velocity* space rather than embedding space — `v = v(anchor) + lambda * (v(hi) - v(lo))`, the same maneuver as CFG, with the CFG-rescale trick applied to keep the velocity norm from blowing up. This was measured to work **for unconditional generation from noise** (`sample.py`): a measured "orientation spread" metric, `waviness.orientation_spread`, falls monotonically as lambda pushes more acidic — 1.94 at pH 8.8 down to 0.97 at lambda=-2.0. It does **not** work for img2img editing — `ph_warp._straighten`'s docstring documents measuring waviness staying flat at ~6.5px across pH 3.0-7.3 during editing at strength 0.7, "no matter how hard the conditioning was pushed," because the reference image anchors the trajectory too strongly. `lambda` itself is **not** derived from the calibrated response curve below — `ph_control.ph_to_lambda` deliberately uses a simple "range-widths past the anchor" formula instead (lambda=1.0 means as far below 5.8 as 5.8 is below 8.8), because an earlier version that inverted the measured response curve turned out to be saturated by the img2img anchor and untrustworthy. Only `ph_calibration.json`'s `lambda_gain`/`max_lambda` scale this; the physics fit (`waviness_slope`/`intercept`) is not consulted for this direction.
- **Above pH 8.8 (wavier), in any context**: velocity extrapolation was tried and measured to go the *wrong* direction here regardless of free generation vs. editing (filaments got smoother/straighter, not wavier — because the pH8.8-minus-pH5.8 velocity direction also encodes texture/thickness/contrast, not just geometry; measured orientation spread *drops* from 1.94 to 1.56 as lambda pushes past +1.0, the opposite of wavier). Instead `ph_warp.py` conditions the model at the pH 8.8 anchor, then geometrically imposes extra undulation: it traces the filament centreline (`waviness.py`), synthesizes a smooth multi-mode transverse displacement sized off a physically-fit pH-vs-waviness law (amplitude linear, wavelength log-linear so it never goes negative), and warps just the filament (with a Gaussian envelope, extending the frame by synthesized background as needed) via `grid_sample`. `refine_texture` then runs a short low-strength img2img pass to re-render texture over the warped geometry without moving it.
- **`ph_warp.edit_to_pH(model, ref_image, source_pH, target_pH, ...)` is the single entry point that dispatches to the right mechanism for *editing* a real image, in either direction** — it's what `test_ph_extrapolation.py` and `calibrate_ph.py` call. For `target_pH < PH_MIN` its actual code path is: edit normally to the pH 5.8 anchor (`target_pH` is clamped before being passed to `edit_image`, so no velocity extrapolation happens in that call at all), then apply `ph_warp._straighten` — a geometric shear of the traced centreline — to the result; this is deliberate, not incidental, precisely because the velocity path doesn't work under an anchor (see above). **`img2img.py`'s CLI (`edit_image`) only ever performs the velocity-extrapolation mechanism** (via `ph_control.velocity_for_pH` inside the per-step contrastive loop, for both `source_pH` and `target_pH`) — it will *accept* an out-of-range `--source_pH`/`--target_pH` in either direction without erroring, but for actual editing this is wrong above `PH_MAX` and largely ineffective below `PH_MIN`. Always use `ph_warp.edit_to_pH` (or `test_ph_extrapolation.py`) instead of the plain CLI whenever editing to an out-of-range pH — not just above `PH_MAX`.
- `calibrate_ph.py` fits two curves: real-crop waviness vs. pH (`waviness.py`, Pearson r=0.84 over the dataset — this is the physics fit that `predicted_waviness()` uses to set the **above-8.8 geometric warp's target**), and generated-output waviness vs. extrapolation strength lambda (recorded mainly to sanity-check monotonicity, not currently inverted to drive the below-5.8 lambda — see above). It writes the fitted constants to `ph_calibration.json` (not committed to git), which `ph_control.py`/`ph_warp.py` read at runtime, falling back to hardcoded defaults if the file is absent. Re-run it after training a new checkpoint — the physics fit doesn't change, but it's keyed to a checkpoint-specific response measurement, and the fallback defaults are for a different model entirely.

**Evaluation (`eval_*.py`, `dino_features.py`)**: FID/KID against ImageNet-Inception is known to be unreliable here because per-pH sample counts are tiny (36-136 images) and Inception features are domain-mismatched for microscopy. The eval scripts encode specific fixes for this:
  - Use `feature=64` for `FrechetInceptionDistance` instead of 2048, since covariance from ~40 samples is otherwise near-singular.
  - Treat KID as the primary metric (unbiased at small n); FID is orientational only.
  - Every real/fake image is evaluated at its own **native resolution**, one at a time — not squashed into a fixed square via `train.py`'s `val_collate_fn`. Real crops are thin, wildly-sized strips; forcing them into a fixed square required so much mirror-padding that most of a "real" eval image was synthetic reflected content, while generated/edited images had no such artifact, creating a systematic asymmetry that inflated the measured distance. `MIN_SIDE`-style filtering drops crops too small to evaluate without being mostly mirror-hallucination.
  - `eval_metrics_dino.py` is the recommended evaluator: it swaps in a DINOv2 ViT-S/14 backbone (`dino_features.py`, self-supervised, transfers better to out-of-distribution/microscopy images than ImageNet-supervised Inception). `dino_features.py` pins `torch.hub.load` to a specific commit (`facebookresearch/dinov2:81b2b64`) rather than `main` — a later commit on `main` added Python 3.10+ only syntax (`float | None`) that crashes on Python 3.9.
  - When comparing real vs. generated images for FID/KID, keep `contrast` consistent between the two (the img2img `contrast` post-processing arg changes pixel statistics and must match, or default to 1.0, on both sides).

## Key invariants to preserve

- `PH_MIN`/`PH_MAX` in `config.py` define the normalization range for every script (`normalize_pH`) — training, sampling, editing, and eval must all agree on it. `ph_control.py` and `ph_warp.py` treat values outside this range as "extrapolate", never as something to feed through the embedding directly.
- Spatial dimensions fed to `ConditionalUNet` must be divisible by 16 (4 downsampling stages in `channel_mults`).
- `pH = NaN` is a sentinel for "unconditional", not a real missing-data marker — don't repurpose it.
- `ph_calibration.json` (produced by `calibrate_ph.py`) is checkpoint-specific — re-run calibration after retraining, or `ph_control.py`/`ph_warp.py` silently fall back to stale/default constants fit against a different model.
- All scripts default `--checkpoint`/`CHECKPOINT_PATH` to `checkpoints/cfm_best_ema.pt`, matching what `train.py` saves — keep any new script consistent with that rather than inventing another checkpoint filename.
