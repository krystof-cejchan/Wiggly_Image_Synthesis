Wiggly Image Synthesis
======================

Microtubules (protein fibers) change shape with the pH of their environment: lower pH keeps
them flat and straight, higher pH makes them buckle into wavy, curved shapes. This project
trains a pH-conditioned generative model on real microscopy crops of microtubules, then uses it
to answer: *given a real photo of this fiber at a known pH, what would it look like at a
different pH?*

The main deliverable is `img2img.py`: feed it one real microscopy crop and its known pH, ask for
a target pH, and it edits that specific image toward the target pH's characteristic waviness —
not a generic new sample. It uses a global ODE integrator combined with a sliding-window
approach for the vector field, so it handles arbitrarily large and wide input images. It also
supports requesting pH values *outside* the 5.8–8.8 range the model was trained on — see below.

For the full command reference (training, evaluation, diagnostics, pH extrapolation) see
**`HOW_TO_RUN.md`**. For a deeper explanation of how and why the pipeline is built this way, see
**`PROJECT_OVERVIEW.md`**.

## Setup

```bash
pip install -r requirements.txt
```

Before running, download the trained model per `checkpoints/download_trained_model.txt` and save
it as `checkpoints/cfm_best_ema.pt` — every script defaults to that path.

## Quick start

```bash
python3 img2img.py \
    --ref_image data/cropped/cropped_output/5.8/20260219_005_Ch3_pos2_MES_pH5_frame0000_crop00.png \
    --source_pH 5.8 --target_pH 8.8 --num_steps 100 --strength 0.65 --contrastive_scale 3.0
```

This opens a comparison plot (original / edited / difference map) and saves the result under
`outputs_img2img/`.

## Important arguments

```
--strength (default: 0.65)
    Denoising strength [0.0, 1.0] - how much of the input image survives the edit. For these
    thin fiber structures, lower values (0.35-0.45) are recommended: at the default, or higher,
    the fiber can break into disconnected segments.

--contrastive_scale (default: 3.0)
    How aggressively the target pH's morphology (curviness) is pushed onto the edit.

--num_steps (default: 100)
    ODE integration steps. Higher = smoother/more accurate, but slower.

--solver (default: heun)
    "heun" (2nd-order predictor-corrector, 2x model calls per step but noticeably lower
    integration error) or "euler" (1st-order, cheaper). Heun is recommended unless you need
    the raw speed.

--contrast (default: 1.0) / --contrast_mode (default: linear)
    Post-processing histogram adjustment, applied after generation - flow matching can wash out
    deep blacks. "linear" rescales pixels around the image's own mean (preserves brightness);
    "gamma" is the original img**contrast behavior (always darkens; kept only to reproduce older
    runs). Keep this consistent between real and generated images when comparing them - it's a
    post-processing step the real images never went through, so mismatched contrast is a
    systematic bias, not a real quality difference.

--repair_gaps (off by default)
    Bridges short bright breaks in the input fiber before editing. Off by default because it
    modifies the input image; writes a before/after diagnostic when it fires.
```

## Requesting pH outside the trained range (5.8–8.8)

The dataset only covers 5.8–8.8, but `img2img.py` handles requests outside it automatically -
just pass a `--source_pH`/`--target_pH` beyond that range, same as any other edit:

```bash
python3 img2img.py --ref_image <crop.png> --source_pH 5.8 --target_pH 11.0 --strength 0.65
```

Under the hood this edits to the nearer trained anchor (5.8 or 8.8) first, then geometrically
reshapes the result toward the requested pH's physically-extrapolated waviness - a real velocity
extrapolation through the network alone was tried and doesn't work once a reference image
anchors the edit (see `HOW_TO_RUN.md` §5 / `PROJECT_OVERVIEW.md` §6 for why). You don't need to
know that to use it; the CLI dispatches to the correct mechanism either way and prints a
one-line summary of what it's doing.

## Evaluation

```bash
python3 eval_metrics_dino.py    # recommended: KID (primary) + FID (secondary), DINOv2 backbone
```

See `HOW_TO_RUN.md` §9 for the other evaluation scripts and why KID, not FID, is the number to
trust here.
