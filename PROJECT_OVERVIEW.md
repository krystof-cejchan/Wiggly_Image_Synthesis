# Wiggly Image Synthesis — Project Overview

*A technical explainer of what this project does and how it works, for presentation purposes.
For copy-pasteable commands, see `HOW_TO_RUN.md`; for a denser file-by-file reference (aimed at
someone editing the code), see `CLAUDE.md`.*

## 1. What problem this solves

Microtubules (protein fibers) change shape with the pH of their environment: at low pH they lie flat and straight, at high pH they buckle into wavy, curved shapes. This project trains a generative model on real microscopy images of microtubules photographed at several known pH values, and uses it to answer the question: **"given a real photo of a microtubule at pH A, what would this exact microtubule look like at pH B?"**

The deliverable is `img2img.py`: feed it one real microscopy crop and its known pH, ask for a target pH, and it outputs a synthesized image of that same fiber morphed toward the target pH's characteristic waviness — not a generic sample, an edit of the specific input image.

## 2. The data

- **Source**: grayscale microscopy crops of microtubule fibers, stored under `data/cropped/cropped_output/<pH>/*.png`. The pH of each image is given by its **folder name**, not anything in the filename.
- **Coverage**: 7 pH buckets spanning **5.8 – 8.8**, with the following counts:

  | pH  | 5.8 | 6.4 | 6.8 | 7.2 | 7.4 | 7.8 | 8.8 |
  |-----|-----|-----|-----|-----|-----|-----|-----|
  | n   | 41  | 42  | 136 | 66  | 54  | 36  | 41  |

  416 crops total. The buckets are unevenly spaced (gaps of 0.2 to 1.0 pH) and unevenly sized (36 to 136 images) — both facts directly shape choices made in training (see §4).
- **Shape**: the crops are thin, non-square strips (tight bounding boxes around one fiber), not natural square photos. This is why training uses dynamic, non-square crop sizes rather than a fixed square resolution.
- **Train/validation split**: deterministic and content-based. Each filename is truncated to its prefix before `_crop`, MD5-hashed, and the hash decides train vs. validation (80/20). This guarantees that multiple crops taken from the *same* source photograph never end up split across train and validation — a hash of the raw filename would leak information about the same underlying fiber into both sets.

## 3. The model: conditional flow matching over a U-Net

### 3.1 What "flow matching" means here

The generative approach is **conditional flow matching / rectified flow**, a diffusion-model alternative. The model doesn't learn to predict noise directly; it learns a **velocity field**. Conceptually:

- Start from pure Gaussian noise `x0`.
- The real image is `x1`.
- Training draws a random point in time `t ∈ [0,1]` and linearly interpolates `xt = (1-t)·x0 + t·x1` — a straight-line path from noise to the real image.
- The network is trained to predict the constant velocity of that straight line, `target = x1 - x0`, given `xt`, `t`, and the conditioning pH. The loss is plain MSE between predicted and true velocity.
- At generation time, integrating this predicted velocity field forward from `t=0` (noise) to `t=1` via an ODE solver reconstructs an image — following the network's learned velocity field rather than the fixed straight line.

This is simpler to train than score-based diffusion and, empirically for this project, integrates well with only 100–1000 steps depending on the use case.

### 3.2 Network architecture — `ConditionalUNet`

A U-Net, **≈60.7 million parameters**, built from:

- **5 encoder stages / 5 decoder stages**, channel widths `64 → 128 → 256 → 512 → 512` (i.e. `base_channels=64` with multipliers `(1, 2, 4, 8, 8)`). Only the first 4 stages downsample (stride-2 conv), so the spatial resolution shrinks by **16×** total before the bottleneck — this is why every image fed to the network must have height and width divisible by 16.
- **2 residual blocks per stage** (`num_res_blocks=2`) on both the encoder and decoder side — 20 residual blocks total in the down/up path, plus 2 more in the bottleneck (22 residual blocks overall).
- **One self-attention block**, at the bottleneck only (the coarsest resolution — 8×8 for a 128×128 input), 4 attention heads. Attention only at the bottleneck keeps compute manageable while still giving the network a global (non-local) view of the whole fiber's layout at the point in the network where that matters most.
- **Skip connections** between matching encoder/decoder resolutions, standard U-Net style.
- Every residual block is **FiLM-conditioned** (`FiLMResBlock`): instead of just doing a convolution, each block computes a per-channel scale and shift from the conditioning embedding and applies it as `h·(1+scale) + shift` after GroupNorm. This is how time `t` and `pH` steer the network's behavior at every layer, not just at the input.

### 3.3 How time and pH are turned into a conditioning signal

Both `t` (the flow-matching time) and `pH` are scalars, and both go through the same recipe (`ScalarEmbedding`): **Fourier features** (a bank of sine/cosine functions at different frequencies — this lets a small MLP represent a scalar with the same kind of positional encoding used in transformers) followed by a 2-layer MLP into a 256-dim embedding. The two embeddings are projected and **summed**, then fed to every FiLM block in the network.

One deliberate asymmetry: `t` uses 64 frequencies up to `max_freq=10`, but `pH` uses only 16 frequencies up to `max_freq=2`. Reasoning: `t` is sampled densely and continuously over `[0,1]` during training, so a high-frequency embedding is safe. `pH` only ever takes one of 7 sparse, unevenly-spaced values — with the same high-frequency embedding as `t`, the Fourier encoding would be free to oscillate arbitrarily between pH buckets with nothing in the training signal to constrain what happens in between, which is exactly wrong for a model meant to interpolate smoothly between pH values.

**Classifier-free guidance (CFG) support**: `pH = NaN` is a reserved sentinel that selects a separate, *learned* null embedding instead of running the Fourier/MLP path. During training, the real pH label is randomly replaced with this null value 20% of the time (`CFG_DROPOUT = 0.2`), which teaches the same network to run both conditioned (knows the pH) and unconditioned (doesn't). This dual capability is what several inference-time tricks (guidance, contrastive steering, extrapolation) below are built on.

## 4. Training procedure (`train.py`)

| Hyperparameter | Value |
|---|---|
| Objective | Flow-matching MSE (predict `x1 - x0` from `xt`, `t`, `pH`) |
| Batch size | 16, with 4-step gradient accumulation → effective batch 64 |
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-4` |
| LR schedule | Cosine annealing, `T_max` = 25,000 **optimizer steps** (100,000 iterations / 4-step accumulation) — the scheduler steps once per accumulated batch, not per iteration, so `T_max` has to be given in those units or the LR never fully decays |
| Gradient clipping | Max norm 1.0 |
| Mixed precision | bfloat16 autocast |
| EMA | Decay ramps up to 0.9999 via `min(0.9999, (1+n)/(10+n))` rather than a flat 0.9999 from the first update, where `n` is the count of *optimizer* steps (not iterations) — with gradient accumulation cutting the number of EMA updates 4x, a flat decay left a measurable fraction of a checkpoint as pure random initialization (this produced visibly washed-out, near-white samples in practice before the ramp was added). The EMA weights (not the raw training weights) are what gets checkpointed and used for inference |
| CFG dropout | 20% of steps train with `pH = NaN` (null conditioning) |
| pH label jitter | Gaussian noise, std **0.08** pH units, added to the label every step (clamped back into range). Deliberately kept smaller than the tightest real gap between buckets (0.2 pH, between 7.2 and 7.4) — a wider std (0.15 was tried) bleeds adjacent buckets into each other and flattens the pH→waviness response in the densely-packed 5.8–7.8 region |
| Early stopping | Evaluate every 500 steps (deterministic: fixed noise generator *and* a center-crop, not random-crop, validation batch — a random crop made the val loss noisy enough to trip early stopping on crop variance alone); stop after **10** evaluations (5000 steps) with no improvement beyond `1e-5` |
| Loss curve | `outputs/training_loss.png` (linear + log scale, train/val, best-checkpoint marker) and `outputs/training_loss.csv` (raw values) are written whenever training ends, whether by early stop or completion |

Three choices are worth calling out specifically because they respond directly to properties of the small, imbalanced dataset described in §2:

- **Class-balanced sampling.** pH buckets range from 36 to 136 images — a ~4× imbalance. Training uses a `WeightedRandomSampler` with weights inversely proportional to each bucket's size, so every pH gets roughly equal gradient signal per step regardless of how many source images it has.
- **pH label jittering** (Gaussian noise, std 0.08, re-applied every training step) is the actual mechanism that teaches the network to **interpolate** between the 7 discrete pH values it was trained on. Without it, the network only ever sees 7 exact labels and has no signal that pH 6.0 should look like something *between* pH 5.8 and pH 6.4 — jittering forces "similar pH → similar output" during training. The std is kept deliberately small relative to the tightest real gap (0.2 pH, between 7.2 and 7.4): a wider value (0.15 was tried first) bleeds adjacent buckets into each other and flattens the pH→waviness response in the densely-packed 5.8–7.8 region rather than helping it.
- **Dynamic aspect-ratio batching.** Each training batch picks one of five sizes at random — `128×128, 64×256, 256×64, 48×384, 80×192` — mirror-pads undersized crops up to that size, then random-crops. This exists because the source images are non-square fiber strips (median ~292×42px); training exclusively on squares would either waste most of a strip crop or force awkward padding on every sample. The sizes themselves have to stay close to that strip aspect ratio for a specific reason: mirror-padding reaches a target size by *repeatedly reflecting* the image, so a large square target (384×384 was tried) turns one ~42px-tall fiber into over a dozen stacked mirror copies before it's even cropped — the model then learns to generate that tiled hall-of-mirrors pattern instead of a single fiber, which is a real failure mode that was observed in practice, not a hypothetical one. Augmentation on top of the crop: random horizontal/vertical flip (50% each) and mild color jitter (±10% brightness/contrast). Validation uses a **fixed, center**-cropped 128×128 (not a random crop) and a fixed random seed, so the validation loss used for early-stopping is a stable, comparable number across evaluations rather than being re-randomized every time.

## 5. Generating images

There are two distinct generation modes in the codebase, and they share the same trained network and the same ODE-integration machinery, but serve different purposes.

### 5.1 Unconditional-style sampling (`sample.py`) — sanity checks

Generates a sample from pure noise at a requested pH, using classifier-free guidance: at every integration step the model is run twice (once conditioned on the target pH, once on the null embedding) and the two velocities are combined as `v_uncond + cfg_scale·(v_cond − v_uncond)`. A "CFG-rescale" step (Lin et al.) renormalizes the guided velocity back to the conditional branch's own statistics, which avoids the ringing/oversaturation artifacts that guidance is prone to, without resorting to a hard value clamp. This mode exists mainly to sanity-check that the base model has learned the pH → morphology relationship at all; it is not the deliverable.

### 5.2 Image-to-image pH editing (`img2img.py`) — the actual product

This is more than SDEdit-style partial denoising; three mechanisms combine:

1. **Partial denoising from a real anchor.** Instead of starting from pure noise, the reference image is noised only partway: `--strength` (default 0.65) sets how much noise is mixed in (`t_start = 1 − strength`), and the ODE integration then runs forward from `t_start` to `t=1`. Low strength keeps the output close to the input structure; high strength gives the model more freedom to reshape it, but thin fiber structures can break into disconnected segments above ~0.45–0.65 strength — the README specifically recommends 0.35–0.45 for that reason, even though the CLI default is higher.
2. **Dual-conditioning contrastive steering.** At *every* integration step, the model is evaluated twice — once conditioned on the known `source_pH`, once on the requested `target_pH` — and the step actually taken follows `v_source + scale·(v_target − v_source)`, where `scale` starts at `--contrastive_scale` (default 3.0) and anneals down to 1.0 over the trajectory. This is the mechanism that actually pushes the specific input fiber's morphology toward the target pH's waviness, rather than just denoising it back to something plausible at the source pH.
3. **ODE solver.** Default is `heun`, a 2nd-order predictor-corrector that evaluates the velocity field twice per step but tracks the true trajectory more closely than 1st-order Euler at the same step count — validated empirically during development. Euler is available for faster/cheaper runs.

**Handling arbitrary image sizes.** The network requires inputs divisible by 16, and real crops are arbitrary sizes, often much larger than the 192–384px width range seen in training. `img2img.py` frames the canvas inside that trained band (`config.TRAIN_SIZES`): height is grown with synthesised background to at least `TRAIN_MIN_H`, width mirror-padded to the next multiple of 16 (or up to `TRAIN_MIN_W`), and the model is then run over the whole crop in one window when it fits, or through a **sliding window** of `TRAIN_MAX_W` (384px) at 50% overlap when it does not, with overlapping windows blended together using a 2D Hann window (`create_blending_mask`) — this is what avoids visible seams between windows. Staying inside the trained band is not cosmetic: the window used to be a fixed 128px, and at that width the model produces no traceable fibre at all and ignores its waviness conditioning completely. (One implementation detail worth knowing: a plain Hann window is exactly zero at its own edges, which — before this was fixed — froze the outermost row/column of the canvas at its initial noise value; the blending mask now uses the *interior* of a slightly larger Hann window to keep every pixel's blend weight strictly positive.)

## 6. Interpolation vs. extrapolation — reaching pH outside 5.8–8.8

This is the more novel part of the project: the tooling doesn't just interpolate within the trained pH range, it also supports **requesting pH values the model never saw during training**, and does so via two different, independently-validated mechanisms depending on direction.

**Why you can't just plug an out-of-range pH into the network.** The pH embedding is Fourier-based, i.e. periodic. Measuring actual embedding distances confirms that feeding in an out-of-range pH does not extrapolate the trend — it *aliases*: pH 11.8 was measured to sit exactly as close to the pH 8.8 embedding as pH 5.8 does. Naively asking for "more alkaline than 8.8" could silently hand back something resembling the *acidic* end instead. So out-of-range requests are never fed through the embedding directly.

### 6.1 Below pH 5.8 (straighter): velocity-space extrapolation (`ph_control.py`)

The two trained endpoints (pH 5.8 and pH 8.8) give the network two velocity fields, `v_lo` and `v_hi`. Their difference, `v_hi − v_lo`, is treated as a direction meaning "become more alkaline / wavier" — pushing *against* it extrapolates past the acidic anchor:

```
v = v_lo − λ · (v_hi − v_lo)        (for λ > 0, going more acidic than pH 5.8)
```

This is the same mathematical move as classifier-free guidance, with the same rescaling trick applied to keep the velocity norm from inflating. It was validated empirically **for free generation from noise** (`sample.py`, no reference image involved): as λ increases, a measured "orientation spread" metric (how much the fiber's local angle varies — a proxy for waviness with no assumption about a single dominant fiber) falls monotonically, i.e. the output really does get straighter.

**This does not carry over to img2img editing.** With a real reference image partially denoised and anchoring the trajectory, pushing the velocity field harder was measured to barely move the output at all — `ph_warp.py`'s straightening function documents measured waviness staying flat at ~6.5px across pH 3.0–7.3 during editing at strength 0.7, "no matter how hard the conditioning was pushed." So for editing specifically, the below-5.8 direction needs the same kind of fallback as the above-8.8 direction below — see the note at the end of §6.2.

λ itself, notably, is **not** derived from the calibrated response curve described in §6.3 below — it uses a simpler "range-widths past the anchor" formula (λ=1 means as far below 5.8 as 5.8 is below 8.8). An earlier version tried inverting the measured response curve to pick λ and found it unreliable: under an img2img anchor the response saturates, so the natural distance-based formula was kept instead, specifically because it means something without depending on a (fragile) calibration measurement.

### 6.2 Above pH 8.8 (wavier): geometric warping (`ph_warp.py`, `waviness.py`)

The same velocity-extrapolation trick was tried in this direction too, and **measured to fail**: pushing past the pH 8.8 anchor produced filaments that got *smoother and more regular*, not wavier — because the `v_hi − v_lo` direction encodes the *entire* difference between the two ends (texture, fiber thickness, contrast — not just geometric waviness), and the model has no way to invent a buckling pattern it was never shown.

The fallback is geometric rather than learned: take the **texture** from the model (conditioned normally at the pH 8.8 anchor) and impose the **geometry** from a separately-measured physical law:

- `waviness.py` traces each fiber's centerline in a real crop and measures its RMS deviation from a straight line (a proxy for "how wavy is it"). Across the dataset this correlates with pH at **Pearson r = 0.84** (rising from ~3.9px RMS deviation at pH 5.8 to ~6.3–6.7px at pH 7.8–8.8) — a straight-line fit lets this be extrapolated past pH 8.8.
- `ph_warp.py` then synthesizes a smooth, multi-mode transverse displacement sized to hit that extrapolated target waviness, and applies it to the *image itself* via `grid_sample` — warping only the traced fiber (inside a Gaussian envelope centered on it, so the background doesn't drag along and pick up unnatural warp artifacts), extending the frame by synthesized background where the warp needs more room than the original crop provides.
- A short low-strength img2img refinement pass (`refine_texture`) then re-renders realistic texture over the newly warped geometry without disturbing the shape.

The same warp machinery also handles requests **below** pH 5.8 during editing, by damping (rather than adding) waviness — shearing the existing traced centerline down to a lower target. This is not an optional simplification; it's necessary, for the reason noted at the end of §6.1: velocity extrapolation doesn't move the output under an img2img anchor, so editing below 5.8 needs the same geometric treatment as editing above 8.8. `ph_warp.edit_to_pH()` is the single entry point that looks at the requested pH and dispatches to the right mechanism automatically, in **either** direction.

**`img2img.py`'s plain CLI only ever performs the velocity-extrapolation mechanism**, for source or target pH in either direction — it will run without erroring on an out-of-range request, but for actual editing this is wrong above 8.8 (produces the opposite of wavier) and largely ineffective below 5.8 (barely changes the output). The geometric-warp path — for *either* direction, when editing — is reached only through `ph_warp.edit_to_pH()` / `test_ph_extrapolation.py`, never the plain CLI. (`sample.py`, which has no reference image to anchor anything, is the one place the plain velocity-extrapolation mechanism is actually correct on its own — and only for the below-5.8 direction; above-8.8 is wrong there too, per §6.2.)

### 6.3 Calibration (`calibrate_ph.py`)

Two curves are measured and fit, but they feed **different** parts of the pipeline, not one combined mapping:

1. **Physics**: real-crop waviness vs. pH (the r=0.84 fit above) — measured from real images only, no model involved. This is what `predicted_waviness()` uses to set the target for the **above-8.8 geometric warp** (§6.2): it directly says "a real filament at this pH should have N pixels of RMS deviation," extrapolated past 8.8.
2. **Response**: generated-output waviness vs. extrapolation strength λ, measured by actually running the model at a sweep of λ values. This is checkpoint-specific and is currently used mainly to **sanity-check monotonicity** (does waviness rise smoothly as λ increases, with no reversal) — it is not inverted to drive the below-5.8 λ mapping (§6.1), which uses a simpler, uncalibrated distance formula instead because the inverted-response approach was found to saturate.

Both fitted curves are written to `ph_calibration.json`, read at runtime by `ph_control.py`/`ph_warp.py`, falling back to hardcoded defaults if the file is missing. Re-run this script after retraining: the physics fit (curve 1) won't change since it's model-independent, but the response curve (curve 2) is specific to one checkpoint, and the fallback defaults are fit against a different model entirely.

## 7. Post-processing

- **Contrast adjustment** (`--contrast`, `--contrast_mode`): Flow matching can wash out deep blacks; `linear` mode (default) rescales pixel values around the image's own mean, preserving overall brightness, while `gamma` mode (`img**contrast`, the original implementation) is kept only for reproducing older runs — it always darkens the image and disproportionately crushes the dark end.
- **Sliding-window Hann blending**: described in §5.2, applied to every image regardless of size.
- **Fiber gap repair** (`--repair_gaps`, off by default): a pre-processing step, not a post-processing one — it repaints short bright breaks in the source fiber *before* editing, because the img2img anchor pins the source image at every ODE step, so a bright gap in the input fiber stays pinned throughout and the output fades or breaks exactly there. Writes a before/after diagnostic image so the repair can be checked by eye.

## 8. Evaluation methodology

Standard FID/KID against an ImageNet-pretrained Inception network is known to be unreliable for this project, for two compounding reasons: per-pH sample counts are tiny (36–136 images), and Inception's features were learned on natural photographs, not microscopy. The evaluation scripts encode specific mitigations:

- `feature=64` instead of the usual 2048-dim Inception features for FID, since a covariance matrix estimated from ~40 samples is otherwise near-singular.
- **KID treated as the primary metric** (it's unbiased at small sample sizes; FID is used only as a secondary, orientational signal).
- The recommended evaluator, `eval_metrics_dino.py`, replaces the Inception backbone with **DINOv2 (ViT-S/14)** — a self-supervised backbone that transfers better to out-of-distribution image domains like microscopy than a supervised ImageNet classifier — and evaluates on the data's **native stripe aspect ratio (64×256)** rather than squashing crops into squares, with identical preprocessing applied to real and generated images. Default eval settings compare pH 5.8 → 8.8 translation at strength 0.40, contrastive scale 2.0, 250 steps, with `contrast` fixed to 1.0 on both real and generated images (any contrast post-processing changes pixel statistics and would bias the comparison if applied asymmetrically).

## 9. Key invariants / things that would break if changed casually

- `PH_MIN`/`PH_MAX` (5.8/8.8 in `config.py`) define the pH normalization range used consistently across training, sampling, editing, and evaluation.
- All spatial dimensions fed to the network must be divisible by 16 (4 downsampling stages).
- `pH = NaN` is reserved exclusively as the "unconditional" sentinel for CFG — never a stand-in for missing data.
- `ph_calibration.json` is tied to one specific checkpoint; retraining without recalibrating silently uses stale constants.

## 10. Honest limitations, in the project's own words

- The above-8.8 extrapolation is **an extrapolation of a fitted statistical trend**, not new physical evidence — it asserts that the pH-vs-waviness relationship measured over 5.8–8.8 continues past it. Whether real chemistry actually behaves that way past pH 8.8 is a question only new microscopy data could answer.
- The geometric warp adds contour length via a shear rather than conserving it the way a physically buckling filament would.
- **Extrapolation via the learned network alone (§6.1's velocity-space mechanism) only works for free generation, not editing.** Once a real reference image anchors the trajectory, that mechanism was measured to barely move the output regardless of direction — which is a real limit on how much the *network itself* can extrapolate, not a code bug. The geometric fallback (§6.2) exists for editing precisely because the learned approach hits this wall; it isn't just a convenience.
- FID/KID scores on this dataset should be read as directional/orientational signals given the small per-pH sample sizes, not as precise, statistically tight numbers.
