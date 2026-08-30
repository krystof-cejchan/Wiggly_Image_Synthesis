import os
import csv
import math
import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")  # training usually runs headless / over ssh - never try to open a window
import matplotlib.pyplot as plt

from config import (PH_MIN, PH_MAX, DEVICE, TRAIN_SIZES,
                    CHECKPOINT_PATH, COND_CHECKPOINT_PATH, FINAL_CHECKPOINT_PATH)
import torchvision.transforms.v2 as T
from model import ConditionalUNet
from dataset import MicrotubuleDataset
from PIL import Image
from waviness import waviness as measure_waviness
from framing import fit_frame, mirror_pad_width, pad_height_background
from ph_warp import bounded_displacement, warp_filament

# hyperparameters
DATA_DIR = "data/cropped/cropped_output"
# Physical batch raised 16->64 with ACCUMULATION_STEPS cut 4->1 on the assumption of a GPU
# with room for it: the EFFECTIVE batch (64) is unchanged, so this is purely "do the same
# 4 micro-batches as one bigger parallel call" rather than 4 small serialized ones - no
# change to what the optimizer sees per step. ITERATIONS and EVAL_INTERVAL are divided by
# the same factor (4) to hold the total optimizer/EMA step count, the LR schedule length
# (T_max below), and the samples-seen-between-evals ratio all exactly where they were -
# PATIENCE counts evals, not samples, and each eval still represents the same amount of
# data as before this rescaling - see PATIENCE itself below for why its VALUE changed
# afterwards, for an unrelated reason. If you had a training run going under old numbers,
# this needs a restart to take effect - editing the file doesn't touch a process already
# running with old values loaded.
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
LR = 1e-4
ITERATIONS = 50_000  # doubled from 25,000: the last run plateaued at step 19,625 (best) and
                     # exhausted PATIENCE at 23,375/25,000 - by then T_max (below, tied to
                     # ITERATIONS) had the cosine LR nearly fully decayed, so it had no room
                     # left to keep improving even if the plateau wasn't final. Doubling gives
                     # the schedule twice the room before LR bottoms out. EVAL_INTERVAL and
                     # PATIENCE are left as absolute eval counts (not rescaled with this) -
                     # they're about how many evals of no-improvement to tolerate before
                     # calling it converged, which doesn't need to track total budget.
CFG_DROPOUT = 0.2
WAVINESS_CFG_DROPOUT = 0.2  # independent of CFG_DROPOUT (own random draw per sample) so the
                            # model gets real practice at "geometry unconditioned, texture
                            # conditioned" and the reverse, not just both-or-neither - this is
                            # what makes guiding each axis separately at inference meaningful
                            # later instead of the two channels always moving together.
PH_JITTER_STD = 0.08  # pH buckets are unevenly spaced (0.2-1.0 apart) - jittering the label
                      # each step teaches the model that nearby pH values should look similar,
                      # the interpolation signal the discrete buckets alone don't provide. Keep
                      # this small relative to the TIGHTEST real gap (7.2->7.4 is only 0.2): a
                      # wider std (previously 0.15) bleeds across adjacent buckets in the densely
                      # packed 5.8-7.8 range and flattens the pH->waviness response there, while
                      # barely reaching into the one big outlier gap (7.8->8.8) anyway.
EMA_DECAY = 0.9999
EVAL_INTERVAL = 125  # scaled down from 500 with ITERATIONS, see the note above BATCH_SIZE
PATIENCE = 30        # val loss is noisy (fluctuates ~1e-2 while MIN_DELTA is 1e-5) - this
                     # comment predicted the exact failure that happened with the old value
                     # of 10: the waviness-conditioned retrain's first eval (step 125) was
                     # also its best, every one of the next 10 sat within noise of it, and
                     # training stopped at step 1375/25000 (5.5%) with the live train loss
                     # still visibly dropping. Every FiLMResBlock's emb_proj is zero-init
                     # (see model.py), so EVERY conditioning signal - not just the new
                     # waviness channel - starts with literally zero effect on the output
                     # and has to be woken up by gradient descent; with a third signal now
                     # sharing that same wake-up budget, 10 evals (80,000 samples) may
                     # simply not have been enough runway. 30 is a reasoned bound, not a
                     # validated one - it triples the runway without disabling early
                     # stopping outright.
MIN_DELTA = 1e-5
# Separate threshold for the conditioning gap, which lives on a completely different scale
# from the val loss: measured on trained checkpoints the gap is ~1e-4 where the loss is ~0.58,
# so MIN_DELTA (1e-5, a tenth of the whole gap) would treat pure jitter as improvement. The
# gap is deterministic (evaluate uses a fixed generator and identical xt for all three
# conditioning variants), so this only has to exceed float noise, not sampling noise.
COND_MIN_DELTA = 2e-6
SEED = 42
# Crop geometry (TRAIN_SIZES) now lives in config.py, imported above. It moved there
# because img2img.py has to frame its editing window inside the same band - it used to
# carry its own independent 128x128 window, which drifted out of this band when the sizes
# here changed and silently cost the waviness conditioning all of its effect at inference
# time. Heights stay in the real data's band on purpose; the old set asked for 128 and even
# 256px tall crops, which forced 1.5-6x VERTICAL mirror tiling on nearly every sample, and
# framing.py documents why that destroyed the waviness label (corr 0.94 with a pure tiling
# artefact) and taught the model to draw stacked fibres. Short crops are grown with
# synthesised background instead, so the taller entries cost nothing but give the warp
# augmentation below room to express a genuinely wavy fibre.

# On-the-fly waviness augmentation. The real data is severely unbalanced in the property we
# actually want to control: only 25 of 361 source images can yield a crop above 12px of
# waviness and only 7 can reach 16px, while pH 15.8 asks for ~14px and pH 20 for ~19px.
# Training on that alone leaves the conditioning axis empty exactly where it matters.
# So a random (NOT pH-derived) smooth displacement is applied to the fibre, and the label is
# then MEASURED from the result rather than assumed. This is deliberately not the same thing
# as pre-generating warped images labelled with a made-up pH: no synthetic pH-to-geometry
# association is taught anywhere - the pH label stays the sample's real pH, and the pH->
# waviness mapping still comes only from calibrate_ph.py's fit on real crops. The
# augmentation populates the waviness axis and nothing else. warp_filament moves only the
# fibre (Gaussian envelope), leaving background grain untouched, so texture stays real.
WARP_AUG_PROB = 0.5
WARP_AUG_MAX_WAVINESS = 26.0

# A frame can only physically show so much waviness before the fibre leaves it: roughly
# (H/2 - margin)/sqrt(2), i.e. ~12px at H=48, ~18px at H=64, ~24px at H=80, ~30px at H=96.
# Reaching the pH 15.8-20 targets therefore needs the taller frames - but padding a 28px
# source up to 96px means inventing more rows than the crop has real ones to donate grain
# from, which shows up as horizontal streaking. So batches are stratified by source height
# (HeightStratifiedBatchSampler) and the collate picks a frame height the batch can
# actually support: tall frames get built from genuinely tall fibres, not padded up from
# tiny ones.
MIN_SOURCE_HEIGHT = 24
TALL_MIN_HEIGHT = 48
TALL_BATCH_FRACTION = 0.4

def set_seed(seed):
    """Zajistí reprodukovatelnost napříč PyTorch i Pythonem."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_pH(pH):
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

def _measure(img4):
    """Waviness of the exact tensor the model will be trained on, or NaN if untraceable.

    Measured after every augmentation, on the final frame, so the label can never disagree
    with the pixels. NaN feeds the same "unconditioned" null path pH already uses.
    """
    value = measure_waviness(img4)
    return float(value) if value is not None else float("nan")


def _warp_augment(img4):
    """Randomly make the fibre wavier, so the conditioning axis is populated where the real
    data is empty. Returns the warped frame; the caller re-measures the label from it.

    The requested amount is drawn uniformly between the fibre's current waviness and
    WARP_AUG_MAX_WAVINESS, which spreads samples across the sparse upper range instead of
    piling them at the bottom where the real distribution already sits (real p50 = 3.3px).
    warp_filament(extend=False) rescales anything that would not fit the frame, so the
    fibre can never be pushed out of view.
    """
    current = measure_waviness(img4)
    if current is None or current >= WARP_AUG_MAX_WAVINESS:
        return img4
    target = random.uniform(current, WARP_AUG_MAX_WAVINESS)
    needed = math.sqrt(max(0.0, target ** 2 - current ** 2))
    if needed < 0.5:
        return img4
    displacement, _ = bounded_displacement(
        img4.shape[3], needed, random.uniform(120.0, 400.0), img4.device)
    warped, _ = warp_filament(img4, displacement, extend=False)
    return warped


def dynamic_collate_fn(batch):
    """Pick an aspect ratio the batch can support, then crop/pad every sample to it.

    The frame height is capped at ~2.2x the batch's 25th-percentile source height so no
    sample ends up mostly synthesised background. Paired with
    HeightStratifiedBatchSampler, which makes some batches entirely tall sources, this is
    what lets the tall frames (needed to express high waviness) exist at all without
    over-padding the short crops.
    """
    heights = [item[0].shape[1] for item in batch]
    cap = 2.2 * float(np.percentile(heights, 25))
    allowed = [s for s in TRAIN_SIZES if s[0] <= cap] or [min(TRAIN_SIZES)]
    target_h, target_w = random.choice(allowed)

    photometric = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    images, wavs = [], []
    for item in batch:
        img = fit_frame(item[0].unsqueeze(0), target_h, target_w)
        img = photometric(img)
        if random.random() < WARP_AUG_PROB:
            img = _warp_augment(img)
        images.append(img.squeeze(0))
        wavs.append(_measure(img))

    phs = [item[1] for item in batch]
    return torch.stack(images), torch.stack(phs), torch.tensor(wavs, dtype=torch.float32)


def val_collate_fn(batch):
    """Fixed 64x256 centre crop, no jitter and no warp augmentation.

    Everything here must be identical on every evaluate() call or the val loss picks up
    augmentation noise (~1e-2) that dwarfs MIN_DELTA (1e-5), which makes early stopping fire
    on noise instead of on a real plateau. The background padding draws random numbers, so
    it gets a fixed-seed generator rather than the global RNG.
    """
    target_h, target_w = 64, 256
    images, wavs = [], []
    for item in batch:
        img = item[0].unsqueeze(0)
        # Centre window, taken from the SOURCE - the same dead-code trap fit_frame had:
        # mirror_pad_width returns exactly target_w, so cropping after it could never move
        # the window and this "centre crop" was really a LEFT crop on every wider source.
        # Deterministic either way, so early stopping is unaffected in kind - but val losses
        # from before this fix are not comparable to ones after it, because they score
        # different pixels.
        if img.shape[3] > target_w:
            left = (img.shape[3] - target_w) // 2
            img = img[:, :, :, left:left + target_w]
        else:
            img = mirror_pad_width(img, target_w)
        h = img.shape[2]
        if h > target_h:
            top = (h - target_h) // 2
            img = img[:, :, top:top + target_h, :]
        elif h < target_h:
            gen = torch.Generator().manual_seed(1234)
            img = pad_height_background(img, target_h, generator=gen, jitter=0.0)
        images.append(img.squeeze(0))
        wavs.append(_measure(img))

    phs = [item[1] for item in batch]
    return torch.stack(images), torch.stack(phs), torch.tensor(wavs, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, dataloader, num_noise_samples=3):
    """Flow-matching MSE on the validation set, plus how much the model USES its conditioning.

    Returns (val_loss, waviness_gap, pH_gap). Each gap is the loss with that conditioning
    channel forced to its null embedding MINUS the loss with it supplied - i.e. how much
    worse the model does when the channel is taken away, which is exactly "how much the
    channel is being used". Positive means used.

    Why this is measured at all. The MSE alone is nearly blind to conditioning: measured on
    two trained checkpoints, dropping a channel moves it by ~1e-4 on a loss of ~0.58, i.e.
    0.02%, while the run-to-run wobble in the same number is ~1e-3 and MIN_DELTA is 1e-5.
    The entire conditioning signal therefore sits an order of magnitude BELOW the noise floor
    of the metric that decides both when training stops and which checkpoint is kept - so a
    checkpoint whose pH/waviness conditioning is still half-developed scores exactly as well
    as one whose conditioning is mature, and early stopping happily fires on the former. That
    is not hypothetical: a run that stopped at step 12,375 with its best at 8,625 produced a
    visibly weaker editor than one that reached 15,625, at an indistinguishable val loss.
    These gaps do rank the two correctly, so they are logged and fed into early stopping.

    Fully deterministic: a fixed generator supplies x0/t, and val_collate_fn centre-crops
    (rather than random-crops), so the same pixels are scored every time and the three
    conditioning variants below see the identical xt.
    """
    model.eval()
    totals = {"both": 0.0, "no_wav": 0.0, "no_pH": 0.0}

    # Set a fixed generator for deterministic validation
    eval_gen = torch.Generator(device=DEVICE)
    eval_gen.manual_seed(12345) 
    
    for x_batch, pH_batch, wav_batch in dataloader:
        x1 = x_batch.to(DEVICE)
        pH = normalize_pH(pH_batch.to(DEVICE).float())
        wav = wav_batch.to(DEVICE).float()
        null = torch.full_like(pH, float("nan"))

        batch = {k: 0.0 for k in totals}

        # Average the loss over multiple random samplings for each batch
        for _ in range(num_noise_samples):
            x0 = torch.randn(x1.shape, generator=eval_gen, device=DEVICE)
            t = torch.rand(x1.shape[0], generator=eval_gen, device=DEVICE)

            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0

            # Same xt for all three, so the differences are the conditioning and nothing else
            for key, (pH_in, wav_in) in (("both", (pH, wav)),
                                         ("no_wav", (pH, null)),
                                         ("no_pH", (null, wav))):
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    loss = F.mse_loss(model(xt, t, pH_in, wav_in), target)
                batch[key] += loss.item()

        for key in totals:
            totals[key] += batch[key] / num_noise_samples

    model.train()
    n = len(dataloader)
    val_loss = totals["both"] / n
    return val_loss, totals["no_wav"] / n - val_loss, totals["no_pH"] / n - val_loss

def save_loss_history(train_steps, train_losses, val_steps, val_losses, out_path,
                      wav_gaps=None):
    """Dump the raw curves next to the plot so they can be re-plotted or compared across runs.

    wav_gaps is evaluate()'s waviness conditioning gap, on the same grid as val_steps. It is
    written alongside the loss because the two tell different stories and only one of them is
    about the deliverable - see evaluate().
    """
    val_lookup = dict(zip(val_steps, val_losses))
    gap_lookup = dict(zip(val_steps, wav_gaps or []))

    def row(s, tl):
        return [s, tl,
                f"{val_lookup[s]:.6f}" if s in val_lookup else "",
                f"{gap_lookup[s]:.8f}" if s in gap_lookup else ""]

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "train_loss_live", "val_loss_ema", "waviness_cond_gap"])
        for s, tl in zip(train_steps, train_losses):
            writer.writerow(row(s, f"{tl:.6f}"))
        # val is sampled on a coarser grid than train, so emit any val-only steps too
        for s in val_steps:
            if s not in set(train_steps):
                writer.writerow(row(s, ""))


def plot_loss_history(train_steps, train_losses, val_steps, val_losses, best_step,
                      out_path="outputs/training_loss.png", wav_gaps=None,
                      best_cond_step=None):
    """Save the train/val loss curves once training ends (either normally or via early stop)."""
    if not train_steps:
        print("No loss history recorded - skipping plot.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # the per-step train loss is very noisy (single batch, random t and noise), so overlay a
    # running mean - otherwise the trend is invisible under the jitter
    window = max(1, min(21, len(train_losses) // 10))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(np.array(train_losses), kernel, mode="valid")
        smooth_steps = train_steps[window - 1:]
    else:
        smoothed, smooth_steps = None, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, log_scale in zip(axes, (False, True)):
        ax.plot(train_steps, train_losses, color="tab:blue", alpha=0.25, lw=0.8,
                label="Train, live model (raw)")
        if smoothed is not None:
            ax.plot(smooth_steps, smoothed, color="tab:blue", lw=1.8,
                    label=f"Train, live model (mean of {window})")
        if val_steps:
            ax.plot(val_steps, val_losses, color="tab:red", lw=1.8, marker="o", ms=3,
                    label="Val, EMA model")
        if best_step is not None:
            ax.axvline(best_step, color="gray", ls="--", lw=1.2,
                       label=f"best checkpoint (step {best_step})")
        ax.set_xlabel("training step")
        ax.set_ylabel("flow-matching MSE")
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training / validation loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Loss curve saved to: {out_path}")

    if wav_gaps:
        gap_path = os.path.splitext(out_path)[0] + "_conditioning.png"
        fig2, ax = plt.subplots(figsize=(9, 4))
        ax.plot(val_steps, wav_gaps, color="tab:green", lw=1.8, marker="o", ms=3,
                label="waviness conditioning gap (EMA)")
        ax.axhline(0.0, color="gray", lw=0.8)
        if best_step is not None:
            ax.axvline(best_step, color="gray", ls="--", lw=1.2, label=f"best val loss ({best_step})")
        if best_cond_step is not None:
            ax.axvline(best_cond_step, color="tab:green", ls=":", lw=1.4,
                       label=f"best conditioning ({best_cond_step})")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss(no waviness) - loss(waviness)")
        ax.set_title("How much the model uses its waviness conditioning\n"
                     "(the val loss above is ~blind to this - see evaluate())", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(gap_path, dpi=130); plt.close(fig2)
        print(f"Conditioning curve saved to: {gap_path}")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    save_loss_history(train_steps, train_losses, val_steps, val_losses, csv_path,
                      wav_gaps=wav_gaps)
    print(f"Loss history saved to: {csv_path}")


class HeightStratifiedBatchSampler(torch.utils.data.Sampler):
    """Batches of similar source height, still weighted by inverse pH-bucket frequency.

    A plain shuffled batch always contains a few very short crops, which forces the collate
    to pick a short frame - so the tall frames that high waviness needs would never occur.
    A fraction of batches is therefore drawn only from sources at least TALL_MIN_HEIGHT
    tall, letting the collate safely choose a tall frame for those. The inverse-pH-frequency
    weighting is applied within whichever pool is used, so pH balance is preserved either
    way.
    """

    def __init__(self, heights, weights, batch_size, num_batches,
                 tall_min=TALL_MIN_HEIGHT, tall_fraction=TALL_BATCH_FRACTION, seed=SEED):
        self.heights = np.asarray(heights)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.tall_fraction = tall_fraction
        self.tall_pool = np.flatnonzero(self.heights >= tall_min)
        self.full_pool = np.arange(len(self.heights))
        self.rng = np.random.default_rng(seed)

    def _draw(self, pool):
        p = self.weights[pool]
        return [int(i) for i in self.rng.choice(pool, size=self.batch_size,
                                                replace=True, p=p / p.sum())]

    def __iter__(self):
        for _ in range(self.num_batches):
            tall = (len(self.tall_pool) >= self.batch_size // 4
                    and self.rng.random() < self.tall_fraction)
            yield self._draw(self.tall_pool if tall else self.full_pool)

    def __len__(self):
        return self.num_batches


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(SEED) 
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set up DataLoaders with dynamic collate functions for training and validation
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_dataset = MicrotubuleDataset(DATA_DIR, is_train=True, min_height=MIN_SOURCE_HEIGHT)
    val_dataset = MicrotubuleDataset(DATA_DIR, is_train=False, min_height=MIN_SOURCE_HEIGHT)

    # pH buckets are heavily imbalanced (e.g. 36 vs 136 images) - weight samples by
    # inverse pH-bucket frequency so each pH gets roughly equal gradient signal.
    train_phs = [ph for _, ph in train_dataset.samples]
    ph_counts = Counter(train_phs)
    train_sample_weights = [1.0 / ph_counts[ph] for ph in train_phs]
    train_heights = [Image.open(p).size[1] for p, _ in train_dataset.samples]
    train_sampler = HeightStratifiedBatchSampler(
        train_heights, train_sample_weights, BATCH_SIZE,
        num_batches=max(1, len(train_dataset) // BATCH_SIZE))
    n_tall = len(train_sampler.tall_pool)
    print(f"training images: {len(train_dataset)} (>= {MIN_SOURCE_HEIGHT}px tall), "
          f"of which {n_tall} are >= {TALL_MIN_HEIGHT}px and can carry a tall frame")

    # Normalization constants for the waviness conditioning channel. These get stored as
    # buffers on the model itself (see WavinessEmbedding in model.py) so they travel with
    # the checkpoint instead of risking going stale in a sidecar file - but they MUST be
    # fit against what training actually feeds the model (dynamic_collate_fn's per-crop,
    # post-augmentation measurement), not the per-source-image labels in
    # train_dataset.samples: measured directly, those two distributions are wildly
    # different (whole-image mean=4.09px std=3.41px vs per-crop mean=6.74px std=8.53px,
    # crops reaching up to 82px) because a small random window can show far more locally
    # exaggerated curvature than the fibre's overall shape. Z-scoring against the wrong
    # (narrower) std means most real training examples land at extreme, saturating z-scores
    # instead of the well-behaved range normalization is supposed to produce - confirmed as
    # the direct cause of a trained checkpoint showing ~zero waviness sensitivity. Sampling
    # through the real collate pipeline here, before training starts, is the only way to
    # get the number that's actually correct for what training will feed the model.
    print("sampling the actual per-crop waviness distribution through dynamic_collate_fn "
          "(this determines the model's normalization scale - see the note above)...")
    # dynamic_collate_fn draws from Python's global random module and torch's global RNG
    # internally (random.choice, T.RandomCrop, T.RandomHorizontalFlip, ...), NOT an
    # isolated generator - sampling through it here, before the main loop, was silently
    # consuming and shifting that global state, so the ACTUAL training run's random crop/
    # flip/jitter sequence was never the one SEED=42 was supposed to produce, and changed
    # any time this sampling loop's size changed. Save and restore both RNGs around it so
    # measuring statistics has no side effect on the run being measured.
    _random_state, _torch_state = random.getstate(), torch.get_rng_state()
    _sample_gen = torch.Generator(); _sample_gen.manual_seed(SEED)
    _crop_wavs = []
    for _ in range(50):
        idx = torch.randint(0, len(train_dataset), (BATCH_SIZE,), generator=_sample_gen).tolist()
        _, _, _wav_batch = dynamic_collate_fn([train_dataset[i] for i in idx])
        _crop_wavs.append(_wav_batch)
    # the stratified sampler also emits all-tall batches, which reach frame heights (and so
    # waviness values) a uniformly-sampled batch never can - include some or the
    # normalization scale would be fit to only the shorter half of the real distribution
    for _ in range(20):
        _, _, _wav_batch = dynamic_collate_fn(
            [train_dataset[i] for i in train_sampler._draw(train_sampler.tall_pool)])
        _crop_wavs.append(_wav_batch)
    random.setstate(_random_state); torch.set_rng_state(_torch_state)
    real_wavs = torch.cat(_crop_wavs)
    real_wavs = real_wavs[~torch.isnan(real_wavs)].numpy()
    if real_wavs.size == 0:
        raise RuntimeError("No sampled crop produced a measurable waviness - check that "
                           "waviness.trace_fibre can find a fibre in this dataset.")
    WAV_MEAN, WAV_STD = float(real_wavs.mean()), float(real_wavs.std())
    print(f"waviness conditioning: {real_wavs.size}/{70 * BATCH_SIZE} sampled crops measurable, "
          f"mean={WAV_MEAN:.2f}px std={WAV_STD:.2f}px")

    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=4, worker_init_fn=seed_worker, generator=g,
        collate_fn=dynamic_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, worker_init_fn=seed_worker,
        collate_fn=val_collate_fn     
    )

    
    model = ConditionalUNet(waviness_mean=WAV_MEAN, waviness_std=WAV_STD).to(DEVICE)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # T_max counts scheduler.step() calls, which happen once per OPTIMIZER step, not once
    # per iteration - with T_max=ITERATIONS the cosine would only traverse 1/ACCUMULATION_STEPS
    # of its cycle and the LR would still be at ~85% of base when training ends.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, ITERATIONS // ACCUMULATION_STEPS))

    best_val_loss = float('inf')
    best_step = None
    best_wav_gap = -float('inf')
    best_cond_step = None
    epochs_without_improvement = 0

    # loss history for the end-of-training plot
    train_hist_steps, train_hist_losses = [], []
    val_hist_steps, val_hist_losses = [], []
    wav_gap_hist = []

    model.train()
    step = 0
    ema_updates = 0
    stop_training = False

    print(f"{DEVICE}.")
    print(f"validation images: {len(val_dataset)}")

    while step < ITERATIONS and not stop_training:
        for x_batch, pH_batch, wav_batch in train_dataloader:
            if step >= ITERATIONS:
                break

            x1 = x_batch.to(DEVICE)
            pH_raw = pH_batch.to(DEVICE).float()
            pH_jittered = (pH_raw + torch.randn_like(pH_raw) * PH_JITTER_STD).clamp(PH_MIN, PH_MAX)
            pH = normalize_pH(pH_jittered)
            # Raw (pixel-unit) waviness - the model normalizes internally via its own
            # buffers (see ConditionalUNet). No PH_JITTER_STD-style jitter: unlike the
            # sparse discrete pH buckets, waviness is already a continuous real measurement
            # with no discreteness to smooth over.
            wav_raw = wav_batch.to(DEVICE).float()

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=DEVICE)

            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0

            drop_mask = torch.rand(x1.shape[0], device=DEVICE) < CFG_DROPOUT
            pH_input = torch.where(drop_mask, torch.full_like(pH, float("nan")), pH)
            # Independent draw from pH's dropout, so the model sees geometry-only,
            # texture-only, both, and neither conditioning in roughly the proportions their
            # separate rates imply - a prerequisite for guiding the two axes independently
            # at inference later. wav_raw is already NaN wherever trace_fibre couldn't
            # measure the source crop, so this where() naturally unions "dropped for CFG"
            # with "unmeasurable" into the same null path.
            wav_drop_mask = torch.rand(x1.shape[0], device=DEVICE) < WAVINESS_CFG_DROPOUT
            wav_input = torch.where(wav_drop_mask, torch.full_like(wav_raw, float("nan")), wav_raw)


            device_type_autocast = "cuda" if "cuda" in DEVICE else "cpu"
            with torch.autocast(device_type=device_type_autocast, dtype=torch.bfloat16):
                pred = model(xt, t, pH_input, wav_input)
                loss = F.mse_loss(pred, target)
                
            # keep the unscaled value for logging - the scaled one is 1/ACCUMULATION_STEPS of
            # the real loss and isn't comparable to the val loss printed next to it
            train_loss_value = loss.item()
            loss = loss / ACCUMULATION_STEPS

            loss.backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == ITERATIONS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
                # EMA with a warmup ramp. A fixed 0.9999 decay keeps 0.9999^n weight on the
                # RANDOM INITIALIZATION after n updates - and because EMA only updates once per
                # optimizer step, gradient accumulation cuts n by ACCUMULATION_STEPS. At 57.5k
                # iterations with ACCUMULATION_STEPS=4 that left ~24% of the checkpointed EMA
                # weights as pure random init, which is what produced washed-out, near-white
                # samples. Ramping the decay in makes early updates track the live model closely
                # so the init washes out immediately, regardless of the accumulation setting.
                with torch.no_grad():
                    ema_updates += 1
                    decay = min(EMA_DECAY, (1 + ema_updates) / (10 + ema_updates))
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(decay).add_(p, alpha=1 - decay)
            
            # record the train curve on its own cadence, independent of which branch prints
            if step % 100 == 0:
                train_hist_steps.append(step)
                train_hist_losses.append(train_loss_value)

            # early stopping
            if step > 0 and step % EVAL_INTERVAL == 0:
                val_loss, wav_gap, ph_gap = evaluate(ema_model, val_dataloader)
                val_hist_steps.append(step)
                val_hist_losses.append(val_loss)
                wav_gap_hist.append(wav_gap)
                # NOTE: train loss is the LIVE model, val loss is the EMA model - early in
                # training the EMA lags badly, so a large gap here means the EMA hasn't caught
                # up yet, not necessarily overfitting. cond= is how much the model USES its
                # waviness / pH conditioning (see evaluate) - watch it keep rising long after
                # the val loss has gone flat.
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss (live): {train_loss_value:.4f} "
                      f"| Val Loss (EMA): {val_loss:.4f} | cond wav {wav_gap:+.5f} pH {ph_gap:+.5f}")

                improved = False
                if val_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = val_loss
                    best_step = step
                    improved = True
                    torch.save(ema_model.state_dict(), CHECKPOINT_PATH)
                # A separate checkpoint for the best-CONDITIONED model. The two criteria
                # disagree, and for this project's actual deliverable - editing an image to a
                # requested pH - the conditioning one is the relevant deliverable, while the
                # MSE one measures denoising. Saving both costs one extra file and settles the
                # question by comparison instead of by an arbitrary weighting between two
                # quantities three orders of magnitude apart.
                if wav_gap > (best_wav_gap + COND_MIN_DELTA):
                    best_wav_gap = wav_gap
                    best_cond_step = step
                    improved = True
                    torch.save(ema_model.state_dict(), COND_CHECKPOINT_PATH)

                # Patience counts evals where NEITHER criterion improved. Counting only the
                # val loss is what let a run stop at step 12,375 with its conditioning still
                # visibly developing - see evaluate()'s docstring for the numbers.
                epochs_without_improvement = 0 if improved else epochs_without_improvement + 1

                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping aktivován na kroku {step} (val loss ani conditioning "
                          f"se nezlepšily {PATIENCE}x za sebou). Trénink ukončen.")
                    stop_training = True
                    break

            elif step % 100 == 0:
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss (live): {train_loss_value:.4f}")

            step += 1

    if not stop_training:
        torch.save(ema_model.state_dict(), FINAL_CHECKPOINT_PATH)
        print(f"Training completed. Final model saved as {FINAL_CHECKPOINT_PATH!r}.")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}. Model saved as {CHECKPOINT_PATH!r}.")
    print(f"Best waviness conditioning gap: {best_wav_gap:+.5f} at step {best_cond_step}. "
          f"Model saved as {COND_CHECKPOINT_PATH!r}. Compare the two - they are selected on "
          f"different things and the conditioning one is usually the better editor.")

    # always plot, whether we finished the full run or stopped early
    plot_loss_history(train_hist_steps, train_hist_losses,
                      val_hist_steps, val_hist_losses, best_step,
                      wav_gaps=wav_gap_hist, best_cond_step=best_cond_step)

if __name__ == "__main__":
    main()