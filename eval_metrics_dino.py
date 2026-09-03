"""
The recommended evaluation of img2img pH translation - combines two changes:

  A) the primary metric is the unbiased KID (FID is orientational only; at small n
     it is biased)
  C) a DINOv2 backbone (dino_features.py) is used instead of ImageNet-Inception

Everything is evaluated at each image's NATIVE resolution (no fixed square or strip).
The DINOv2 wrapper rescales the image to 224x224 itself regardless of input shape, so
unlike the earlier approach (mirror-padding to a fixed 64x256 strip before edit_image)
no artificial padding is needed. That padding used to be applied twice to translated
images (once here, once inside edit_image's sliding window) and contaminated the output
with mirrored pixels through the global self-attention - the same bug that was fixed in
eval_fid_img2img.py.

Requires a downloaded checkpoint (see checkpoints/download_trained_model.txt) and
internet access the first time, to fetch the DINOv2 weights.
"""
import os
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_PATH
from model import from_state_dict
from dataset import MicrotubuleDataset
from img2img import edit_image, load_and_preprocess_image
from dino_features import DinoV2Features

# --- Konfigurace ---
DATA_DIR = "data/cropped/cropped_output"
SOURCE_PH = 5.8
TARGET_PH = 8.8

# Crops this thin need to be almost entirely mirror-hallucinated to reach any
# usable window size, so they'd dominate a small-sample metric with synthetic
# content rather than real structure. Drop them from both the real and fake sets.
MIN_SIDE = 32

# img2img parameters (identical to eval_fid_img2img.py, so the results are comparable)
STRENGTH, SCALE, NUM_STEPS = 0.40, 2.0, 250


def filter_paths(samples, ph, limit=None):
    """Collect image paths for one pH bucket, dropping crops too small to
    evaluate without being mostly synthetic mirror-padding."""
    paths = [path for path, s_ph in samples if abs(s_ph - ph) < 0.1]
    kept, skipped = [], 0
    for path in paths:
        _, (w, h) = load_and_preprocess_image(path)
        if min(w, h) < MIN_SIDE:
            skipped += 1
        else:
            kept.append(path)
    if skipped:
        print(f"Skipped {skipped} crop(s) below MIN_SIDE={MIN_SIDE}px for pH {ph}.")
    return kept[:limit] if limit else kept


def prepare_for_dino(img_tensor):
    """The DINO wrapper expects float [0, 1]. Real images are in [-1, 1], fakes already in [0, 1]."""
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    return img_tensor.clamp(0, 1).float()


@torch.no_grad()
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Missing checkpoint {CHECKPOINT_PATH} (see checkpoints/download_trained_model.txt).")
        return

    model = from_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), DEVICE)
    model.eval()

    # C) one shared DINOv2 backbone for both the real and the fake samples
    print("Loading DINOv2 backbone (torch.hub; the first run downloads the weights)...")
    feature_extractor = DinoV2Features().to(DEVICE).eval()

    # data - equal numbers of source and target images, for a fair comparison
    dataset = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    target_paths = filter_paths(dataset.samples, TARGET_PH)
    source_paths = filter_paths(dataset.samples, SOURCE_PH, limit=len(target_paths))
    target_paths = target_paths[:len(source_paths)]
    n = len(target_paths)
    print(f"Real (pH {TARGET_PH}): {n} | source (pH {SOURCE_PH}): {len(source_paths)}")
    if n < 2:
        print("Too few samples to compute the metric.")
        return

    # A) KID (primary, unbiased) + FID (orientational). The custom feature extractor
    #    (DinoV2Features) bypasses torchmetrics' `normalize` cast entirely - images go
    #    straight into DinoV2Features.forward(), which itself expects float [0,1]
    #    (see prepare_for_dino), so the `normalize` value here is informational only.
    subset = min(n, 50)
    kid = KernelInceptionDistance(feature=feature_extractor, subset_size=subset, normalize=True).to(DEVICE)
    fid = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(DEVICE)

    # NOTE: native resolution, one image at a time - DinoV2Features resizes to
    # 224x224 internally regardless of input shape, so per-image native sizes
    # (including different shapes between real and fake) are fine.
    for path in tqdm(target_paths, desc="Real (target)"):
        real_img, _ = load_and_preprocess_image(path)
        real = prepare_for_dino(real_img.to(DEVICE))
        kid.update(real, real=True)
        fid.update(real, real=True)

    # fakes = source -> target translation, at native resolution (just like a real
    # img2img.py call), with contrast=1.0 so they match the real images
    for path in tqdm(source_paths, desc="Translation (source->target)"):
        src_img, _ = load_and_preprocess_image(path)
        edited = edit_image(
            model, src_img, source_pH=SOURCE_PH, target_pH=TARGET_PH,
            denoising_strength=STRENGTH, num_steps=NUM_STEPS,
            contrastive_scale=SCALE, contrast=1.0,
        )
        edited = prepare_for_dino(edited.to(DEVICE))
        kid.update(edited, real=False)
        fid.update(edited, real=False)

    kid_mean, kid_std = kid.compute()
    fid_score = fid.compute()

    print("=" * 64)
    print(f"DINOv2  KID (x100): {kid_mean.item() * 100:.4f} +/- {kid_std.item() * 100:.4f}   <- PRIMARY")
    print(f"DINOv2  FID       : {fid_score.item():.4f}   (orientational; at n={n} it is biased even with DINOv2)")
    print(f"Real n={n}, Fake n={len(source_paths)} | backbone=DINOv2 | Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}")
    print("=" * 64)


if __name__ == "__main__":
    main()
