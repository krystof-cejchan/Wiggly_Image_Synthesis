import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset

# imports from local files
from img2img import edit_image, load_and_preprocess_image
from sample import normalize_pH

# Crops this thin need to be almost entirely mirror-hallucinated to reach any
# usable window size, so they'd dominate a small-sample metric with synthetic
# content rather than real structure. Drop them from both the real and fake sets.
MIN_SIDE = 32

def prepare_images_for_fid(img_tensor):
    """Prepare tensors for InceptionV3 (convert to RGB and uint8)."""
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2

    img_tensor = (img_tensor * 255).to(torch.uint8)

    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)

    return img_tensor

def warmup_gpu(model, num_runs=5):
    """Run a few warm-up passes so GPU timing and memory stabilize."""
    print(f"Running {num_runs} warm-up iterations...")
    model.eval()
    dummy_x = torch.randn(2, 1, 128, 128, device=DEVICE)
    dummy_t = torch.rand(2, device=DEVICE)
    dummy_ph = normalize_pH(torch.tensor([7.0, 7.0])).to(DEVICE)

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_x, dummy_t, dummy_ph)

    if "cuda" in DEVICE:
        torch.cuda.synchronize()
    print("Hardware warmed up and ready.")

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

@torch.no_grad()
def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    CHECKPOINT_PATH = "checkpoints/cfm_best_ema.pt"
    DATA_DIR = "data/cropped/cropped_output"

    # Translation FID settings
    SOURCE_PH = 5.8   # starting pH
    TARGET_PH = 12.8   # target pH
    MAX_SAMPLES = 1000

    # ==========================================
    # OPTIMALIZOVANÉ PARAMETRY PRO LEPŠÍ FID
    # ==========================================
    STRENGTH = 0.80   # Sníženo pro zachování reálné textury pozadí
    SCALE = 5.0       # Sníženo pro omezení přesaturace pixelů
    NUM_STEPS = 100   # Zvýšeno pro hladší integraci a méně mikro-šumu

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} was not found.")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # A) feature=64 místo 2048 (viz DOPORUCENI_METRIKY.md)
    fid = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)
    warmup_gpu(model, num_runs=5)

    # ==========================================
    # 2. Process real target images
    # ==========================================
    print(f"\n--- Extracting features from real images (pH {TARGET_PH}) ---")

    dataset = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    target_paths = filter_paths(dataset.samples, TARGET_PH, limit=MAX_SAMPLES)
    print(f"Found {len(target_paths)} real images for pH {TARGET_PH}.")

    if len(target_paths) < 2:
        print("Error: At least 2 images are required in the target set to compute FID.")
        return

    # NOTE: images are evaluated at their NATIVE resolution (no forced square
    # crop) and fed one at a time — real crops are non-square strips of wildly
    # varying size (down to a few tens of px), and torchmetrics' FID resizes
    # each update() call to 299x299 internally, so per-image shapes are fine.
    # Squashing every crop into a fixed 128x128 square beforehand (the old
    # approach, reusing train.py's val_collate_fn) required so much mirror
    # padding that most "real" eval images were mostly synthetic content too.
    for path in tqdm(target_paths, desc="Real data (Target)"):
        real_img, _ = load_and_preprocess_image(path)
        real_images_fid = prepare_images_for_fid(real_img)
        fid.update(real_images_fid, real=True)

    # ==========================================
    # 3. Process generated images from source pH to target pH
    # ==========================================
    print(f"\n--- Extracting features from synthetic images (translation from pH {SOURCE_PH} to {TARGET_PH}) ---")

    source_paths = filter_paths(dataset.samples, SOURCE_PH, limit=len(target_paths))
    print(f"Found {len(source_paths)} source images for pH {SOURCE_PH}.")

    if len(source_paths) < 2:
        print("Error: At least 2 images are required in the source set to compute FID.")
        return

    # Each source image goes through edit_image() at its own native resolution,
    # exactly like a real img2img.py invocation — same sliding-window/stride
    # defaults a user would get, so the measured FID reflects actual product
    # behavior instead of an eval-only double-padding artifact.
    for path in tqdm(source_paths, desc="Synthetic data (Translation)"):
        source_img, _ = load_and_preprocess_image(path)

        edited_img = edit_image(
            model=model,
            ref_image=source_img,
            source_pH=SOURCE_PH,
            target_pH=TARGET_PH,
            denoising_strength=STRENGTH,
            num_steps=NUM_STEPS,
            contrastive_scale=SCALE,
            contrast=1.0,  # bugfix: reálné obrázky kontrast nedostávají, default 1.2 zkresloval FID
        )

        fake_images_fid = prepare_images_for_fid(edited_img)
        fid.update(fake_images_fid, real=False)

    # ==========================================
    # 4. Compute FID score
    # ==========================================
    print("\nComputing translation Fréchet Inception Distance...")
    fid_score = fid.compute()
    print("=" * 60)
    print(f"Final FID score (Translation {SOURCE_PH} -> {TARGET_PH}): {fid_score.item():.4f}")
    print(f"Real n={len(target_paths)}, Fake n={len(source_paths)}")
    print(f"Fixed parameters used: Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}")
    print("=" * 60)

if __name__ == "__main__":
    main()
