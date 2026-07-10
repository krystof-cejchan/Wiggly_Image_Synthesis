import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset

# imports from local files
from train import val_collate_fn
from img2img import edit_image  
from sample import normalize_pH

def prepare_images_for_fid(img_tensor):
    """Prepare images for FID: convert grayscale to RGB and map to uint8."""
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

@torch.no_grad()
def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    CHECKPOINT_PATH = "checkpoints/cfm_best_ema.pt"
    DATA_DIR = "data/cropped/cropped_output"
    
    # Translation FID settings
    SOURCE_PH = 5.8   # starting pH
    TARGET_PH = 8.8   # target pH
    NUM_SAMPLES = 1000 
    BATCH_SIZE = 16
    
    # ==========================================
    # OPTIMALIZOVANÉ PARAMETRY PRO LEPŠÍ FID
    # ==========================================
    STRENGTH = 0.40   # Sníženo pro zachování reálné textury pozadí
    SCALE = 2.0       # Sníženo pro omezení přesaturace pixelů
    NUM_STEPS = 250   # Zvýšeno pro hladší integraci a méně mikro-šumu

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
    
    dataset_target = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    target_samples = [item for item in dataset_target.samples if abs(item[1] - TARGET_PH) < 0.1]
    dataset_target.samples = target_samples[:NUM_SAMPLES]
    
    actual_target_samples = len(dataset_target.samples)
    print(f"Found {actual_target_samples} real images for pH {TARGET_PH}.")
    
    if actual_target_samples < 2:
        print("Error: At least 2 images are required in the target set to compute FID.")
        return

    dataloader_target = DataLoader(
        dataset_target, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, collate_fn=val_collate_fn
    )

    for real_batch, _ in tqdm(dataloader_target, desc="Real data (Target)"):
        real_batch = real_batch.to(DEVICE)
        real_images_fid = prepare_images_for_fid(real_batch)
        fid.update(real_images_fid, real=True)

    # ==========================================
    # 3. Process generated images from source pH to target pH
    # ==========================================
    print(f"\n--- Extracting features from synthetic images (translation from pH {SOURCE_PH} to {TARGET_PH}) ---")
    
    dataset_source = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    source_samples = [item for item in dataset_source.samples if abs(item[1] - SOURCE_PH) < 0.1]
    
    # limit the source set to the same size as the target set for a fair comparison
    dataset_source.samples = source_samples[:actual_target_samples]
    
    actual_source_samples = len(dataset_source.samples)
    print(f"Found {actual_source_samples} source images for pH {SOURCE_PH}.")
    
    if actual_source_samples < 2:
        print("Error: At least 2 images are required in the source set to compute FID.")
        return

    dataloader_source = DataLoader(
        dataset_source, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, collate_fn=val_collate_fn
    )

    for source_batch, _ in tqdm(dataloader_source, desc="Synthetic data (Translation)"):
        source_batch = source_batch.to(DEVICE)
        
        # run the image edit step from img2img.py
        edited_batch = edit_image(
            model=model,
            ref_image=source_batch,
            source_pH=SOURCE_PH,
            target_pH=TARGET_PH,
            denoising_strength=STRENGTH,
            num_steps=NUM_STEPS,
            contrastive_scale=SCALE,
            contrast=1.0,  # bugfix: reálné obrázky kontrast nedostávají, default 1.2 zkresloval FID
        )
        
        fake_images_fid = prepare_images_for_fid(edited_batch)
        fid.update(fake_images_fid, real=False)

    # ==========================================
    # 4. Compute FID score
    # ==========================================
    print("\nComputing translation Fréchet Inception Distance...")
    fid_score = fid.compute()
    print("=" * 60)
    print(f"Final FID score (Translation {SOURCE_PH} -> {TARGET_PH}): {fid_score.item():.4f}")
    print(f"Fixed parameters used: Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}")
    print("=" * 60)

if __name__ == "__main__":
    main()