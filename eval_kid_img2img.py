import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from config import DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset

# imports from local files
from train import val_collate_fn
from img2img import edit_image
from sample import normalize_pH

def prepare_images_for_metric(img_tensor):
    """Prepare tensors for InceptionV3 (convert to RGB and uint8)."""
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
        
    img_tensor = (img_tensor * 255).to(torch.uint8)
    
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
    return img_tensor

def warmup_gpu(model, num_runs=5):
    """Run a few warm-up passes so the GPU and memory settle before timing."""
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
    # 1. CONFIGURATION
    # ==========================================
    CHECKPOINT_PATH = "checkpoints/cfm_best_ema.pt"
    DATA_DIR = "data/cropped/cropped_output"
    
    SOURCE_PH = 5.8
    TARGET_PH = 8.8
    NUM_SAMPLES = 1000 
    BATCH_SIZE = 16
    
    # Parameters tuned for visual fidelity
    STRENGTH = 0.40
    SCALE = 2.0
    NUM_STEPS = 250

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} was not found.")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    warmup_gpu(model, num_runs=5)

    # ==========================================
    # 2. PROCESS REAL IMAGES (TARGET)
    # ==========================================
    print(f"\n--- Extracting features from real images (pH {TARGET_PH}) ---")
    
    dataset_target = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    target_samples = [item for item in dataset_target.samples if abs(item[1] - TARGET_PH) < 0.1]
    dataset_target.samples = target_samples[:NUM_SAMPLES]
    
    actual_target_samples = len(dataset_target.samples)
    print(f"Found {actual_target_samples} real images for pH {TARGET_PH}.")

    if actual_target_samples < 2:
        print("Error: At least 2 images are required to compute the metric.")
        return

    # Inicalizace KID s dynamickou velikostí subsetu
    # KID běžně počítá v dávkách (subsetech) po 50 nebo 100.
    # Pokud máme méně fotek než 50, musíme subset omezit na jejich celkový počet, jinak by metrika spadla.
    subset_size = min(actual_target_samples, 50)
    kid = KernelInceptionDistance(feature=2048, subset_size=subset_size, normalize=False).to(DEVICE)

    dataloader_target = DataLoader(
        dataset_target, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, collate_fn=val_collate_fn
    )

    for real_batch, _ in tqdm(dataloader_target, desc="Real data (Target)"):
        real_batch = real_batch.to(DEVICE)
        real_images_metric = prepare_images_for_metric(real_batch)
        kid.update(real_images_metric, real=True)

    # ==========================================
    # 3. PROCESS GENERATED IMAGES (TRANSLATION)
    # ==========================================
    print(f"\n--- Extracting features from synthetic images (Translation from pH {SOURCE_PH} to {TARGET_PH}) ---")
    
    dataset_source = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    source_samples = [item for item in dataset_source.samples if abs(item[1] - SOURCE_PH) < 0.1]
    dataset_source.samples = source_samples[:actual_target_samples]
    
    actual_source_samples = len(dataset_source.samples)
    print(f"Found {actual_source_samples} source images for pH {SOURCE_PH}.")

    if actual_source_samples < 2:
        print("Error: Not enough source images.")
        return

    dataloader_source = DataLoader(
        dataset_source, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, collate_fn=val_collate_fn
    )

    for source_batch, _ in tqdm(dataloader_source, desc="Synthetic data (Translation)"):
        source_batch = source_batch.to(DEVICE)
        
        edited_batch = edit_image(
            model=model,
            ref_image=source_batch,
            source_pH=SOURCE_PH,
            target_pH=TARGET_PH,
            denoising_strength=STRENGTH,
            num_steps=NUM_STEPS,
            contrastive_scale=SCALE,
            contrast=1.0  
        )
        
        fake_images_metric = prepare_images_for_metric(edited_batch)
        kid.update(fake_images_metric, real=False)

    print("\nPočítám Kernel Inception Distance (KID)...")
    kid_mean, kid_std = kid.compute()
    
    print("=" * 60)
    print(f"Výsledné KID skóre (Překlad {SOURCE_PH} -> {TARGET_PH})")
    print(f"KID (x100): {kid_mean.item() * 100:.4f} ± {kid_std.item() * 100:.4f}")
    print(f"Fixed parameters used: Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}, Contrast=1.0")
    print("=" * 60)

if __name__ == "__main__":
    main()