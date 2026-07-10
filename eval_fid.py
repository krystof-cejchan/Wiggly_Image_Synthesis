import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import PH_MIN, PH_MAX, DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset
from sample import sample, normalize_pH  # normalize_pH is used by the model inputs
from train import val_collate_fn

def prepare_images_for_fid(img_tensor):
    """Convert image tensors to the format used by InceptionV3."""
    # If the image is in [-1, 1], rescale it to [0, 1]
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
        
    # Map values to 0-255 and switch to uint8
    img_tensor = (img_tensor * 255).to(torch.uint8)
    
    # If the image is grayscale, duplicate it across RGB channels
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
    return img_tensor

def warmup_gpu(model, num_runs=5):
    """Run a few warm-up passes so the GPU settles before timed operations."""
    print(f"Running {num_runs} warm-up iterations...")
    model.eval()
    dummy_x = torch.randn(2, 1, 128, 128, device=DEVICE)
    dummy_t = torch.rand(2, device=DEVICE)
    dummy_ph = normalize_pH(torch.tensor([7.0, 7.0])).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_x, dummy_t, dummy_ph)
    # bugfix: na stroji bez CUDA (Mac/CPU) by holé synchronize() spadlo
    if "cuda" in DEVICE:
        torch.cuda.synchronize()
    print("Hardware warmed up and ready.")

@torch.no_grad()
def main():
    # 1. Configuration
    CHECKPOINT_PATH = "./checkpoints/cfm_best_ema.pt"
    DATA_DIR = "./data/cropped/cropped_output"
    TARGET_PH = 8.8  # pH value to evaluate
    BATCH_SIZE = 16
    NUM_SAMPLES = 1000  # aim for 1000+ images for a reliable FID

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} was not found.")
        return

    # Initialize model
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Initialize FID metric
    # A) feature=64 místo 2048: kovarianci 64x64 jde z desítek vzorků odhadnout
    #    řádově lépe než 2048x2048. FID i tak zůstává při malém n biased -> ber
    #    ho relativně, primární metrika je KID (viz DOPORUCENI_METRIKY.md).
    fid = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)

    # Warm up the model and GPU before we start collecting stats
    warmup_gpu(model, num_runs=5)

    # 2. Process real images
    print(f"\n--- Extracting features from real images (pH {TARGET_PH}) ---")
    # Load the evaluation set without training augmentations
    dataset = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    
    filtered_samples = [item for item in dataset.samples if abs(item[1] - TARGET_PH) < 0.1]
    dataset.samples = filtered_samples[:NUM_SAMPLES]
    
    actual_num_samples = len(dataset.samples)
    print(f"Found {actual_num_samples} real images for pH {TARGET_PH} (requested {NUM_SAMPLES}).")
    
    if actual_num_samples < 2:
        print(f"Error: Not enough real images found for pH {TARGET_PH} to compute FID. At least 2 are required.")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=val_collate_fn
    )

    for real_batch, _ in tqdm(dataloader, desc="Real data"):
        real_batch = real_batch.to(DEVICE)
        real_images_fid = prepare_images_for_fid(real_batch)
        fid.update(real_images_fid, real=True)

    # Process synthetic images
    print(f"\n--- Generating and extracting features from synthetic images (pH {TARGET_PH}) ---")
    num_batches = (actual_num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for _ in tqdm(range(num_batches), desc="Synthetic data"):
        current_batch_size = min(BATCH_SIZE, actual_num_samples)
        actual_num_samples -= current_batch_size

        fake_batch = sample(model, pH_query=TARGET_PH, num_samples=current_batch_size, num_steps=100)
        
        fake_images_fid = prepare_images_for_fid(fake_batch)
        fid.update(fake_images_fid, real=False)

    fid_score = fid.compute()
    print("=" * 50)
    print(f"Final FID score (for pH {TARGET_PH}): {fid_score.item():.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()