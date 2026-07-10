"""
Doporučená evaluace img2img překladu pH — spojuje body A + B + C z
DOPORUCENI_METRIKY.md:

  A) primární metrika je nevychýlené KID (FID jen orientačně, při malém n je biased)
  B) vyhodnocuje se na PROUŽKOVÉM rozměru (64x256), který odpovídá datům i
     trénovacím velikostem — ne na čtverci 128x128; reálné i falešné vzorky se
     zpracují identicky
  C) místo ImageNet-Inceptionu se používá DINOv2 backbone (dino_features.py)

Nutný stažený checkpoint (viz checkpoints/download_trained_model.txt) a internet
pro první stažení DINOv2 vah.
"""
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset
from img2img import edit_image, safe_mirror_pad_4d
from dino_features import DinoV2Features

# --- Konfigurace ---
CHECKPOINT_PATH = "checkpoints/cfm_best_ema.pt"
DATA_DIR = "data/cropped/cropped_output"
SOURCE_PH = 5.8
TARGET_PH = 8.8
BATCH_SIZE = 8

# B) proužkový rozměr (H, W) — je i mezi trénovacími velikostmi (TRAIN_SIZES),
#    dělitelný 16 (nutné kvůli 4x downsamplu v UNetu).
EVAL_H, EVAL_W = 64, 256

# img2img parametry (shodné s eval_kid_img2img.py, ať jsou výsledky srovnatelné)
STRENGTH, SCALE, NUM_STEPS = 0.40, 2.0, 250


def strip_collate_fn(batch):
    """B) Mirror-pad na (EVAL_H, EVAL_W) a deterministicky ořízni na přesný rozměr.
    Stejné zpracování pro reálné i (přes edit_image) falešné vzorky."""
    imgs = []
    for img, _ in batch:                          # img: (1, h, w) v [-1, 1]
        padded = safe_mirror_pad_4d(img.unsqueeze(0), EVAL_H, EVAL_W)  # (1,1,EVAL_H,EVAL_W)
        imgs.append(padded.squeeze(0))
    phs = torch.stack([ph for _, ph in batch])
    return torch.stack(imgs), phs


def prepare_for_dino(img_tensor):
    """DINO wrapper čeká float [0, 1]. Reálné jsou v [-1, 1], falešné už v [0, 1]."""
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    return img_tensor.clamp(0, 1).float()


def load_ph_dataset(ph):
    ds = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    ds.samples = [s for s in ds.samples if abs(s[1] - ph) < 0.1]
    return ds


@torch.no_grad()
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Chybí checkpoint {CHECKPOINT_PATH} (viz checkpoints/download_trained_model.txt).")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # C) sdílený DINOv2 backbone pro reálné i falešné vzorky
    print("Načítám DINOv2 backbone (torch.hub, první běh stahuje váhy)...")
    feature_extractor = DinoV2Features().to(DEVICE).eval()

    # data — stejný počet zdrojových a cílových pro férové srovnání
    ds_target = load_ph_dataset(TARGET_PH)
    ds_source = load_ph_dataset(SOURCE_PH)
    n = min(len(ds_target), len(ds_source))
    ds_target.samples = ds_target.samples[:n]
    ds_source.samples = ds_source.samples[:n]
    print(f"Reálných (pH {TARGET_PH}): {len(ds_target)} | zdrojových (pH {SOURCE_PH}): {len(ds_source)}")
    if n < 2:
        print("Málo vzorků pro výpočet metriky.")
        return

    dl_target = DataLoader(ds_target, batch_size=BATCH_SIZE, shuffle=False, collate_fn=strip_collate_fn)
    dl_source = DataLoader(ds_source, batch_size=BATCH_SIZE, shuffle=False, collate_fn=strip_collate_fn)

    # A) KID (primární, nevychýlené) + FID (orientačně). normalize=True -> předáváme float [0,1];
    #    u custom feature extractoru se obrázky posílají rovnou do DinoV2Features.
    subset = min(n, 50)
    kid = KernelInceptionDistance(feature=feature_extractor, subset_size=subset, normalize=True).to(DEVICE)
    fid = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(DEVICE)

    # reálné (cílové pH)
    for real, _ in tqdm(dl_target, desc="Reálné (target)"):
        real = prepare_for_dino(real.to(DEVICE))
        kid.update(real, real=True)
        fid.update(real, real=True)

    # falešné = překlad source -> target (B: proužkový rozměr, contrast=1.0)
    for src, _ in tqdm(dl_source, desc="Překlad (source->target)"):
        src = src.to(DEVICE)
        edited = edit_image(
            model, src, source_pH=SOURCE_PH, target_pH=TARGET_PH,
            denoising_strength=STRENGTH, num_steps=NUM_STEPS,
            contrastive_scale=SCALE, contrast=1.0,
        )
        edited = prepare_for_dino(edited.to(DEVICE))
        kid.update(edited, real=False)
        fid.update(edited, real=False)

    kid_mean, kid_std = kid.compute()
    fid_score = fid.compute()

    print("=" * 64)
    print(f"DINOv2  KID (x100): {kid_mean.item() * 100:.4f} +/- {kid_std.item() * 100:.4f}   <- PRIMÁRNÍ")
    print(f"DINOv2  FID       : {fid_score.item():.4f}   (orientačně; při n={n} je i s DINOv2 biased)")
    print(f"Rozměr={EVAL_H}x{EVAL_W} | backbone=DINOv2 | Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}")
    print("=" * 64)


if __name__ == "__main__":
    main()
