"""
Doporučená evaluace img2img překladu pH — spojuje body A + C z
DOPORUCENI_METRIKY.md:

  A) primární metrika je nevychýlené KID (FID jen orientačně, při malém n je biased)
  C) místo ImageNet-Inceptionu se používá DINOv2 backbone (dino_features.py)

Vyhodnocuje se na NATIVNÍM rozměru každého snímku (žádný pevný čtverec/proužek).
DINOv2 wrapper si obrázek sám přeškáluje na 224x224 bez ohledu na vstupní tvar,
takže na rozdíl od dřívějšího přístupu (mirror-pad na pevný 64x256 proužek před
edit_image) není potřeba žádné umělé dorovnávání - to dřív u přeložených snímků
kombinovalo dvojí padding (jednou zde, jednou uvnitř edit_image's sliding window)
a kontaminovalo výstup zrcadlenými pixely přes globální self-attention (stejný
bug, jaký byl opraven v eval_fid_img2img.py).

Nutný stažený checkpoint (viz checkpoints/download_trained_model.txt) a internet
pro první stažení DINOv2 vah.
"""
import os
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_PATH
from model import ConditionalUNet
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

# img2img parametry (shodné s eval_fid_img2img.py, ať jsou výsledky srovnatelné)
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
    """DINO wrapper čeká float [0, 1]. Reálné jsou v [-1, 1], falešné už v [0, 1]."""
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    return img_tensor.clamp(0, 1).float()


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
    dataset = MicrotubuleDataset(DATA_DIR, is_train=False, val_split_ratio=1.0)
    target_paths = filter_paths(dataset.samples, TARGET_PH)
    source_paths = filter_paths(dataset.samples, SOURCE_PH, limit=len(target_paths))
    target_paths = target_paths[:len(source_paths)]
    n = len(target_paths)
    print(f"Reálných (pH {TARGET_PH}): {n} | zdrojových (pH {SOURCE_PH}): {len(source_paths)}")
    if n < 2:
        print("Málo vzorků pro výpočet metriky.")
        return

    # A) KID (primární, nevychýlené) + FID (orientačně). Custom feature extraktor
    #    (DinoV2Features) ignoruje torchmetrics' `normalize` cast úplně - obrázky
    #    jdou rovnou do DinoV2Features.forward(), které samo čeká float [0,1]
    #    (viz prepare_for_dino), takže hodnota `normalize` zde je informativní.
    subset = min(n, 50)
    kid = KernelInceptionDistance(feature=feature_extractor, subset_size=subset, normalize=True).to(DEVICE)
    fid = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(DEVICE)

    # NOTE: native resolution, one image at a time - DinoV2Features resizes to
    # 224x224 internally regardless of input shape, so per-image native sizes
    # (including different shapes between real and fake) are fine.
    for path in tqdm(target_paths, desc="Reálné (target)"):
        real_img, _ = load_and_preprocess_image(path)
        real = prepare_for_dino(real_img.to(DEVICE))
        kid.update(real, real=True)
        fid.update(real, real=True)

    # falešné = překlad source -> target, na nativním rozměru (jako skutečné
    # img2img.py volání), contrast=1.0 aby se shodoval s reálnými snímky
    for path in tqdm(source_paths, desc="Překlad (source->target)"):
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
    print(f"DINOv2  KID (x100): {kid_mean.item() * 100:.4f} +/- {kid_std.item() * 100:.4f}   <- PRIMÁRNÍ")
    print(f"DINOv2  FID       : {fid_score.item():.4f}   (orientačně; při n={n} je i s DINOv2 biased)")
    print(f"Real n={n}, Fake n={len(source_paths)} | backbone=DINOv2 | Strength={STRENGTH}, Scale={SCALE}, Steps={NUM_STEPS}")
    print("=" * 64)


if __name__ == "__main__":
    main()
