import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from config import PH_MIN, PH_MAX, DEVICE
from model import ConditionalUNet
from dataset import MicrotubuleDataset
from sample import sample, normalize_pH  # Využívá vaši stávající funkci pro generování z šumu

def prepare_images_for_fid(img_tensor):
    """
    Připraví tenzory pro InceptionV3 síť.
    InceptionV3 vyžaduje:
    1. 3 barevné kanály (RGB) - zkopírujeme náš 1 šedotónový kanál
    2. Formát torch.uint8 v rozsahu [0, 255]
    """
    # Pokud je obrázek v rozsahu [-1, 1] (výstup z datasetu), převedeme na [0, 1]
    if img_tensor.min() < 0:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
        
    # Převod na [0, 255] a uint8
    img_tensor = (img_tensor * 255).to(torch.uint8)
    
    # Zkopírování 1 kanálu do 3 kanálů (B, 3, H, W)
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
    return img_tensor

def warmup_gpu(model, num_runs=5):
    """
    Zajišťuje rigorózní metodiku testování. 
    Provede 5+ zahřívacích iterací naprázdno, aby se plně stabilizovaly 
    takty GPU a paměti před samotným měřením a generováním sady.
    """
    print(f"Provádím {num_runs} zahřívacích iterací (warm-up) pro stabilizaci hardwaru...")
    model.eval()
    dummy_x = torch.randn(2, 1, 128, 128, device=DEVICE)
    dummy_t = torch.rand(2, device=DEVICE)
    dummy_ph = normalize_pH(torch.tensor([7.0, 7.0])).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_x, dummy_t, dummy_ph)
    torch.cuda.synchronize()
    print("Hardware zahřátý a připraven.")

@torch.no_grad()
def main():
    # 1. Konfigurace
    CHECKPOINT_PATH = "checkpoints/cfm_best_ema.pt"
    DATA_DIR = "data/cropped/cropped_output"
    TARGET_PH = 8.8  # Testujeme na konkrétní hodnotě pH
    BATCH_SIZE = 16
    NUM_SAMPLES = 1000  # Pro relevantní FID je potřeba alespoň 1000+ obrázků

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Chyba: Checkpoint {CHECKPOINT_PATH} nebyl nalezen.")
        return

    # Inicializace modelu
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Inicializace metriky FID (feature=2048 je standardní výstupní vrstva Inception)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)

    # Zahřátí hardwaru pro rigorózní benchmark
    warmup_gpu(model, num_runs=5)

    # 2. Zpracování REÁLNÝCH obrázků
    print(f"\n--- Extrakce rysů z reálných obrázků (pH {TARGET_PH}) ---")
    # Zde využijeme upravený dataset, který načte všechny obrázky (is_train nezáleží, nepotřebujeme augmentace)
    dataset = MicrotubuleDataset(DATA_DIR, is_train=False)
    
    # Filtrace datasetu pouze na požadované pH
    filtered_samples = [item for item in dataset.samples if abs(item[1] - TARGET_PH) < 0.1]
    dataset.samples = filtered_samples[:NUM_SAMPLES] # Omezíme na požadovaný počet
    
    actual_num_samples = len(dataset.samples)
    print(f"Nalezeno {actual_num_samples} reálných obrázků pro pH {TARGET_PH}.")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    for real_batch, _ in tqdm(dataloader, desc="Reálná data"):
        real_batch = real_batch.to(DEVICE)
        real_images_fid = prepare_images_for_fid(real_batch)
        fid.update(real_images_fid, real=True)


    #Zpracování GENEROVANÝCH obrázků
    print(f"\n--- Generování a extrakce rysů syntetických obrázků (pH {TARGET_PH}) ---")
    num_batches = (actual_num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for _ in tqdm(range(num_batches), desc="Syntetická data"):
        current_batch_size = min(BATCH_SIZE, actual_num_samples)
        actual_num_samples -= current_batch_size

        fake_batch = sample(model, pH_query=TARGET_PH, num_samples=current_batch_size, num_steps=100)
        
        fake_images_fid = prepare_images_for_fid(fake_batch)
        fid.update(fake_images_fid, real=False)

    fid_score = fid.compute()
    print("=" * 50)
    print(f"Výsledné FID skóre (pro pH {TARGET_PH}): {fid_score.item():.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()