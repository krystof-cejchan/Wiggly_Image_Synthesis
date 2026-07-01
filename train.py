import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
import random
import numpy as np

from config import PH_MIN, PH_MAX, DEVICE
import torchvision.transforms.v2 as T
from model import ConditionalUNet
from dataset import MicrotubuleDataset

# Nastavení a konstanty
DATA_DIR = "data/cropped/cropped_output"
BATCH_SIZE = 64
LR = 1e-4
ITERATIONS = 100_000
CFG_DROPOUT = 0.2
EVAL_INTERVAL = 500  
PATIENCE = 5        
MIN_DELTA = 1e-5     
SEED = 42 # Pevný seed pro reprodukovatelnost
TRAIN_SIZES = [(128, 128), (64, 256), (256, 64), (48, 384), (80, 192)]

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

def safe_mirror_pad(img_tensor, target_h, target_w):
    """
    Zrcadlově množí obrázek, dokud není dostatečně velký.
    Vyhne se tím limitům PyTorche a nevytváří umělé pruhy.
    """
    _, h, w = img_tensor.shape
    
    # Zrcadlení na výšku
    while h < target_h:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[1])], dim=1)
        h = img_tensor.shape[1]
        
    # Zrcadlení na šířku
    while w < target_w:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[2])], dim=2)
        w = img_tensor.shape[2]
        
    return img_tensor

def dynamic_collate_fn(batch):
    """Pro každý batch náhodně vybere poměr stran a ořízne předpřipravené obrázky."""
    target_h, target_w = random.choice(TRAIN_SIZES)
    
    # Odebráno pad_if_needed a padding_mode
    transform = T.Compose([
        T.RandomCrop((target_h, target_w)),
        T.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    images = []
    for item in batch:
        img = item[0]
        # Nejprve obrázek bezpečně nafoukneme zrcadlením
        img_padded = safe_mirror_pad(img, target_h, target_w)
        # Následně z něj vyřízneme dynamický rozměr
        images.append(transform(img_padded))
        
    phs = [item[1] for item in batch]
    return torch.stack(images), torch.stack(phs)

def val_collate_fn(batch):
    """Validace běží na stabilním rozlišení pro konzistentní výpočet loss."""
    target_h, target_w = 128, 128
    transform = T.RandomCrop((target_h, target_w))
    
    images = []
    for item in batch:
        img = item[0]
        img_padded = safe_mirror_pad(img, target_h, target_w)
        images.append(transform(img_padded))
        
    phs = [item[1] for item in batch]
    return torch.stack(images), torch.stack(phs)

@torch.no_grad()
def evaluate(model, dataloader, num_noise_samples=3):
    """
    Spočítá Flow Matching MSE na validačním datasetu.
    Validace je nyní deterministická (používá pevný generátor) a 
    pro každý batch průměruje loss přes více náhodných losování (num_noise_samples),
    aby byla validační křivka hladká a spolehlivá pro early stopping.
    """
    model.eval()
    total_loss = 0.0
    
    # Lokální generátor zajišťuje, že pro stejný batch vygenerujeme stejný šum každou epochu
    eval_gen = torch.Generator(device=DEVICE)
    eval_gen.manual_seed(12345) 
    
    for x_batch, pH_batch in dataloader:
        x1 = x_batch.to(DEVICE)
        pH = normalize_pH(pH_batch.to(DEVICE).float())
        
        batch_loss = 0.0
        
        # Zprůměrování přes K losování pro stabilnější odhad ztráty
        for _ in range(num_noise_samples):
            x0 = torch.randn(x1.shape, generator=eval_gen, device=DEVICE)
            t = torch.rand(x1.shape[0], generator=eval_gen, device=DEVICE)
            
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0
            
            # Pro jistotu použijeme autocast i u validace
            with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
                pred = model(xt, t, pH)
                loss = F.mse_loss(pred, target)
                
            batch_loss += loss.item()
            
        total_loss += batch_loss / num_noise_samples
        
    model.train()
    return total_loss / len(dataloader)

def seed_worker(worker_id):
    """Seed pro jednotlivé workery DataLoaderu."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(SEED) # 1. Globální seed
    os.makedirs("checkpoints", exist_ok=True)
    
    # Nastavení generátoru pro DataLoader, aby bylo míchání reprodukovatelné
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_dataset = MicrotubuleDataset(DATA_DIR, is_train=True)
    val_dataset = MicrotubuleDataset(DATA_DIR, is_train=False)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
        collate_fn=dynamic_collate_fn  # Přidán collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, worker_init_fn=seed_worker,
        collate_fn=val_collate_fn      # Přidán collate_fn
    )

    
    model = ConditionalUNet().to(DEVICE)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=ITERATIONS)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    model.train()
    step = 0
    
    print(f"Zařízení: {DEVICE}.")
    print(f"Trénovacích obrázků: {len(train_dataset)}, Validačních: {len(val_dataset)}")

    while step < ITERATIONS:
        for x_batch, pH_batch in train_dataloader:
            if step >= ITERATIONS:
                break
                
            x1 = x_batch.to(DEVICE)
            pH = normalize_pH(pH_batch.to(DEVICE).float())
            
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=DEVICE)
            
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0
            
            drop_mask = torch.rand(x1.shape[0], device=DEVICE) < CFG_DROPOUT
            pH_input = torch.where(drop_mask, torch.full_like(pH, float("nan")), pH)
            
            optimizer.zero_grad()
            
            # 2. Mixed Precision (Autocast do bfloat16) pro obrovské zrychlení
            # Pokud na tvém hardwaru nefunguje bfloat16, změň na torch.float16 a přidej GradScaler.
            device_type_autocast = "cuda" if "cuda" in DEVICE else "cpu"
            with torch.autocast(device_type=device_type_autocast, dtype=torch.bfloat16):
                pred = model(xt, t, pH_input)
                loss = F.mse_loss(pred, target)
            
            # U bfloat16 voláme zpětný průchod normálně (bez scaleru)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(0.9999).add_(p, alpha=0.0001)
            
            # early stopping
            if step > 0 and step % EVAL_INTERVAL == 0:
                val_loss = evaluate(ema_model, val_dataloader)
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                
                if val_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(ema_model.state_dict(), "checkpoints/cfm_best_ema.pt")
                else:
                    epochs_without_improvement += 1
                    
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping aktivován na kroku {step}. Trénink ukončen.")
                    return  
                    
            elif step % 100 == 0:
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss: {loss.item():.4f}")
                
            step += 1

    torch.save(ema_model.state_dict(), "checkpoints/cfm_final_ema.pt")
    print("Trénink dokončen.")

if __name__ == "__main__":
    main()