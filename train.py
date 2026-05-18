import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

from model import ConditionalUNet
from dataset import MicrotubuleDataset

# Nastavení a konstanty
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/cropped/cropped_output"
BATCH_SIZE = 32
LR = 1e-4
ITERATIONS = 100_000
CFG_DROPOUT = 0.1

# Normalizace na základě tvých dat
PH_MIN, PH_MAX = 5.8, 8.8

def normalize_pH(pH):
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = MicrotubuleDataset(DATA_DIR, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    model = ConditionalUNet().to(DEVICE)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=ITERATIONS)

    model.train()
    step = 0
    
    print(f"Zahajuji trénink na {DEVICE}. Dat velikost: {len(dataset)}")

    while step < ITERATIONS:
        for x_batch, pH_batch in dataloader:
            if step >= ITERATIONS:
                break
                
            x1 = x_batch.to(DEVICE)
            pH = normalize_pH(pH_batch.to(DEVICE).float())
            
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=DEVICE)
            
            # OT path
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0
            
            # CFG dropout
            drop_mask = torch.rand(x1.shape[0], device=DEVICE) < CFG_DROPOUT
            pH_input = torch.where(drop_mask, torch.full_like(pH, float("nan")), pH)
            
            pred = model(xt, t, pH_input)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # EMA Update
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(0.9999).add_(p, alpha=0.0001)
            
            if step % 100 == 0:
                print(f"Krok: {step:06d}/{ITERATIONS} | Loss: {loss.item():.4f}")
                
            if step % 5000 == 0 and step > 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"checkpoints/cfm_step_{step}.pt")
                
            step += 1

    # Uložení finálního modelu
    torch.save(ema_model.state_dict(), "checkpoints/cfm_final_ema.pt")
    print("Trénink dokončen.")

if __name__ == "__main__":
    main()