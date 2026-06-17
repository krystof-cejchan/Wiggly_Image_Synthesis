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
EVAL_INTERVAL = 1000  
PATIENCE = 10        
MIN_DELTA = 1e-5     

# ┌─ REVIEW ─────────────────────────────────────────────
#   to samé co předtím: dal bych konstanty i funkci do jednoho config.py a všude importoval:
#     from config import PH_MIN, PH_MAX, normalize_pH
# └──────────────────────────────────────────────────────
# Normalizace na základě tvých dat
PH_MIN, PH_MAX = 5.8, 8.8

def normalize_pH(pH):
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

@torch.no_grad()
def evaluate(model, dataloader):
    """Spočítá Flow Matching MSE na validačním datasetu."""
    model.eval()
    total_loss = 0.0
    
    for x_batch, pH_batch in dataloader:
        x1 = x_batch.to(DEVICE)
        pH = normalize_pH(pH_batch.to(DEVICE).float())
        
        # ┌─ REVIEW ─────────────────────────────────────────────
        #   při každém volání evaluate() se losuje ČERSTVÝ šum x0 a
        #   náhodné t. Validační loss tím šumí sama od sebe a early-stopping
        #   níže ji porovnává s MIN_DELTA=1e-5 → porovnáváš šum, ne zlepšení
        #   modelu. Rozhodnutí o "nejlepším modelu" je pak náhodné.
        #   NÁHRADA: udělej validaci DETERMINISTICKOU — zafixuj seed:
        #     gen = torch.Generator(device=DEVICE).manual_seed(0)
        #     x0 = torch.randn(x1.shape, generator=gen, device=DEVICE)
        #     t  = torch.rand(x1.shape[0], generator=gen, device=DEVICE)
        #   nebo (lépe) zprůměruj loss přes K pevných losování t/šumu, ať je
        #   odhad hladký a srovnatelný mezi kroky.
        # └──────────────────────────────────────────────────────
        # Simulace stejného procesu jako při tréninku
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=DEVICE)
        
        t_expand = t.view(-1, 1, 1, 1)
        xt = (1 - t_expand) * x0 + t_expand * x1
        target = x1 - x0
        
        pred = model(xt, t, pH)
        loss = F.mse_loss(pred, target)
        total_loss += loss.item()
        
    model.train()
    return total_loss / len(dataloader)

def main():
    # ┌─ REVIEW ─────────────────────────────────────────────
    #   žádný globální seed = běhy nejdou zopakovat, nelze spravedlivě
    #   porovnat dva experimenty (jiný start = jiný výsledek).
    #   NÁHRADA: na začátku main() nastav seed (kvůli DataLoaderu i per-worker),
    #   ať je trénink reprodukovatelný:
    #     torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    # └──────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    
    train_dataset = MicrotubuleDataset(DATA_DIR, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    val_dataset = MicrotubuleDataset(DATA_DIR, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
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
    
    print(f"{DEVICE}.")
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
            
            # ┌─ REVIEW ─────────────────────────────────────────────
            #   trénuje se celé v fp32, bez mixed precision. Na
            #   moderní GPU přicházíš o 1.5–2× rychlost skoro zadarmo.
            #   NÁHRADA: obal forward/loss do autocastu (bf16) a uprav krok:
            #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            #         pred = model(xt, t, pH_input)
            #         loss = F.mse_loss(pred, target)
            #   bf16 nepotřebuje GradScaler; pro fp16 přidej scaler + unscale_
            #   PŘED clip_grad_norm_.
            # └──────────────────────────────────────────────────────
            pred = model(xt, t, pH_input)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
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
                    # Uložíme nejlepší model
                    torch.save(ema_model.state_dict(), "checkpoints/cfm_best_ema.pt")
                    print("  -> Nový nejlepší model uložen!")
                else:
                    epochs_without_improvement += 1
                    print(f"  -> Bez zlepšení. Patience: {epochs_without_improvement}/{PATIENCE}")
                    
                if epochs_without_improvement >= PATIENCE:
                    print(f"\n[Early Stopping] Trénink ukončen v kroku {step}. Validační chyba se nezlepšila {PATIENCE} po sobě jdoucích kontrol.")
                    return  # Ukončí funkci main() a tím i skript
                    
            elif step % 100 == 0:
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss: {loss.item():.4f}")
                
            step += 1

    torch.save(ema_model.state_dict(), "checkpoints/cfm_final_ema.pt")
    print("Trénink dokončen.")

if __name__ == "__main__":
    main()