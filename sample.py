import os
import torch
import torchvision.utils as vutils
from model import ConditionalUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PH_MIN, PH_MAX = 5.8, 8.8

def normalize_pH(pH):
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

@torch.no_grad()
def sample(model, pH_query, num_samples=16, num_steps=50, cfg_scale=3.0, seed=42):
    if seed is not None:
        torch.manual_seed(seed)
        
    pH_norm = normalize_pH(torch.tensor([pH_query] * num_samples)).to(DEVICE)
    pH_null = torch.full((num_samples,), float("nan"), device=DEVICE)
    
    x = torch.randn(num_samples, 1, 128, 128, device=DEVICE)   
    for i in range(num_steps):
        t = torch.full((num_samples,), i / num_steps, device=DEVICE)
        
        v_cond = model(x, t, pH_norm)
        v_uncond = model(x, t, pH_null)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        x = x + v_cfg * (1.0 / num_steps)
    
    # Denormalizace [-1, 1] → [0, 1]
    return (x.clamp(-1, 1) + 1) / 2

def main():
    checkpoint_path = "checkpoints/cfm_best_ema.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} nenalezen!")
        return
        
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs", exist_ok=True)
    
    # Testujeme inferenci pro několik různých pH vč. neviděných mezihodnot
    target_phs = [5.8, 6.4, 7.0, 7.4, 8.2, 8.8]
    
    for ph in target_phs:
        print(f"Generuji vzorky pro pH = {ph} ...")
        samples = sample(model, pH_query=ph, num_samples=16, cfg_scale=3.0)
        
        # Uložení jako mřížka 4x4
        save_path = f"outputs/sample_pH_{ph}.png"
        vutils.save_image(samples, save_path, nrow=4)
        print(f"Uloženo do: {save_path}")

if __name__ == "__main__":
    main()