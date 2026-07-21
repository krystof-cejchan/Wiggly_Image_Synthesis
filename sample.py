import os
import torch
import torchvision.utils as vutils
from model import ConditionalUNet
from config import PH_MIN, PH_MAX, DEVICE

def normalize_pH(pH):
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

@torch.no_grad()
def sample(model, pH_query, num_samples=1, num_steps=1000, cfg_scale=2.0, seed=None,
           solver="heun", guidance_rescale=0.7):
    if seed is not None:
        torch.manual_seed(seed)

    pH_norm = normalize_pH(torch.tensor([pH_query] * num_samples)).to(DEVICE)
    pH_null = torch.full((num_samples,), float("nan"), device=DEVICE)

    def compute_v_cfg(x_in, step_idx):
        t = torch.full((num_samples,), step_idx / num_steps, device=DEVICE)
        v_cond = model(x_in, t, pH_norm)
        v_uncond = model(x_in, t, pH_null)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

        # CFG rescale (Lin et al., "Common Diffusion Noise Schedules and Sample
        # Steps are Flawed") instead of a hard elementwise clamp: rescale toward
        # the conditional branch's std, which tames divergence from aggressive
        # extrapolation without the clipping artifacts a fixed [-5,5] bound causes.
        std_cond = v_cond.std(dim=(1, 2, 3), keepdim=True)
        std_cfg = v_cfg.std(dim=(1, 2, 3), keepdim=True)
        v_rescaled = v_cfg * (std_cond / std_cfg.clamp(min=1e-8))
        return guidance_rescale * v_rescaled + (1 - guidance_rescale) * v_cfg

    x = torch.randn(num_samples, 1, 128, 128, device=DEVICE)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        v1 = compute_v_cfg(x, i)

        if solver == "euler":
            x = x + v1 * dt
        elif solver == "heun":
            x_pred = x + v1 * dt
            v2 = compute_v_cfg(x_pred, i + 1)
            x = x + 0.5 * (v1 + v2) * dt
        else:
            raise ValueError(f"Unknown solver: {solver!r} (expected 'euler' or 'heun')")

    # Denormalizace [-1, 1] → [0, 1]
    return (x.clamp(-1, 1) + 1) / 2

def main():
    checkpoint_path = "checkpoints/cfm_best_ema3.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} nenalezen!")
        return
        
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs", exist_ok=True)
    
    target_phs = [5.8, 6.4, 7.0, 7.4, 8.2, 8.8]
    
    for ph in target_phs:
        print(f"Generuji vzorky pro pH = {ph} ...")
        samples = sample(model, pH_query=ph)
        
        # Uložení jako mřížka 4x4
        save_path = f"outputs/sample_pH_{ph}.png"
        vutils.save_image(samples, save_path, nrow=4)
        print(f"Uloženo do: {save_path}")

if __name__ == "__main__":
    main()
