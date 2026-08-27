import os
import argparse
import torch
import torchvision.utils as vutils
from model import ConditionalUNet
from config import PH_MIN, PH_MAX, DEVICE
from ph_control import normalize_pH, velocity_for_pH, predicted_waviness_native, describe as describe_pH

@torch.no_grad()
def sample(model, pH_query, num_samples=1, num_steps=1000, cfg_scale=2.0, seed=None,
           solver="heun", guidance_rescale=0.7, geometry_mode="embedding"):
    """Generate samples at any pH.

    geometry_mode="embedding" (default): pH_query may lie outside the trained range;
    velocity_for_pH extrapolates along the acidic->alkaline direction instead of feeding an
    out-of-range value into the periodic embedding. Validated for BELOW-range generation
    only (see ph_control.py's docstring) - there is still no correct path above range
    through this mode.
    geometry_mode="native": drives geometry through the model's own waviness conditioning
    instead (model.py's WavinessEmbedding), in EITHER direction - needs a checkpoint
    trained with it, NOT YET VALIDATED (no such checkpoint exists as of this writing).
    Free generation has no reference-image anchor for this to fight the way img2img.py's
    editing does, so there is a real chance it works both directions here where the warp-
    based mechanism in ph_warp.py needed one direction handled geometrically instead - but
    that is a hypothesis, not yet a measurement.
    Classifier-free guidance applies on top of either mode, unchanged.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if geometry_mode not in ("embedding", "native"):
        raise ValueError(f"Unknown geometry_mode: {geometry_mode!r} (expected 'embedding' or 'native')")
    target_waviness = predicted_waviness_native(pH_query) if geometry_mode == "native" else None

    pH_null = torch.full((num_samples,), float("nan"), device=DEVICE)

    def compute_v_cfg(x_in, step_idx):
        t = torch.full((num_samples,), step_idx / num_steps, device=DEVICE)
        v_cond = velocity_for_pH(model, x_in, t, pH_query, waviness=target_waviness)
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
    parser = argparse.ArgumentParser(description="Generate microtubule samples at any pH")
    parser.add_argument("--checkpoint", default="checkpoints/cfm_best_ema.pt")
    parser.add_argument("--pH", type=float, nargs="+",
                        default=[4.0,4.4,4.8,5.2,5.8, 6.4, 7.0, 7.4, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8, 12.4],
                        help=f"pH values to generate. Outside the trained range "
                             f"{PH_MIN}-{PH_MAX} the model extrapolates; see ph_control.py")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--geometry_mode", type=str, default="embedding", choices=["embedding", "native"],
                        help="'embedding' (default) is the validated velocity-extrapolation "
                             "mechanism, correct below range only. 'native' drives geometry "
                             "through the model's own waviness conditioning instead - requires "
                             "a checkpoint trained with it; NOT YET VALIDATED.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} nenalezen!")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    os.makedirs("outputs", exist_ok=True)

    for ph in args.pH:
        print(f"Generuji vzorky pro pH = {ph} ... ({describe_pH(ph, geometry_mode=args.geometry_mode)})")
        samples = sample(model, pH_query=ph, num_samples=args.num_samples,
                         num_steps=args.num_steps, cfg_scale=args.cfg_scale, seed=args.seed,
                         geometry_mode=args.geometry_mode)

        save_path = f"outputs/sample_pH_{ph}.png"
        vutils.save_image(samples, save_path, nrow=4)
        print(f"Uloženo do: {save_path}")

if __name__ == "__main__":
    main()
