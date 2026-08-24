"""Generate every figure used in the flow-matching study document.

Two kinds of figure:
  * 2D toy panels - a tiny velocity network trained here on a sine-shaped target, so the
    maths in the document is illustrated by a model that actually ran, not a sketch.
  * real panels - interpolation paths and samples from the project's own v2 checkpoint.

Rasters carry no baked-in text (labels live in the HTML so they follow the page theme) and
plots are saved with a transparent background in a mid grey that reads on light and dark.
"""
import os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/krystof/prg/Wiggly_Image_Synthesis")
os.chdir("/home/krystof/prg/Wiggly_Image_Synthesis")
from model import ConditionalUNet
from img2img import load_and_preprocess_image
from config import DEVICE

OUT = "/tmp/claude-1001/-home-krystof-prg-Wiggly-Image-Synthesis/94a0a51f-8d0c-4334-9c62-454de430eec2/scratchpad/figs"
os.makedirs(OUT, exist_ok=True)

GREY = "#8c8c8c"
TEAL = "#0f9b8e"
AMBER = "#d4831f"
PLUM = "#8b5fbf"
matplotlib.rcParams.update({
    "text.color": GREY, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "axes.edgecolor": GREY, "axes.linewidth": 0.8, "font.size": 9,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------- 2D toy target
def sample_target(n, gen):
    """A wiggly ribbon - a sine curve with thickness. On-theme, and clearly non-Gaussian."""
    x = (torch.rand(n, 1, generator=gen) * 2 - 1) * 2.6
    y = torch.sin(x * 1.9) * 1.15 + torch.randn(n, 1, generator=gen) * 0.13
    return torch.cat([x, y], dim=1)


class TinyVelocity(nn.Module):
    """v_theta(x, t): the 2D analogue of the project's UNet."""
    def __init__(self, h=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def train_toy():
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    net = TinyVelocity()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(4000):
        x1 = sample_target(512, gen)
        x0 = torch.randn(512, 2, generator=gen)
        t = torch.rand(512, 1, generator=gen)
        xt = (1 - t) * x0 + t * x1
        loss = ((net(xt, t) - (x1 - x0)) ** 2).mean()   # exactly the project's training loss
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"toy velocity net trained, final loss {loss.item():.4f}")
    return net, gen


@torch.no_grad()
def integrate(net, x, steps=100, keep=None):
    """Euler-integrate dx/dt = v(x,t) from t=0 to t=1, optionally recording snapshots."""
    dt, snaps = 1.0 / steps, {}
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i / steps)
        if keep and i in keep:
            snaps[i] = x.clone()
        x = x + net(x, t) * dt
    return x, snaps


def fig_conditional_paths(gen):
    """Each training pair is one straight line. That IS the conditional probability path."""
    torch.manual_seed(3)
    x1 = sample_target(14, gen); x0 = torch.randn(14, 2, generator=gen)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    tgt = sample_target(1500, gen)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=2, c=TEAL, alpha=0.22, linewidths=0)
    noise = torch.randn(1500, 2, generator=gen)
    ax.scatter(noise[:, 0], noise[:, 1], s=2, c=GREY, alpha=0.18, linewidths=0)
    for a, b in zip(x0, x1):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=AMBER, lw=1.1, alpha=0.85)
    ax.scatter(x0[:, 0], x0[:, 1], s=22, c=GREY, zorder=3, edgecolors="none")
    ax.scatter(x1[:, 0], x1[:, 1], s=22, c=TEAL, zorder=3, edgecolors="none")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.2, 3.2); bare(ax)
    save(fig, "toy_conditional")


def fig_marginal_average(gen):
    """Many conditional velocities pass through one point; the net learns their average."""
    torch.manual_seed(7)
    xq = torch.tensor([0.55, 0.35])
    x1 = sample_target(9000, gen); x0 = torch.randn(9000, 2, generator=gen)
    t = 0.5
    xt = (1 - t) * x0 + t * x1
    near = ((xt - xq).norm(dim=1) < 0.30).nonzero().flatten()[:26]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    tgt = sample_target(1500, gen)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=2, c=TEAL, alpha=0.18, linewidths=0)
    v = (x1 - x0)[near]
    for vi in v:
        ax.arrow(float(xq[0]), float(xq[1]), float(vi[0]) * 0.22, float(vi[1]) * 0.22,
                 color=AMBER, alpha=0.5, lw=0.9, head_width=0.07, length_includes_head=True)
    vm = v.mean(0)
    ax.arrow(float(xq[0]), float(xq[1]), float(vm[0]) * 0.22, float(vm[1]) * 0.22,
             color=PLUM, lw=2.6, head_width=0.17, length_includes_head=True, zorder=5)
    ax.scatter([xq[0]], [xq[1]], s=55, c=PLUM, zorder=6, edgecolors="none")
    ax.set_xlim(-3.0, 3.4); ax.set_ylim(-2.6, 2.8); bare(ax)
    save(fig, "toy_marginal")


def fig_learned_field_and_flow(net, gen):
    """The learned field, and the trajectories you get by following it."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    g = torch.linspace(-3.0, 3.0, 17)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    with torch.no_grad():
        v = net(pts, torch.full((pts.shape[0], 1), 0.5))
    axes[0].quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], color=GREY, alpha=0.75,
                   width=0.004, scale=45)
    tgt = sample_target(1500, gen)
    axes[0].scatter(tgt[:, 0], tgt[:, 1], s=2, c=TEAL, alpha=0.25, linewidths=0)
    axes[0].set_xlim(-3.2, 3.2); axes[0].set_ylim(-3.2, 3.2); bare(axes[0])

    torch.manual_seed(11)
    start = torch.randn(220, 2)
    keep = list(range(0, 100, 4))
    end, snaps = integrate(net, start.clone(), steps=100, keep=set(keep))
    traj = torch.stack([snaps[k] for k in sorted(snaps)] + [end])
    for j in range(traj.shape[1]):
        axes[1].plot(traj[:, j, 0], traj[:, j, 1], color=AMBER, lw=0.5, alpha=0.35)
    axes[1].scatter(start[:, 0], start[:, 1], s=4, c=GREY, alpha=0.6, linewidths=0)
    axes[1].scatter(end[:, 0], end[:, 1], s=6, c=TEAL, linewidths=0)
    axes[1].set_xlim(-3.2, 3.2); axes[1].set_ylim(-3.2, 3.2); bare(axes[1])
    save(fig, "toy_field_flow")


def fig_straight_vs_curved():
    """Flow matching's linear path vs a diffusion-style cosine schedule, same endpoints."""
    x0 = np.array([-1.9, -1.35]); x1 = np.array([1.9, 1.15])
    t = np.linspace(0, 1, 240)
    lin = (1 - t)[:, None] * x0 + t[:, None] * x1
    ang = t * math.pi / 2
    cur = np.cos(ang)[:, None] * x0 + np.sin(ang)[:, None] * x1
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(lin[:, 0], lin[:, 1], color=TEAL, lw=2.4)
    ax.plot(cur[:, 0], cur[:, 1], color=AMBER, lw=2.4, ls=(0, (5, 3)))
    for frac in [0.25, 0.5, 0.75]:
        i = int(frac * 239)
        ax.scatter(*lin[i], s=26, c=TEAL, zorder=4, linewidths=0)
        ax.scatter(*cur[i], s=26, c=AMBER, zorder=4, linewidths=0)
    ax.scatter(*x0, s=70, c=GREY, zorder=5, linewidths=0)
    ax.scatter(*x1, s=70, c=PLUM, zorder=5, linewidths=0)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.0, 2.0); bare(ax)
    save(fig, "straight_vs_curved")


# ----------------------------------------------------------------- real panels
def strip(arr, name):
    """Save a bare grayscale raster - no axes, no text."""
    fig = plt.figure(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(arr, cmap="gray", vmin=0, vmax=1); ax.axis("off")
    fig.savefig(f"{OUT}/{name}.png", dpi=150, transparent=True)
    plt.close(fig)


def fig_real_interpolation():
    """x_t = (1-t)x0 + t x1 on an actual crop: what the network sees during training."""
    ref, size = load_and_preprocess_image(
        "data/cropped/cropped_output/8.8/20260219_003_Ch1_pos2_pH8_frame0000_crop04.png")
    w, h = size
    x1 = ref[0, 0, :h, :w].cpu()
    torch.manual_seed(4)
    x0 = torch.randn_like(x1)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xt = (1 - t) * x0 + t * x1
        strip(((xt.clamp(-1, 1) + 1) / 2).numpy(), f"interp_{int(t*100):03d}")
    print("wrote interpolation strips")


@torch.no_grad()
def fig_model_trajectory(model):
    """Integrate the real checkpoint from pure noise, snapshotting along the way."""
    steps, cfg = 160, 2.0
    for ph, tag in [(5.8, "low"), (8.8, "high")]:
        torch.manual_seed(20)
        phn = torch.tensor([2 * (ph - 5.8) / (8.8 - 5.8) - 1.0], device=DEVICE)
        phnull = torch.full((1,), float("nan"), device=DEVICE)
        x = torch.randn(1, 1, 128, 128, device=DEVICE)
        snap_at = {0: "000", 40: "025", 80: "050", 120: "075"}
        for i in range(steps):
            if i in snap_at:
                strip(((x[0, 0].clamp(-1, 1).cpu() + 1) / 2).numpy(),
                      f"traj_{tag}_{snap_at[i]}")
            t = torch.full((1,), i / steps, device=DEVICE)
            vc = model(x, t, phn); vu = model(x, t, phnull)
            v = vu + cfg * (vc - vu)
            v = v * (vc.std() / v.std().clamp(min=1e-8)) * 0.7 + v * 0.3
            x = x + v / steps
        strip(((x[0, 0].clamp(-1, 1).cpu() + 1) / 2).numpy(), f"traj_{tag}_100")
        print(f"wrote trajectory pH {ph}")


def main():
    net, gen = train_toy()
    fig_conditional_paths(gen)
    fig_marginal_average(gen)
    fig_learned_field_and_flow(net, gen)
    fig_straight_vs_curved()
    fig_real_interpolation()
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/cfm_best_emav2.pt", map_location=DEVICE))
    model.eval()
    fig_model_trajectory(model)
    print("\nall figures in", OUT)


if __name__ == "__main__":
    main()
