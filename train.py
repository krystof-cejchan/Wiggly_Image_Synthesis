import os
import csv
import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")  # training usually runs headless / over ssh - never try to open a window
import matplotlib.pyplot as plt

from config import PH_MIN, PH_MAX, DEVICE
import torchvision.transforms.v2 as T
from model import ConditionalUNet
from dataset import MicrotubuleDataset

# hyperparameters
DATA_DIR = "data/cropped/cropped_output"
BATCH_SIZE = 16          
ACCUMULATION_STEPS = 4  
LR = 1e-4
ITERATIONS = 100_000
CFG_DROPOUT = 0.2
PH_JITTER_STD = 0.08  # pH buckets are unevenly spaced (0.2-1.0 apart) - jittering the label
                      # each step teaches the model that nearby pH values should look similar,
                      # the interpolation signal the discrete buckets alone don't provide. Keep
                      # this small relative to the TIGHTEST real gap (7.2->7.4 is only 0.2): a
                      # wider std (previously 0.15) bleeds across adjacent buckets in the densely
                      # packed 5.8-7.8 range and flattens the pH->waviness response there, while
                      # barely reaching into the one big outlier gap (7.8->8.8) anyway.
EMA_DECAY = 0.9999
EVAL_INTERVAL = 500
PATIENCE = 10        # val loss is noisy (fluctuates ~1e-2 while MIN_DELTA is 1e-5), so a
                     # short patience stops on noise rather than on a real plateau
MIN_DELTA = 1e-5
SEED = 42
# Crops are thin landscape strips (median 292x42). Training sizes must stay close to that
# aspect ratio: safe_mirror_pad reaches a target by repeatedly reflecting the image, so a
# 384x384 target turns one 42px-tall fiber into 16 stacked mirror copies and the model
# learns to generate that tiled hall-of-mirrors instead of a single fiber.
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
   multiple mirror padding to ensure the image tensor is at least target_h x target_w in size.
    If the image is already larger than the target dimensions, it will be returned unchanged.
    """
    _, h, w = img_tensor.shape
    
    # Mirror padding to ensure the image tensor is at least target_h x target_w in size
    while h < target_h:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[1])], dim=1)
        h = img_tensor.shape[1]
        
    # Mirror padding to ensure the image tensor is at least target_h x target_w in size
    while w < target_w:
        img_tensor = torch.cat([img_tensor, img_tensor.flip(dims=[2])], dim=2)
        w = img_tensor.shape[2]
        
    return img_tensor

def dynamic_collate_fn(batch):
    """For each batch, randomly select an aspect ratio and crop the pre-prepared images."""
    target_h, target_w = random.choice(TRAIN_SIZES)
    
    transform = T.Compose([
        T.RandomCrop((target_h, target_w)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    images = []
    for item in batch:
        img = item[0]
        # Pad the image to ensure it is at least target_h x target_w in size
        img_padded = safe_mirror_pad(img, target_h, target_w)
        # Apply the random crop and color jitter
        images.append(transform(img_padded))
        
    phs = [item[1] for item in batch]
    return torch.stack(images), torch.stack(phs)

def val_collate_fn(batch):
    """For validation, we will use a fixed crop size (128x128) and no color jittering.

    CenterCrop, not RandomCrop: the crop must be identical on every evaluate() call or the
    val loss picks up crop-to-crop noise (~1e-2) that dwarfs MIN_DELTA (1e-5), which makes
    early stopping fire on noise instead of on a real plateau.
    """
    target_h, target_w = 128, 128
    transform = T.CenterCrop((target_h, target_w))

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
    Calculates Flow Matching MSE on the validation dataset.
    Fully deterministic: a fixed generator supplies x0/t, and val_collate_fn center-crops
    (rather than random-crops) so the same pixels are scored every time. For each batch the
    loss is averaged over multiple noise samplings (num_noise_samples) to smooth the curve.
    """
    model.eval()
    total_loss = 0.0
    
    # Set a fixed generator for deterministic validation
    eval_gen = torch.Generator(device=DEVICE)
    eval_gen.manual_seed(12345) 
    
    for x_batch, pH_batch in dataloader:
        x1 = x_batch.to(DEVICE)
        pH = normalize_pH(pH_batch.to(DEVICE).float())
        
        batch_loss = 0.0
        
        # Average the loss over multiple random samplings for each batch
        for _ in range(num_noise_samples):
            x0 = torch.randn(x1.shape, generator=eval_gen, device=DEVICE)
            t = torch.rand(x1.shape[0], generator=eval_gen, device=DEVICE)
            
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0
            
            
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                pred = model(xt, t, pH)
                loss = F.mse_loss(pred, target)
                
            batch_loss += loss.item()
            
        total_loss += batch_loss / num_noise_samples
        
    model.train()
    return total_loss / len(dataloader)

def save_loss_history(train_steps, train_losses, val_steps, val_losses, out_path):
    """Dump the raw curves next to the plot so they can be re-plotted or compared across runs."""
    val_lookup = dict(zip(val_steps, val_losses))
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "train_loss_live", "val_loss_ema"])
        for s, tl in zip(train_steps, train_losses):
            writer.writerow([s, f"{tl:.6f}", f"{val_lookup[s]:.6f}" if s in val_lookup else ""])
        # val is sampled on a coarser grid than train, so emit any val-only steps too
        for s in val_steps:
            if s not in set(train_steps):
                writer.writerow([s, "", f"{val_lookup[s]:.6f}"])


def plot_loss_history(train_steps, train_losses, val_steps, val_losses, best_step,
                      out_path="outputs/training_loss.png"):
    """Save the train/val loss curves once training ends (either normally or via early stop)."""
    if not train_steps:
        print("No loss history recorded - skipping plot.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # the per-step train loss is very noisy (single batch, random t and noise), so overlay a
    # running mean - otherwise the trend is invisible under the jitter
    window = max(1, min(21, len(train_losses) // 10))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(np.array(train_losses), kernel, mode="valid")
        smooth_steps = train_steps[window - 1:]
    else:
        smoothed, smooth_steps = None, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, log_scale in zip(axes, (False, True)):
        ax.plot(train_steps, train_losses, color="tab:blue", alpha=0.25, lw=0.8,
                label="Train, live model (raw)")
        if smoothed is not None:
            ax.plot(smooth_steps, smoothed, color="tab:blue", lw=1.8,
                    label=f"Train, live model (mean of {window})")
        if val_steps:
            ax.plot(val_steps, val_losses, color="tab:red", lw=1.8, marker="o", ms=3,
                    label="Val, EMA model")
        if best_step is not None:
            ax.axvline(best_step, color="gray", ls="--", lw=1.2,
                       label=f"best checkpoint (step {best_step})")
        ax.set_xlabel("training step")
        ax.set_ylabel("flow-matching MSE")
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training / validation loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Loss curve saved to: {out_path}")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    save_loss_history(train_steps, train_losses, val_steps, val_losses, csv_path)
    print(f"Loss history saved to: {csv_path}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(SEED) 
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set up DataLoaders with dynamic collate functions for training and validation
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_dataset = MicrotubuleDataset(DATA_DIR, is_train=True)
    val_dataset = MicrotubuleDataset(DATA_DIR, is_train=False)

    # pH buckets are heavily imbalanced (e.g. 36 vs 136 images) - weight samples by
    # inverse pH-bucket frequency so each pH gets roughly equal gradient signal.
    train_phs = [ph for _, ph in train_dataset.samples]
    ph_counts = Counter(train_phs)
    train_sample_weights = [1.0 / ph_counts[ph] for ph in train_phs]
    train_sampler = WeightedRandomSampler(
        train_sample_weights, num_samples=len(train_dataset), replacement=True, generator=g
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=4, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
        collate_fn=dynamic_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, worker_init_fn=seed_worker,
        collate_fn=val_collate_fn     
    )

    
    model = ConditionalUNet().to(DEVICE)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # T_max counts scheduler.step() calls, which happen once per OPTIMIZER step, not once
    # per iteration - with T_max=ITERATIONS the cosine would only traverse 1/ACCUMULATION_STEPS
    # of its cycle and the LR would still be at ~85% of base when training ends.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, ITERATIONS // ACCUMULATION_STEPS))

    best_val_loss = float('inf')
    best_step = None
    epochs_without_improvement = 0

    # loss history for the end-of-training plot
    train_hist_steps, train_hist_losses = [], []
    val_hist_steps, val_hist_losses = [], []

    model.train()
    step = 0
    ema_updates = 0
    stop_training = False

    print(f"{DEVICE}.")
    print(f"training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")

    while step < ITERATIONS and not stop_training:
        for x_batch, pH_batch in train_dataloader:
            if step >= ITERATIONS:
                break

            x1 = x_batch.to(DEVICE)
            pH_raw = pH_batch.to(DEVICE).float()
            pH_jittered = (pH_raw + torch.randn_like(pH_raw) * PH_JITTER_STD).clamp(PH_MIN, PH_MAX)
            pH = normalize_pH(pH_jittered)

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=DEVICE)
            
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            target = x1 - x0
            
            drop_mask = torch.rand(x1.shape[0], device=DEVICE) < CFG_DROPOUT
            pH_input = torch.where(drop_mask, torch.full_like(pH, float("nan")), pH)
                        
          
            device_type_autocast = "cuda" if "cuda" in DEVICE else "cpu"
            with torch.autocast(device_type=device_type_autocast, dtype=torch.bfloat16):
                pred = model(xt, t, pH_input)
                loss = F.mse_loss(pred, target)
                
            # keep the unscaled value for logging - the scaled one is 1/ACCUMULATION_STEPS of
            # the real loss and isn't comparable to the val loss printed next to it
            train_loss_value = loss.item()
            loss = loss / ACCUMULATION_STEPS

            loss.backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == ITERATIONS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
                # EMA with a warmup ramp. A fixed 0.9999 decay keeps 0.9999^n weight on the
                # RANDOM INITIALIZATION after n updates - and because EMA only updates once per
                # optimizer step, gradient accumulation cuts n by ACCUMULATION_STEPS. At 57.5k
                # iterations with ACCUMULATION_STEPS=4 that left ~24% of the checkpointed EMA
                # weights as pure random init, which is what produced washed-out, near-white
                # samples. Ramping the decay in makes early updates track the live model closely
                # so the init washes out immediately, regardless of the accumulation setting.
                with torch.no_grad():
                    ema_updates += 1
                    decay = min(EMA_DECAY, (1 + ema_updates) / (10 + ema_updates))
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(decay).add_(p, alpha=1 - decay)
            
            # record the train curve on its own cadence, independent of which branch prints
            if step % 100 == 0:
                train_hist_steps.append(step)
                train_hist_losses.append(train_loss_value)

            # early stopping
            if step > 0 and step % EVAL_INTERVAL == 0:
                val_loss = evaluate(ema_model, val_dataloader)
                val_hist_steps.append(step)
                val_hist_losses.append(val_loss)
                # NOTE: train loss is the LIVE model, val loss is the EMA model - early in
                # training the EMA lags badly, so a large gap here means the EMA hasn't caught
                # up yet, not necessarily overfitting.
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss (live): {train_loss_value:.4f} | Val Loss (EMA): {val_loss:.4f}")

                if val_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = val_loss
                    best_step = step
                    epochs_without_improvement = 0
                    torch.save(ema_model.state_dict(), "checkpoints/cfm_best_ema.pt")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping aktivován na kroku {step}. Trénink ukončen.")
                    stop_training = True
                    break

            elif step % 100 == 0:
                print(f"Krok: {step:06d}/{ITERATIONS} | Train Loss (live): {train_loss_value:.4f}")

            step += 1

    if not stop_training:
        torch.save(ema_model.state_dict(), "checkpoints/cfm_final_ema.pt")
        print("Training completed. Final model saved as 'checkpoints/cfm_final_ema.pt'.")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}. Model saved as 'checkpoints/cfm_best_ema.pt'.")

    # always plot, whether we finished the full run or stopped early
    plot_loss_history(train_hist_steps, train_hist_losses,
                      val_hist_steps, val_hist_losses, best_step)

if __name__ == "__main__":
    main()