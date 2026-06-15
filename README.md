Wiggly Image Synthesis
======================

Overview
--------
This repository contains code for training and running a microscopy-image synthesis model (CFM-style). It includes training scripts, model definitions, dataset utilities, and example inference/image-to-image workflows.

Quick start
-----------

- Typical steps:
  - Prepare/verify your dataset under `data/` (see `dataset.py`).
  - Train with `train.py` (see script flags for hyperparameters).
  - Generate samples with `sample.py` or run image-to-image with `img2img.py` / `img2img_attention.py`.

Repository layout (important files)
---------------------------------
- `train.py` — training loop and checkpointing
- `sample.py` — generate sample images from the model
- `img2img.py`, `img2img_attention.py` — image-to-image utilities (with/without full attention)
- `model.py`, `model_attention.py` — model definitions
- `dataset.py` — dataset loader and preprocessing helpers
- `model_attention.py` — full-attention-enabled model variant
- `checkpoints/` — saved checkpoint files (e.g., `cfm_best_ema.pt`, `cfm_best_ema_attention.pt`)
- `outputs/`, `outputs_img2img/` — default output folders for results

Data layout
-----------
- Source images are organized under `data/input/` and `data/cropped/` in subfolders by condition.
- `checkpoints/` stores model weights used for inference and resuming training.

