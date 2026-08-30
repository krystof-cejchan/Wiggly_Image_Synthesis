import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PH_MIN, PH_MAX = 5.8, 8.8

# Frame geometry the model is trained on, matched to the real data: source heights are
# 10-121px (median 43, p90 66) and widths run to ~950. All dims stay divisible by 16 for the
# U-Net's 4 downsampling stages. See framing.py for why short crops are grown with
# synthesised background rather than mirrored, and train.py's WARP_AUG_MAX_WAVINESS for why
# the taller frames have to exist at all.
#
# This lives in config.py, next to PH_MIN/PH_MAX, because it is a property of the CHECKPOINT
# that inference has to respect - not a training-only hyperparameter. img2img.py's editing
# window used to be set independently (128x128, from when training used square crops) and
# silently drifted out of this band when TRAIN_SIZES moved to thin strips. Measured on the
# waviness-conditioned checkpoint, that cost the waviness conditioning ALL of its effect:
# free generation at 128px wide produced no traceable fibre at all and conditioning
# 3/11/20px of waviness gave back 1.2/1.0/1.4px - no response - where the same request on a
# 384px-wide frame gives 3.1/9.0/13.0px. img2img.py now derives its window from this same
# list, so the two cannot drift apart again.
TRAIN_SIZES = [(48, 384), (64, 256), (64, 384), (80, 256), (96, 192)]
TRAIN_MIN_H = min(h for h, _ in TRAIN_SIZES)
TRAIN_MAX_H = max(h for h, _ in TRAIN_SIZES)
TRAIN_MIN_W = min(w for _, w in TRAIN_SIZES)
TRAIN_MAX_W = max(w for _, w in TRAIN_SIZES)

# The checkpoint every script reads and train.py writes, in one place for the same reason
# TRAIN_SIZES is: a filename that each script spells out for itself is a filename that goes
# stale in some of them. The "_ex" suffix marks the waviness-conditioned architecture (see
# model.py's WavinessEmbedding) - a checkpoint saved before that embedding existed has no
# waviness_* tensors and will NOT load into today's ConditionalUNet, so the two generations
# deliberately do not share a name.
CHECKPOINT_PATH = "checkpoints/cfm_best_ema_ex.pt"
# Selected on how much the model USES its conditioning rather than on flow-matching MSE
# - see train.py's evaluate(). The two criteria disagree, and this is the one that
# tracks the actual deliverable (editing to a requested pH).
COND_CHECKPOINT_PATH = "checkpoints/cfm_best_cond_ex.pt"
FINAL_CHECKPOINT_PATH = "checkpoints/cfm_final_ema_ex.pt"

# The source-conditioned (paired-edit) generation of the architecture: conv_in takes two
# channels, the second being the image being edited. Trained by train.py on before/after
# pairs, and rendered by img2img without any strength/anchor at all. Kept under its own name
# because it will not load into the 1-channel model and vice versa - model.from_state_dict
# infers which one a file is.
PAIR_CHECKPOINT_PATH = "checkpoints/cfm_best_ema_pair.pt"
PAIR_COND_CHECKPOINT_PATH = "checkpoints/cfm_best_cond_pair.pt"
PAIR_FINAL_CHECKPOINT_PATH = "checkpoints/cfm_final_ema_pair.pt"
