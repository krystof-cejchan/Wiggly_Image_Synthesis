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
# stale in some of them.
#
# The suffix marks the ARCHITECTURE GENERATION, and the generations deliberately do not
# share a name because they do not load into each other - model.from_state_dict infers which
# one a file is from its tensors:
#
#   (no suffix) pre-waviness            1-channel, pH only
#   _ex         waviness-conditioned    adds WavinessEmbedding (waviness_* tensors)
#   _pair       source-conditioned      conv_in takes 2 channels; adds the undulation-PERIOD
#                                       channel (log_period_* tensors)
#   _ripple     current                 replaces period with the fine-undulation RIPPLE rms
#                                       (ripple_* tensors). Period was measured dead on the
#                                       _pair checkpoint - sweeping it over its whole range
#                                       moved the velocity field 0.05-0.15% - and the model
#                                       put 3x too much centreline energy into one long wave
#                                       and 4x too little into fine ripple. See train.py's
#                                       WARP_AUG_*_RIPPLE_FRACTION and model.py's
#                                       ripple_conditioned.
#
# CHECKPOINT_PATH is what train.py WRITES and what every script's --checkpoint defaults to.
# Those two used to disagree - train.py wrote the _pair names while every reader defaulted to
# a _ex path that did not exist on disk at all, so the CLI could not run on its own defaults.
CHECKPOINT_PATH = "checkpoints/cfm_best_ema_ripple.pt"
# Selected on how much the model USES its conditioning rather than on flow-matching MSE
# - see train.py's evaluate(). The two criteria disagree, and this is the one that
# tracks the actual deliverable (editing to a requested geometry).
COND_CHECKPOINT_PATH = "checkpoints/cfm_best_cond_ripple.pt"
FINAL_CHECKPOINT_PATH = "checkpoints/cfm_final_ema_ripple.pt"

# Previous generations. Nothing defaults to these any more; they exist so an older
# checkpoint can still be named explicitly (--checkpoint) and so the names are not
# accidentally reused by a future run.
EX_CHECKPOINT_PATH = "checkpoints/cfm_best_ema_ex.pt"
PAIR_CHECKPOINT_PATH = "checkpoints/cfm_best_ema_pair.pt"
PAIR_COND_CHECKPOINT_PATH = "checkpoints/cfm_best_cond_pair.pt"
PAIR_FINAL_CHECKPOINT_PATH = "checkpoints/cfm_final_ema_pair.pt"
