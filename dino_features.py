"""
DINOv2 feature extractor.

A wrapper that can be passed as `feature=` to torchmetrics'
FrechetInceptionDistance / KernelInceptionDistance in place of ImageNet-Inception.
DINOv2 is a self-supervised ViT (Meta AI), and its features transfer to
out-of-distribution data (microscopy) better than ImageNet-supervised Inception does.

Expects float images in [0, 1], shape (N, 1 or 3, H, W). Returns (N, 384).

Note: the model is downloaded via torch.hub (needs internet on first run to fetch weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Features(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", img_size: int = 224):
        super().__init__()
        # Pinned to a commit predating facebookresearch/dinov2#528, which added
        # `float | None` union-type syntax to attention.py - that's Python 3.10+
        # only and breaks `torch.hub.load` on 3.9 (TypeError: unsupported operand
        # type(s) for |). `main` floats and could pick up other breaking changes
        # over time regardless, so pin to a known-good ref rather than chase this.
        # skip_validation is required because torch.hub's ref-validation only
        # recognizes branches/tags, not arbitrary commit SHAs.
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2:81b2b64", model_name, trust_repo=True, skip_validation=True
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # img_size must be divisible by patch_size (14). 224 / 14 = 16.
        self.img_size = img_size
        # torchmetrics infers the dimension from the output; num_features is kept just in case.
        self.num_features = getattr(self.backbone, "embed_dim", 384)

        # DINOv2 was trained with ImageNet normalization.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defensively: accept uint8 [0,255] too.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.float()

        # greyscale -> RGB (DINOv2 expects 3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # resize to a square divisible by patch_size + ImageNet normalization
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        # CLS embedding (N, embed_dim)
        return self.backbone(x)
