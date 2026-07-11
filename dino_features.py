"""
DINOv2 feature extractor (bod C z DOPORUCENI_METRIKY.md).

Wrapper, který lze předat jako `feature=` do torchmetrics
FrechetInceptionDistance / KernelInceptionDistance místo ImageNet-Inceptionu.
DINOv2 je self-supervised ViT (Meta AI) a jeho features se na out-of-distribution
data (mikroskopie) přenášejí líp než ImageNet-supervised Inception.

Očekává float obrázky v [0, 1], tvar (N, 1 nebo 3, H, W). Vrací (N, 384).

Pozn.: model se stahuje přes torch.hub (nutný internet + první stažení vah).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Features(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", img_size: int = 224):
        super().__init__()
        # ViT-S/14: rychlý, 384-dim embedding. Pro víc kapacity lze dinov2_vitb14 (768-dim).
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # img_size musí být dělitelné patch_size (14). 224 / 14 = 16.
        self.img_size = img_size
        # torchmetrics si dimenzi zjistí z výstupu; num_features držíme pro jistotu.
        self.num_features = getattr(self.backbone, "embed_dim", 384)

        # DINOv2 byl trénován s ImageNet normalizací.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defenzivně: přijmi i uint8 [0,255].
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.float()

        # šedotón -> RGB (DINOv2 čeká 3 kanály)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # resize na čtverec dělitelný patch_size + ImageNet normalizace
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        # CLS embedding (N, embed_dim)
        return self.backbone(x)
