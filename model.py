import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    """
    Fourier feature mapping is applied to the input coordinates before they are fed into the MLP.
    This mapping involves transforming each input coordinate pair (x, y)
    using a series of sinusoidal functions (sine and cosine) with varying frequencies.
    used to better capture high-frequency signals in data (https://medium.com/@aadityaza/understanding-fourier-feature-mapping-through-simple-image-regression-examples-570731f95e4a)
    """
    def __init__(self, num_freqs=64, max_freq=10.0):
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_freq), num_freqs)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x):
        # x: (...,) scalar inputs (e.g. time or pH)
        # produces concatenated sin/cos features of shape (..., 2 * num_freqs)
        angles = x.unsqueeze(-1) * self.freqs * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

class ScalarEmbedding(nn.Module):
    """
    takes the output from FourierFeatures and passes it through a small MLP to produce a dense embedding vector.
    """
    def __init__(self, emb_dim=256, num_freqs=64, max_freq=10.0):
        super().__init__()
        self.fourier = FourierFeatures(num_freqs, max_freq)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, x):
        # Map scalar -> high-dimensional embedding used for FiLM
        return self.mlp(self.fourier(x))

class FiLMResBlock(nn.Module):        
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=32, dropout=0.1):
        super().__init__()
        # first normalization operates over `in_ch` channels
        # Using math.gcd to prevent crashes if in_ch is not divisible by num_groups
        self.norm1 = nn.GroupNorm(math.gcd(num_groups, in_ch), in_ch)
        # first convolution changes channel dimension in_ch -> out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        # initialize projection to produce near-zero scale/shift at start
        nn.init.zeros_(self.emb_proj[-1].weight)
        nn.init.zeros_(self.emb_proj[-1].bias)
        
        # second normalization is applied before FiLM modulation
        self.norm2 = nn.GroupNorm(math.gcd(num_groups, out_ch), out_ch)
        # small dropout between activations and final conv
        self.dropout = nn.Dropout(dropout)
        # final conv preserves channel count (out_ch -> out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # zero-init final conv so block initially behaves like identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, emb):
        # Pre-activation: normalize then SiLU non-linearity
        # x shape: (B, in_ch, H, W); emb shape: (B, emb_dim)
        h = F.silu(self.norm1(x))
        # first conv produces (B, out_ch, H, W)
        h = self.conv1(h)

        # Project conditioning embedding to FiLM params.
        # emb_proj -> (B, 2 * out_ch). We split into `scale` and `shift`.
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        # reshape to (B, out_ch, 1, 1) so they broadcast across spatial dims
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Normalize features before modulation
        h = self.norm2(h)
        # Apply FiLM: elementwise scaling and shifting per-channel.
        # Using (1 + scale) keeps modulation identity-centered (scale~0 -> factor~1).
        h = h * (1 + scale) + shift

        # Activation -> dropout -> final conv
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Add residual (skip path may be a 1x1 conv when channels differ)
        return self.skip(x) + h

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        # dimension per attention head
        self.head_dim = channels // num_heads

        # GroupNorm over channels to stabilize activations
        self.norm = nn.GroupNorm(math.gcd(num_groups, channels), channels)
        # compute q, k, v in a single 1x1 conv; outputs 3 * channels
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        # final linear projection after attention (per-channel)
        self.proj = nn.Conv2d(channels, channels, 1)

        # initialize proj to near-zero so block starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        # Multi-head self-attention over spatial positions.
        B, C, H, W = x.shape
        h = self.norm(x)
        # produce q, k, v and reshape for multi-head attention.
        # after qkv conv: (B, 3*C, H, W) -> reshape to (B, 3, num_heads, head_dim, H*W)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)  # each is (B, num_heads, head_dim, H*W)

        # scaled_dot_product_attention expects (..., seq_len, head_dim).
        # transpose to (B, num_heads, H*W, head_dim)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))

        # compute attention over spatial positions: output shape
        # (B, num_heads, H*W, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)

        # transpose back and reshape to (B, C, H, W)
        out = out.transpose(-1, -2).reshape(B, C, H, W)

        # final linear projection and residual connection
        return x + self.proj(out)

class DownStage(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_blocks=2, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(num_blocks):
            self.blocks.append(FiLMResBlock(ch, out_ch, emb_dim))
            ch = out_ch
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else nn.Identity()
    
    def forward(self, x, emb):
        # Apply sequence of FiLM residual blocks, collect skip connections
        skips = []
        for block in self.blocks:
            x = block(x, emb)
            skips.append(x)

        # optionally downsample spatial resolution by 2
        x = self.downsample(x)
        return x, skips

class UpStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, num_blocks=2, upsample=True):
        super().__init__()
        # Změna z nearest na bilinear pro zamezení checkerboard efektu
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else nn.Identity()
        self.blocks = nn.ModuleList()
        ch = in_ch + skip_ch
        for _ in range(num_blocks):
            self.blocks.append(FiLMResBlock(ch, out_ch, emb_dim))
            ch = out_ch + skip_ch
        self.blocks_skip_concat = num_blocks
    
    def forward(self, x, skips, emb):
        # Upsample and sequentially concatenate corresponding skip connections
        x = self.upsample(x)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, emb)
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels, emb_dim, num_heads=4):
        super().__init__()
        self.res1 = FiLMResBlock(channels, channels, emb_dim)
        self.attn = SelfAttention2d(channels, num_heads=num_heads)
        self.res2 = FiLMResBlock(channels, channels, emb_dim)
    
    def forward(self, x, emb):
        # Bottleneck: res -> attention -> res
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x

class ConditionalUNet(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        base_channels=64,              # Safer default
        channel_mults=(1, 2, 4, 8, 8), # Extended to 5 stages -> 128 to 64, 32, 16, 8, 8 (Bottleneck at 8x8)
        num_res_blocks=2, 
        emb_dim=256, 
        num_heads=4
    ):
        super().__init__()
        
        self.t_embed = ScalarEmbedding(emb_dim)
        self.pH_embed = ScalarEmbedding(emb_dim)
        self.null_pH_emb = nn.Parameter(torch.zeros(emb_dim))
        
        # Přidáno MLP pro modelování komplexnější interakce mezi časem a pH
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.pH_proj = nn.Linear(emb_dim, emb_dim)
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.down_stages = nn.ModuleList()
        ch_list = [base_channels]
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = (i == len(channel_mults) - 1)
            self.down_stages.append(DownStage(ch, out_ch, emb_dim, num_res_blocks, downsample=not is_last))
            ch = out_ch
            ch_list.append(ch)
        
        self.bottleneck = Bottleneck(ch, emb_dim, num_heads=num_heads)
        
        self.up_stages = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            is_first = (i == 0)
            is_last = (i == len(channel_mults) - 1)
            in_ch = ch
            skip_ch = ch_list[-(i + 1)]
            out_ch = base_channels * (channel_mults[-(i + 2)] if not is_last else 1)
            self.up_stages.append(UpStage(in_ch, skip_ch, out_ch, emb_dim, num_res_blocks, upsample=not is_first))
            ch = out_ch
        
        # Zajištěno, aby norm_out nepadala
        self.norm_out = nn.GroupNorm(math.gcd(32, base_channels), base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t, pH):
        """
        Forward pass for conditional U-Net.

        Args:
            x: image tensor (B, C_in, H, W)
            t: scalar tensor for time/step (B,)
            pH: scalar tensor for conditional pH (B,) or NaN for null condition

        Returns:
            output tensor (B, C_out, H, W)
        """
        # embed time and pH scalars
        t_emb = self.t_embed(t)

        # replace NaN with a learned null embedding
        is_null = torch.isnan(pH)
        pH_safe = torch.where(is_null, torch.zeros_like(pH), pH)
        pH_emb_real = self.pH_embed(pH_safe)
        pH_emb = torch.where(
            is_null.unsqueeze(-1),
            self.null_pH_emb.expand_as(pH_emb_real),
            pH_emb_real,
        )

        # Sloučení přes MLP pro modelování vzájemné závislosti podmínek
        emb = F.silu(self.t_proj(t_emb) + self.pH_proj(pH_emb))
        # input conv
        x = self.conv_in(x)
        all_skips = []

        # collect skips from each downstage
        for stage in self.down_stages:
            x, skips = stage(x, emb)
            all_skips.extend(skips)

        # bottleneck
        x = self.bottleneck(x, emb)

        # consume skip connections in reverse order
        for stage in self.up_stages:
            x = stage(x, all_skips, emb)

        # final normalization and conv
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x