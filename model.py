import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    def __init__(self, num_freqs=64, max_freq=10.0):
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_freq), num_freqs)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x):
        angles = x.unsqueeze(-1) * self.freqs * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

class ScalarEmbedding(nn.Module):
    def __init__(self, emb_dim=256, num_freqs=64, max_freq=10.0):
        super().__init__()
        self.fourier = FourierFeatures(num_freqs, max_freq)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, x):
        return self.mlp(self.fourier(x))

class FiLMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=32, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        nn.init.zeros_(self.emb_proj[-1].weight)
        nn.init.zeros_(self.emb_proj[-1].bias)
        
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.skip(x) + h

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)
        
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
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
        skips = []
        for block in self.blocks:
            x = block(x, emb)
            skips.append(x)
        x = self.downsample(x)
        return x, skips

class UpStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, num_blocks=2, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity()
        self.blocks = nn.ModuleList()
        ch = in_ch + skip_ch
        for _ in range(num_blocks):
            self.blocks.append(FiLMResBlock(ch, out_ch, emb_dim))
            ch = out_ch + skip_ch
        self.blocks_skip_concat = num_blocks
    
    def forward(self, x, skips, emb):
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
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x

class ConditionalUNet(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        base_channels=32, 
        # ZMĚNA: Ubrali jsme jeden krok (8x násobitel). Nyní: 32 -> 64 -> 128 -> bottleneck
        channel_mults=(1, 2, 4), 
        num_res_blocks=2, 
        emb_dim=256, 
        num_heads=4
    ):
        super().__init__()
        
        self.t_embed = ScalarEmbedding(emb_dim)
        self.pH_embed = ScalarEmbedding(emb_dim)
        self.null_pH_emb = nn.Parameter(torch.zeros(emb_dim))
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
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
        
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t, pH):
        # ... forward pass zůstává úplně STEJNÝ ...
        t_emb = self.t_embed(t)
        
        is_null = torch.isnan(pH)
        pH_safe = torch.where(is_null, torch.zeros_like(pH), pH)
        pH_emb_real = self.pH_embed(pH_safe)
        pH_emb = torch.where(
            is_null.unsqueeze(-1),
            self.null_pH_emb.expand_as(pH_emb_real),
            pH_emb_real,
        )
        
        emb = t_emb + pH_emb
        
        x = self.conv_in(x)
        all_skips = []
        for stage in self.down_stages:
            x, skips = stage(x, emb)
            all_skips.extend(skips)
        
        x = self.bottleneck(x, emb)
        
        for stage in self.up_stages:
            x = stage(x, all_skips, emb)
        
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x