# Conditional Flow Matching pro mikrotubuly s pH conditioning

> Reference dokument shrnující kompletní pipeline pro problém: generovat obrázky mikrotubulí podmíněné na spojitou hodnotu pH, trénink z 5 diskrétních hladin pH × 100 obrázků = 500 grayscale obrázků 256×256.

---

## Obsah

1. [Problem setup](#1-problem-setup)
2. [Flow Matching — základní teorie](#2-flow-matching--základní-teorie)
3. [Conditional Flow Matching](#3-conditional-flow-matching)
4. [Designová rozhodnutí pro náš case](#4-designová-rozhodnutí-pro-náš-case)
5. [Architektura U-Netu](#5-architektura-u-netu)
6. [Klíčové koncepty](#6-klíčové-koncepty)
7. [Kompletní kód architektury](#7-kompletní-kód-architektury)
8. [Trénink](#8-trénink)
9. [Sampling pro libovolné pH](#9-sampling-pro-libovolné-ph)
10. [Validace](#10-validace)
11. [Doporučený postup](#11-doporučený-postup)

---

## 1. Problem setup

**Vstup tréninku:**
- 5 diskrétních hladin pH (např. 5.5, 6.0, 6.5, 7.0, 7.5)
- 100 nezávislých obrázků mikrotubulí na každou hladinu
- 500 obrázků celkem, 256×256 px, grayscale fluorescence
- Žádné párování — různé mikrotubule v každé pH skupině

**Cíl:**
- Naučit se generativní model $p(x \mid \text{pH})$ takový, aby fungoval pro **libovolnou spojitou hodnotu pH**, nejen pro 5 trénovacích hladin.

**Klíčové výzvy:**
1. **Velmi malý dataset** (500 obrázků) → riziko overfittingu, potřeba silné augmentace
2. **Řídké vzorkování spojité podmínky** → model musí interpolovat mezi 5 viděnými hodnotami
3. **Extrapolace mimo trénovací rozsah** je v zásadě nespolehlivá — žádný univerzální fix
4. **Předpoklad spojitosti** $p(x \mid \text{pH})$: pokud existuje fázový přechod (např. depolymerizace při extrémním pH), interpolace přes hranici nebude fungovat

---

## 2. Flow Matching — základní teorie

### 2.1 Co Flow Matching dělá

Flow Matching (Lipman et al. 2022) trénuje **Continuous Normalizing Flow** tak, aby přenášel pravděpodobnostní hmotu z jednoduché distribuce $p_0 = \mathcal{N}(0, I)$ na cílovou datovou distribuci $p_1 = q(x)$. Místo simulace ODE během tréninku se učí přímo **vektorové pole** $v_\theta(x, t)$, které popisuje tok hmoty v čase.

### 2.2 Optimal Transport path

Pro pár $(x_0, x_1) \sim p(x_0) q(x_1)$ definujeme **lineární interpolaci** v čase $t \in [0, 1]$:

$$x_t = (1 - t) x_0 + t \, x_1$$

Cílové vektorové pole pro tento conditional path je konstantní v čase:

$$u_t(x_t \mid x_1) = x_1 - x_0$$

Tahle volba (OT path) dává **rovné trajektorie** — částice se pohybuje konstantní rychlostí mezi $x_0$ a $x_1$. To je výhodné pro sampling: při řešení ODE Eulerem stačí výrazně méně kroků než u zakřivených diffusion trajektorií.

### 2.3 Trénovací cíl

Conditional Flow Matching loss:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\, x_1 \sim q,\, x_0 \sim \mathcal{N}(0,I)} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2$$

Důležitý teorém z paperu: gradienty CFM = gradienty FM (až na konstantu). Tj. trénink per-sample dá správné marginální vektorové pole.

### 2.4 Sampling

Řešením ODE $\frac{dx}{dt} = v_\theta(x, t)$ od $t=0$ ($x_0$ z gaussovského šumu) do $t=1$. Pro OT path stačí Euler s 25–50 kroky:

```
x = randn(...)
for i in range(N):
    t = i / N
    x = x + v_theta(x, t) * (1/N)
```

---

## 3. Conditional Flow Matching

### 3.1 Rozšíření o podmínku

Pro náš případ má každý trénovací obrázek $x_1$ asociovanou podmínku $c$ (hodnota pH). Vektorové pole bere podmínku jako další vstup:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\, (x_1, c) \sim \mathcal{D},\, x_0 \sim \mathcal{N}(0,I)} \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2$$

Žádná hluboká teoretická změna — podmínka je jen extra vstup sítě.

### 3.2 Kontinuální vs diskrétní conditioning

| Typ | Embedding | Interpolace |
|---|---|---|
| **Diskrétní** (class) | Lookup table (jeden vektor per class) | ❌ Nefunguje smysluplně |
| **Kontinuální** (scalar) | Fourier features → MLP | ✅ Smooth |

Pro pH **musíme použít kontinuální embedding**, jinak by model viděl jen 5 izolovaných „klastrů" a interpolace mezi pH = 6.0 a pH = 6.5 by byla undefined.

---

## 4. Designová rozhodnutí pro náš case

### 4.1 Pixel space vs latent space

**Volba: pixel space.** Důvody:

- **SD VAE (3.x, FLUX) je trénovaná na LAION** — přirozené RGB obrazy. Na grayscale fluorescenci rozmazává jemné struktury filamentů (přesně to, co tě zajímá).
- **256×256 grayscale = 65 536 hodnot** — perfektně zvládnutelné pixel-space FM s menším U-Netem.
- **Latent space dává smysl** primárně pro vysoké rozlišení a velké datasety; ani jedno tu neplatí.

**Alternativa:** Trénovat malý vlastní VAE na tvých datech. Pro tvůj cíl ale nepřináší výhodu, jen další engineering.

### 4.2 Architekturální choice

Pro 500 obrázků a 256×256 grayscale:
- **Malý U-Net**, ~30–50M parametrů
- 4 down/up stages, 32 base channels
- Self-attention pouze v bottlenecku
- Time + pH conditioning přes **FiLM/AdaGN** v každém ResBlocku

---

## 5. Architektura U-Netu

### 5.1 High-level pohled

```
Input: (B, 1, 256, 256) grayscale image x_t
       + scalar t ∈ [0, 1]
       + scalar pH ∈ ℝ (normalized)

Output: (B, 1, 256, 256) vector field prediction
```

### 5.2 Tok dat skrz síť

```
                         ┌──────────────────────────────┐
                         │ t   →  FourierFeats → MLP    │ t_emb
                         │ pH  →  FourierFeats → MLP    │ pH_emb
                         │                              │
                         │ emb = t_emb + pH_emb         │
                         └──────────┬───────────────────┘
                                    │ (broadcast do každého ResBlocku)
                                    ▼
Input image x_t  ──► Conv 3×3 (1→32)
  256×256×1                │
                           ▼
                    ┌── ResBlock(32) ──┐    skip ──────────────────┐
   stage 1          │   ResBlock(32)   │                           │
   256×256×32       └──── DownSample ──┘ (avg pool 2×2)            │
                           │                                       │
                           ▼                                       │
                    ┌── ResBlock(64) ──┐    skip ────────────────┐ │
   stage 2          │   ResBlock(64)   │                         │ │
   128×128×64       └──── DownSample ──┘                         │ │
                           │                                     │ │
                           ▼                                     │ │
                    ┌── ResBlock(128) ─┐    skip ──────────────┐ │ │
   stage 3          │   ResBlock(128)  │                       │ │ │
   64×64×128        └──── DownSample ──┘                       │ │ │
                           │                                   │ │ │
                           ▼                                   │ │ │
                    ┌── ResBlock(256) ─┐    skip ────────────┐ │ │ │
   stage 4          │   ResBlock(256)  │                     │ │ │ │
   32×32×256        └──── DownSample ──┘                     │ │ │ │
                           │                                 │ │ │ │
                           ▼                                 │ │ │ │
        ┌──────── BOTTLENECK 16×16×256 ────────┐             │ │ │ │
        │   ResBlock(256)                      │             │ │ │ │
        │   SelfAttention2d(256, heads=4)      │             │ │ │ │
        │   ResBlock(256)                      │             │ │ │ │
        └──────────────────────────────────────┘             │ │ │ │
                           │                                 │ │ │ │
                           ▼                                 │ │ │ │
                    ┌── UpSample (nearest 2×) ─┐  ◄──────────┘ │ │ │
                    │  concat skip             │               │ │ │
                    │  ResBlock(256)           │               │ │ │
                    │  ResBlock(256)           │               │ │ │
                    └──────────────────────────┘               │ │ │
                           │                                   │ │ │
                           ▼                                   │ │ │
                    (analogické up bloky pro stage 3, 2, 1)    │ │ │
                           │                              ◄────┘ │ │
                           ▼                              ◄──────┘ │
                           │                              ◄────────┘
                    GroupNorm → SiLU → Conv 3×3 (32→1)
                           │
                           ▼
                    Output: (B, 1, 256, 256) v_θ(x_t, t, pH)
```

### 5.3 Komponenty

| Komponenta | Funkce |
|---|---|
| **FourierFeatures** | Skalár → bohatý vektor přes sin/cos různých frekvencí |
| **Embedding MLP** | Projektace Fourier features do `emb_dim` (sdíleno přes síť) |
| **FiLMResBlock** | 2× Conv + GroupNorm + AdaGN scale-shift z emb |
| **DownSample** | Average pooling 2×2 |
| **UpSample** | Nearest-neighbor upsampling 2× |
| **SelfAttention2d** | Multi-head self-attention nad flattened 2D feature mapou |
| **Skip connections** | Concatenace features ze stejné rozlišovací úrovně |

---

## 6. Klíčové koncepty

### 6.1 Fourier features (sinusoidal embedding)

**Problém:** Sítě s ReLU/SiLU se těžko učí high-frequency závislosti na skalárním vstupu. Pokud bys přímo dal `t` nebo `pH` jako jediné číslo, síť bude mít tendenci se naučit jen low-frequency („smooth") závislosti.

**Řešení:** Embedovat skalár do vektoru pomocí různě naškálovaných sinusoid:

$$\phi(x) = \big[\sin(2\pi f_1 x), \cos(2\pi f_1 x), \sin(2\pi f_2 x), \cos(2\pi f_2 x), \ldots\big]$$

Síť pak má k dispozici reprezentaci $x$ na různých časových škálách současně. Stejný princip jako positional encoding v Transformerech.

**Důležité pro naše použití:**
- pH normalizovat do rozsahu, kde má smysl vzorkovat frekvence (např. `(pH - 7) / 2`)
- `t` a `pH` mají **separátní MLP**, jinak se začnou plést
- Default: 64 frekvencí, max_freq = 10

### 6.2 FiLM (Feature-wise Linear Modulation) / AdaGN

**Co to je:** Per-channel scale a shift aplikovaný na feature mapy, generovaný z externího conditioning vektoru.

$$\text{FiLM}(h, c) = \gamma(c) \cdot h + \beta(c)$$

kde:
- $h$ je feature mapa $(B, C, H, W)$
- $\gamma(c), \beta(c) \in \mathbb{R}^C$ generované MLP z embedding $c$
- $\gamma, \beta$ jsou **stejné pro celý prostor** $(H, W)$ — modulují kanály, ne pixely

**AdaGN** = AdaGroupNorm = aplikace FiLM po GroupNorm:

```
h_normed = GroupNorm(h)
h_modulated = h_normed * (1 + scale) + shift
```

Zápis `(1 + scale)` místo `scale` znamená, že při scale=0 (init) AdaGN = obyčejná GroupNorm. Spolu se zero-init MLP to dává **identity initialization** — síť startuje jako kdyby conditioning nebyl, a postupně se učí ho používat.

**Proč scale-shift, ne jen shift:**
- *Pouze shift* (originální DDPM): conditioning přidává konstantní offset → jen zvýrazňuje/potlačuje kanály
- *Scale-shift* (ADM, current standard): conditioning může i přeškálovat dynamiku kanálu → silnější effect

Pro pH je scale-shift relevantní: různá pH nejen mění *přítomnost* určitých struktur, ale i jejich *intenzitu/výraznost*.

### 6.3 Self-attention v 2D

V bottlenecku je rozlišení 16×16 = 256 tokenů. Full self-attention je perfektně zvládnutelná (quadratic v počtu tokenů, ale 256² = 65k operací = nic).

**Mechanismus:**
1. Feature mapa $(B, C, H, W)$ se proloží jako $(B, HW, C)$ — každý prostorový bod = 1 token
2. Z každého tokenu se spočítá Query, Key, Value pomocí 1×1 conv
3. Attention: $\text{softmax}(QK^T / \sqrt{d_{\text{head}}}) \cdot V$
4. Output se proloží zpět na $(B, C, H, W)$ a přičte k vstupu (residual)

**Proč:**
Konvoluce má omezené receptive field. Filamenty mikrotubulů jsou prostorově rozsáhlé struktury, které pokračují přes celý obraz. Attention umožňuje, aby každý bod „viděl" každý jiný bod a model se naučil long-range dependencies (např. „filamenty mají tendenci pokračovat v určitém úhlu").

**Kam dát attention:**
- Bottleneck (16×16): **ano** — tady je to skoro zadarmo a má velký benefit
- Středně nízká rozlišení (32×32, 64×64): možná, pokud máš compute
- Vysoká rozlišení (128×128+): příliš drahé pro full attention, potřeba window/linear variants

Pro náš case: **jen bottleneck**.

### 6.4 Classifier-free guidance (CFG)

**Trénink:** S pravděpodobností ~10 % nahraď podmínku za speciální „null" hodnotu. Síť se učí oba režimy (conditional i unconditional) v jednom modelu.

**Sampling:** Kombinuj predikce s a bez podmínky pro „zesílení" conditioning effectu:

$$v_{\text{cfg}}(x, t, c) = v_\theta(x, t, \emptyset) + w \cdot \big[v_\theta(x, t, c) - v_\theta(x, t, \emptyset)\big]$$

- `w = 0`: ignoruje podmínku (unconditional)
- `w = 1`: standardní conditional
- `w > 1`: zesílená podmínka — model „víc poslouchá" pH

Pro silnější pH dependenci zkus `w` mezi 2 a 5. Pozor: vysoké `w` může zhoršit obrazovou kvalitu (saturace, artefakty).

### 6.5 Zero-initialization triky

Tři místa v síti, která záměrně inicializujeme na nuly:

1. **Poslední lineární vrstva FiLM MLP** → AdaGN na startu = obyčejná GroupNorm
2. **Poslední konvoluce v ResBlocku** → ResBlock na startu = identity skrz skip
3. **Poslední projekce v self-attention** → attention na startu = identity skrz residual

Důsledek: na začátku tréninku se chová celá síť jako lineární funkce nad input, postupně se učí přidávat nelineární transformace. To výrazně stabilizuje trénink, zejména u hlubších sítí.

---

## 7. Kompletní kód architektury

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Embedding moduly
# -----------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Skalár → (2 * num_freqs)-rozměrný vektor přes sin/cos."""
    
    def __init__(self, num_freqs=64, max_freq=10.0):
        super().__init__()
        # Logaritmické rozdělení frekvencí dává dobré pokrytí škál
        freqs = torch.logspace(0.0, math.log10(max_freq), num_freqs)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x):
        # x: (B,) skalár
        angles = x.unsqueeze(-1) * self.freqs * math.pi   # (B, num_freqs)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ScalarEmbedding(nn.Module):
    """Fourier + 2-vrstvá MLP → emb_dim."""
    
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


# -----------------------------------------------------------------------------
# Core building blocks
# -----------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    """ResBlock se scale-shift conditioningem (AdaGN)."""
    
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=32, dropout=0.1):
        super().__init__()
        # První conv blok (bez conditioningu)
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # FiLM projekce: emb_dim → 2 * out_ch (scale a shift)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        nn.init.zeros_(self.emb_proj[-1].weight)
        nn.init.zeros_(self.emb_proj[-1].bias)
        
        # Druhý conv blok (s FiLM modulací)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Zero-init poslední conv: ResBlock startuje jako identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Skip projection pokud se mění počet kanálů
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # FiLM modulace
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
    """Multi-head self-attention nad (B, C, H, W) feature mapou."""
    
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Zero-init: attention startuje jako identity skrz residual
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)             # (B, heads, head_dim, HW)
        
        # Transpose pro scaled_dot_product_attention: (B, heads, HW, head_dim)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


# -----------------------------------------------------------------------------
# Down / Up stages
# -----------------------------------------------------------------------------

class DownStage(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_blocks=2, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(num_blocks):
            self.blocks.append(FiLMResBlock(ch, out_ch, emb_dim))
            ch = out_ch
        self.downsample = (
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            if downsample else nn.Identity()
        )
    
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
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity()
        )
        # Po concat má vstup (in_ch + skip_ch) kanálů
        self.blocks = nn.ModuleList()
        ch = in_ch + skip_ch
        for i in range(num_blocks):
            self.blocks.append(FiLMResBlock(ch, out_ch, emb_dim))
            ch = out_ch + skip_ch   # další blok bude opět concat se skipem
        # Poslední blok bez dalšího skipu
        self.blocks_skip_concat = num_blocks
    
    def forward(self, x, skips, emb):
        x = self.upsample(x)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, emb)
        return x


# -----------------------------------------------------------------------------
# Bottleneck
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Hlavní U-Net
# -----------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    U-Net pro Conditional Flow Matching.
    Vstup: (B, 1, 256, 256), t ∈ [0,1], pH (normalized scalar nebo NaN pro null)
    Výstup: (B, 1, 256, 256) — predikované vektorové pole
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),       # → 32, 64, 128, 256
        num_res_blocks=2,
        emb_dim=256,
        num_heads=4,
    ):
        super().__init__()
        
        # Embedding moduly
        self.t_embed = ScalarEmbedding(emb_dim)
        self.pH_embed = ScalarEmbedding(emb_dim)
        self.null_pH_emb = nn.Parameter(torch.zeros(emb_dim))
        
        # Input projekce
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Down stages
        self.down_stages = nn.ModuleList()
        ch_list = [base_channels]
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = (i == len(channel_mults) - 1)
            self.down_stages.append(
                DownStage(ch, out_ch, emb_dim, num_res_blocks, downsample=not is_last)
            )
            ch = out_ch
            ch_list.append(ch)
        
        # Bottleneck
        self.bottleneck = Bottleneck(ch, emb_dim, num_heads=num_heads)
        
        # Up stages (reverse order)
        self.up_stages = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            is_first = (i == 0)
            is_last = (i == len(channel_mults) - 1)
            in_ch = ch
            skip_ch = ch_list[-(i + 1)]
            out_ch = base_channels * (channel_mults[-(i + 2)] if not is_last else 1)
            self.up_stages.append(
                UpStage(in_ch, skip_ch, out_ch, emb_dim,
                        num_res_blocks, upsample=not is_first)
            )
            ch = out_ch
        
        # Output head
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t, pH):
        # Embeddings
        t_emb = self.t_embed(t)
        
        # pH embedding s null-condition handling pro CFG
        is_null = torch.isnan(pH)
        pH_safe = torch.where(is_null, torch.zeros_like(pH), pH)
        pH_emb_real = self.pH_embed(pH_safe)
        pH_emb = torch.where(
            is_null.unsqueeze(-1),
            self.null_pH_emb.expand_as(pH_emb_real),
            pH_emb_real,
        )
        
        emb = t_emb + pH_emb
        
        # Forward přes U-Net
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
```

**Poznámka:** Tento kód je kostra pro pochopení architektury. Pro production použití (např. tvůj wound healing scratch assay paper) doporučuji opřít se o `diffusers.UNet2DModel` nebo o čistou EDM referenci (NVlabs/edm) a jen upravit conditioning a Flow Matching loss. Vyhneš se tím subtle bugům.

---

## 8. Trénink

### 8.1 Augmentace (kritické pro 500 obrázků)

```python
import torchvision.transforms.v2 as T

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomChoice([T.RandomRotation(d) for d in [0, 90, 180, 270]]),
    T.RandomCrop(224),                                      # z 256 cropuj 224
    T.Resize(256, antialias=True),                          # zpátky na 256
    T.ColorJitter(brightness=0.1, contrast=0.1),            # mírný intensity jitter
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.5], std=[0.5]),                     # do [-1, 1]
])
```

**Co NEdělat:**
- Žádné affine deformace (shear, scale != 1)
- Žádné aspect ratio změny — měnily by fyzikální vlastnosti filamentů
- Žádné rozdílné augmentace per-pH (musí být stejné distribučně, jinak shortcut)

### 8.2 Trénovací loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

device = "cuda"
model = ConditionalUNet().to(device)
ema_model = deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad = False

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100_000)

# pH normalizace (přizpůs si rozsahu)
pH_min, pH_max = 5.5, 7.5
def normalize_pH(pH):
    return 2 * (pH - pH_min) / (pH_max - pH_min) - 1   # → [-1, 1]

def train_step(x_batch, pH_batch, cfg_dropout=0.1):
    x1 = x_batch.to(device)
    pH = normalize_pH(pH_batch.to(device).float())
    
    # Sample noise & time
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    
    # OT path interpolation
    t_expand = t.view(-1, 1, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0
    
    # Classifier-free guidance dropout
    drop_mask = torch.rand(x1.shape[0], device=device) < cfg_dropout
    pH_input = torch.where(drop_mask, torch.full_like(pH, float("nan")), pH)
    
    # Forward & loss
    pred = model(xt, t, pH_input)
    loss = F.mse_loss(pred, target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    # EMA update
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.mul_(0.9999).add_(p, alpha=0.0001)
    
    return loss.item()
```

### 8.3 Doporučené hyperparametry

| Parametr | Hodnota | Poznámka |
|---|---|---|
| Batch size | 32–64 | Limit GPU paměti |
| Learning rate | 1e-4 | AdamW |
| Weight decay | 1e-4 | Důležité pro malý dataset |
| Gradient clipping | 1.0 | Stabilizuje trénink |
| EMA decay | 0.9999 | Sampluj z EMA, ne raw modelu |
| CFG dropout | 0.1 | 10 % batch bez podmínky |
| Iterace | ~50k–200k | Dle konvergence |
| Warmup | 1000 kroků | Lineární od 0 na peak |

### 8.4 Co sledovat během tréninku

- **Loss** by měl klesat plynule, žádné spikes
- **Sample quality** každých 5k kroků (sampluj 16 obrázků pro každé z 5 trénovacích pH)
- **Interpolace** každých 10k kroků (sampluj sérii pro pH od 5.5 do 7.5 s krokem 0.1, stejný initial noise)
- **EMA weights** dávají vždy lepší samples než raw

---

## 9. Sampling pro libovolné pH

```python
@torch.no_grad()
def sample(pH_query, num_samples=16, num_steps=50, cfg_scale=3.0, seed=None):
    """Sampluj obrázky pro dané pH."""
    if seed is not None:
        torch.manual_seed(seed)
    
    pH_norm = normalize_pH(torch.tensor([pH_query] * num_samples)).to(device)
    pH_null = torch.full((num_samples,), float("nan"), device=device)
    
    x = torch.randn(num_samples, 1, 256, 256, device=device)
    
    for i in range(num_steps):
        t = torch.full((num_samples,), i / num_steps, device=device)
        
        v_cond   = ema_model(x, t, pH_norm)
        v_uncond = ema_model(x, t, pH_null)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        x = x + v_cfg * (1.0 / num_steps)
    
    # Denormalize [-1, 1] → [0, 1]
    return (x.clamp(-1, 1) + 1) / 2
```

**Volba `cfg_scale`:** Zkus 1.0, 2.0, 3.0, 5.0 a vizuálně porovnej. Vyšší = silnější pH dependence, ale i víc artefaktů.

**Volba `num_steps`:** S OT path stačí 25–50. Pro lepší kvalitu midpoint solver:

```python
# Midpoint integration step
def midpoint_step(x, t, dt, pH_norm, pH_null, w):
    def v(x_in, t_in):
        v_c = ema_model(x_in, t_in, pH_norm)
        v_u = ema_model(x_in, t_in, pH_null)
        return v_u + w * (v_c - v_u)
    
    k1 = v(x, t)
    k2 = v(x + 0.5 * dt * k1, t + 0.5 * dt)
    return x + dt * k2
```

---

## 10. Validace

### 10.1 Conditional FID per pH

1. Z každé trénovací pH skupiny odlož 10–20 obrázků jako validační set
2. Sampluj 1000 obrázků pro každé trénovací pH
3. Spočítej FID(generated_pH=k, val_pH=k) — měl by být nízký
4. Spočítej cross-FID FID(generated_pH=k, val_pH=j) pro k ≠ j — měl by být vyšší

Pokud `cross-FID < self-FID`, model nerozlišuje pH a generuje stejné distribuce nezávisle na podmínce.

### 10.2 Interpolation smoothness

```python
# Stejný noise, různé pH
torch.manual_seed(42)
x0_fixed = torch.randn(1, 1, 256, 256, device=device)

frames = []
for pH_val in np.arange(5.5, 7.51, 0.05):
    x = x0_fixed.clone()
    # ... sample loop s tímhle fixním x0 ...
    frames.append(x)

# Uložit jako gif/video — měla by být plynulá morfovací sekvence
```

Pokud vidíš diskrétní skoky v trénovacích pH hodnotách, model overfittl na 5 klastrů.

### 10.3 Klasifikátor sanity check

1. Natrénuj small CNN klasifikátor pH (5-class) na **reálných** datech, val accuracy cross-validation
2. Aplikuj na **vygenerované** obrázky pro každé trénovací pH
3. Confusion matrix — diagonála by měla být dominantní

Pokud klasifikátor dává 50–80 % na reálných, ale 25 % na generovaných, model nedostatečně podmiňuje.

### 10.4 Fyzikální deskriptory (nejsilnější validace)

Pro tvou doménu — spočítej z reálných i generovaných obrázků:
- Průměrná délka filamentu (skeletonization + measurement)
- Total filament density (binary mask area / total area)
- Branching ratio
- Mean angle distribution
- Persistence length (pokud relevantní)

Porovnej **trendy přes pH** mezi reálnými a generovanými. To je tvůj golden standard — pokud sedí, model se naučil biologicky smysluplnou závislost.

---

## 11. Doporučený postup

V tomto pořadí:

### Krok 1: Sanity check dat (před tréninkem)
- Vizualizuj 10 obrázků z každé pH skupiny vedle sebe
- Spočítej základní deskriptory per pH a podívej se, jestli existuje viditelný trend
- Pokud reálná data nejsou viditelně podmíněna na pH, model to taky nezachytí

### Krok 2: Unconditional baseline
- Trénuj FM na všech 500 obrázcích **bez pH podmínky**
- Zkontroluj, že model generuje věrohodné mikrotubuly
- Pokud tohle nefunguje, conditional verze nemá smysl ladit
- Diagnostika: malý model? málo iterací? augmentace OK?

### Krok 3: Conditional FM bez CFG
- Přidej pH embedding přes Fourier features + FiLM
- Bez classifier-free guidance dropout
- Validuj: FID per pH, interpolace, klasifikátor

### Krok 4: Přidej CFG
- Dropout 10 %, null embedding
- Sampluj s `w ∈ {1, 2, 3, 5}` a porovnej
- Měla by se zlepšit pH dependence

### Krok 5: Fine-tune detaily
- EMA decay
- LR schedule
- Augmentace intenzita
- Možná víc/méně self-attention

### Krok 6: Pokud potřebuješ víc kvality
- Pretraining na příbuzných datech (CytoImageNet, BBBC fluorescenční datasety) → fine-tune
- Větší model + víc augmentace
- EDM-style preconditioning (Karras et al. 2022) — pro malé datasety často lepší než vanilla FM

---

## Reference

- **Lipman et al. 2022.** Flow Matching for Generative Modeling. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Liu et al. 2022.** Flow Straight and Fast (Rectified Flow). [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- **Esser et al. 2024.** Scaling Rectified Flow Transformers (SD3). [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
- **Dhariwal & Nichol 2021.** Diffusion Models Beat GANs (ADM, AdaGN). [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
- **Karras et al. 2022.** EDM: Elucidating the Design Space of Diffusion Models. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- **Ho et al. 2022.** Classifier-Free Diffusion Guidance. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- **Perez et al. 2017.** FiLM: Visual Reasoning with a General Conditioning Layer. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
