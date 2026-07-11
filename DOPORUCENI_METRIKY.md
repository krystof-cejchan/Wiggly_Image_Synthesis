# Doporučení k evaluačním metrikám (FID / KID)

Tento dokument shrnuje, **proč FID vychází vysoké a KID lépe**, a navrhuje tři
konkrétní vylepšení (A, B, C). Součástí jsou i změny v kódu (viz sekce *Co je
implementované*).

---

## TL;DR

1. **FID je vysoké hlavně kvůli malému počtu vzorků.** Na jedno pH máme jen
   **36–136 reálných obrázků** (např. pH 8.8 = 41). FID je *vychýlený*
   estimátor — spolehlivě funguje až od řádově tisíců vzorků. **KID je
   nevychýlené**, proto při stejných datech dává rozumnější čísla. Váš rozpor
   „FID velké, KID lepší" tedy **není chyba modelu**, ale vlastnost těch dvou
   estimátorů.
2. **InceptionV3 je natrénovaná na ImageNetu** (přírodní fotky), ne na
   mikroskopii. Absolutní FID/KID proto **nejsou interpretovatelné** ani
   srovnatelné napříč pH — používejte je jen **relativně** (varianta A vs. B
   na *témž* experimentu).
3. Řešení: **(A)** odstranit bias FID, **(B)** sjednotit reálné a generované
   vzorky, **(C)** přejít na doménově vhodný backbone — **DINOv2**.

---

## Diagnóza (proč to vychází, jak vychází)

| Příčina | Vysvětlení | Dopad |
|---|---|---|
| **Málo vzorků** | FID odhaduje dvě kovariance 2048×2048. Z 41 vzorků má kovariance hodnost ≤ 40 → je z 98 % singulární. | FID je masivně **vychýlené nahoru**; absolutní číslo je nepoužitelné. |
| **KID je unbiased** | Používá MMD s polynomiálním jádrem přes podmnožiny, žádnou kovarianci neinvertuje. | Funguje i od desítek vzorků → proto vypadá lépe. |
| **Doménový posun** | ImageNet ≠ šedotónové mikrotubuly. | Features neměří to, na čem u vláken záleží; hodnoty nelze srovnat s literaturou. |
| **Mirror-padding** | Crops jsou tenké proužky (medián **292×42 px**); `val_collate_fn` je zrcadlově tapetuje na 128×128. | Reálná reference má umělé švy; u generování z šumu navíc proužek ≠ čtverec. |

---

## A. Odstranit bias FID

- **Primárně reportovat KID (± std)**, FID brát jen orientačně.
- **Snížit dimenzi features:** `feature=64` místo `2048`. Menší kovarianci
  (64×64) jde ze 41 vzorků odhadnout řádově lépe. Rychlá, velká změna.
- **FID-∞ (Chong & Forsyth, 2020):** spočítat FID pro rostoucí *n* a
  extrapolovat na *n → ∞*, čímž se bias odečte. Principiálně nejčistší řešení,
  pokud chcete reportovat i FID.
- **Získat víc vzorků**, pokud to data dovolí (augmentace jen pro referenci
  nepomůže — musí to být reálná rozmanitost).

> I s DINOv2 (dim 384) zůstává FID při *n = 41* vychýlené. Proto je **KID
> hlavní metrika**, FID doplňkový.

## B. Sjednotit reálné a generované vzorky

Data jsou **tenké proužky vlákna** (medián 292×42 px), ne čtverce. Současný
`eval_fid.py` srovnává zrcadlově tapetované proužky (reálné) proti čtvercovým
128×128 generacím — jiná globální struktura → nafouknuté FID.

- **Vyhodnocovat na proužkovém rozměru**, který odpovídá datům *i* trénovacím
  velikostem (`TRAIN_SIZES` obsahuje `(64, 256)`). Model je plně konvoluční +
  attention, takže umí generovat i nečtvercově (stačí rozměr dělitelný 16).
- **Reálné i falešné vzorky zpracovat identicky** (stejný rozměr, stejné
  paddingy) → metrika pak měří rozdíl distribucí, ne rozdíl předzpracování.

## C. Doménový feature extractor — **DINOv2** (doporučeno)

Místo ImageNet-Inceptionu použít **DINOv2** (self-supervised ViT od Meta AI):

- **Proč DINOv2:** self-supervised features se na *out-of-distribution* data
  (mikroskopie) přenášejí prokazatelně líp než ImageNet-*supervised* Inception;
  nepotřebuje labely; načte se jedním řádkem přes `torch.hub`.
- **Jak:** malý wrapper `dino_features.py` (převede šedotón → RGB, resize na
  224, ImageNet normalizace, vrátí 384-dim CLS embedding) se předá jako
  `feature=` do torchmetrics KID/FID.
- **Poznámka k závislostem:** DINOv2 se stahuje přes `torch.hub` (nutný
  internet + první stažení vah).
- **Alternativa (nejvíc „doménová"):** natrénovat malý autoencoder/klasifikátor
  přímo na mikrotubulech a použít jeho bottleneck features. Nejvěrnější doméně,
  ale víc práce.

> **Nad rámec A/B/C — nejcennější pro biologii:** doplnit *interpretovatelné*
> morfologické metriky (hustota vláken, orientační histogram, délka/zakřivení
> přes skeletonizaci) a porovnávat jejich rozdělení (KS test / Wasserstein).
> To přímo odpoví „vypadá změna s pH správně?", což FID nikdy neřekne.

---

## Opravené bugy v eval kódu

1. **Nefér kontrast** — `eval_fid_img2img.py` volal `edit_image` bez
   `contrast`, takže se na *generované* obrázky aplikovalo `torch.pow(out, 1.2)`
   (ztmavení), na *reálné* ne → systematický posun, který uměle zvyšoval FID.
   Opraveno na `contrast=1.0` (shodně s `eval_kid_img2img.py`).
2. **Pád na CPU** — `eval_fid.py` volal `torch.cuda.synchronize()` bez
   podmínky; na stroji bez CUDA (Mac) skript spadl. Obaleno `if "cuda" in DEVICE`.

---

## Co je implementované (změny v kódu)

| Soubor | Změna | Body |
|---|---|---|
| `eval_fid.py` | `feature=64`, oprava `cuda.synchronize` | A, bug |
| `eval_fid_img2img.py` | `feature=64`, `contrast=1.0` | A, bug |
| `eval_kid_img2img.py` | `feature=64` | A |
| **`dino_features.py`** *(nový)* | DINOv2 wrapper pro torchmetrics | C |
| **`eval_metrics_dino.py`** *(nový)* | referenční eval spojující A+B+C: KID (primární) + FID s DINOv2 na proužkovém rozměru 64×256 | A, B, C |

## Jak spustit

```bash
pip install -r requirements.txt          # torchmetrics[image], tqdm, ...
# 1) stáhnout checkpoint dle checkpoints/download_trained_model.txt

# 2) rychlé opravené FID/KID (Inception, feature=64) — relativní srovnání
python eval_kid_img2img.py

# 3) doporučená evaluace A+B+C (DINOv2, proužkový rozměr) — nutný internet
python eval_metrics_dino.py
```

**Interpretace:** sledujte **KID (± std)** a berte ji **relativně** — porovnávejte
varianty modelu / hyperparametry na *stejném* experimentu (stejné pH, stejná
reference). Absolutní hodnotu neinterpretujte jako „kvalitu" v absolutním smyslu.
