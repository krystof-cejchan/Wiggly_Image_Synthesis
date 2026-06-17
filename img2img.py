import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.v2.functional as TF
from model import ConditionalUNet

#matplotlib.use('Qt5Agg')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PH_MIN/PH_MAX a normalize_pH jsou zkopírované i v train.py a sample.py.
# asi bych imprtoval z jednoho config.py. - ale to je jen detail
PH_MIN, PH_MAX = 5.8, 8.8

def normalize_pH(pH):
    """Normalizuje pH na interval [-1, 1]."""
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

def load_and_preprocess_image(image_path):
    """Načte referenční obrázek, vycpe ho na násobek 16 a vrátí původní velikost."""
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)
    
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    _, _, h, w = img_tensor.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=-1.0)
    
    return img_tensor, original_size

@torch.no_grad()
def edit_image(model, ref_image, source_pH, target_pH, denoising_strength=0.5, num_steps=100, cfg_scale=3.0, seed=None):
    """Upraví referenční obrázek pomocí Contrastive Guidance."""
    if seed is not None:
        torch.manual_seed(seed)
        
    # Nyní normalizujeme obě hodnoty pH
    pH_source_norm = normalize_pH(torch.tensor([source_pH])).to(DEVICE)
    pH_target_norm = normalize_pH(torch.tensor([target_pH])).to(DEVICE)
    
    t_start = 1.0 - denoising_strength
    noise = torch.randn_like(ref_image)
    x = (1 - t_start) * noise + t_start * ref_image 
    
    start_step = int(t_start * num_steps)

    for i in range(start_step, num_steps):
        t = torch.full((1,), i / num_steps, device=DEVICE)
        
        #  Predikce pro původní pH (udržuje vlákno na místě)
        v_source = model(x, t, pH_source_norm)
        
        #  Predikce pro cílové pH (kam se chceme dostat)
        v_target = model(x, t, pH_target_norm)
        
        # ┌─ REVIEW ─────────────────────────────────────────────
        #   tahle "contrastive guidance" interpoluje mezi dvěma
        #   PODMÍNĚNÝMI rychlostmi (source vs. target pH) a říká tomu cfg_scale.
        #   To zaměňuje guidance (cond vs. NULL) s interpolací mezi podmínkami —
        #   funguje, ale těžko se ladí a nedá se srovnat s literaturou.
        #   porovnej s klasickým SDEdit — zašum referenci na úroveň
        #   danou denoising_strength a denoise rovnou na CÍLOVÉ pH s běžným CFG
        #   (cond vs. null). Pokud chceš tenhle přístup zachovat, pojmenuj
        #   parametr jinak než cfg_scale, ať je jasné, že to není CFG.
        # └──────────────────────────────────────────────────────
        # Vektorový rozdíl izoluje čistě vliv pH na morfologii
        v_cfg = v_source + cfg_scale * (v_target - v_source)
        
        # Eulerův krok
        x = x + v_cfg * (1.0 / num_steps)
    
    out= (x.clamp(-1, 1) + 1) / 2
    # ┌─ REVIEW ─────────────────────────────────────────────
    # ✗ ZÁPLATA: napevno zesílený kontrast 1.5 je kosmetika, která maskuje, že
    #   model vrací málo kontrastní/rozmazaný výstup. Navíc zkresluje "mapu
    #   rozdílů" níže (rozdíl pak měří i tvůj kontrastní filtr, ne jen efekt pH).
    # └──────────────────────────────────────────────────────
    out = TF.adjust_contrast(out, 1.5)  # Zesílení kontrastu pro lepší vizualizaci
    return out.clamp(0, 1)

def visualize_difference(original_tensor, edited_tensor, original_size):
    """Zobrazí matplotlib okno s originálem, výsledkem a mapou rozdílů."""
    # Odříznutí vycpávky (paddingu) z obou tenzorů
    orig_w, orig_h = original_size
    orig_crop = original_tensor[:, :, :orig_h, :orig_w]
    edit_crop = edited_tensor[:, :, :orig_h, :orig_w]
    
    # Převod na CPU a denormalizace originálu (z [-1, 1] na [0, 1]) pro zobrazení
    orig_img = (orig_crop.squeeze().cpu() + 1) / 2
    edit_img = edit_crop.squeeze().cpu()
    
    # Výpočet absolutního rozdílu
    diff_map = torch.abs(orig_img - edit_img)
    
    # Vytvoření vizualizace (grafu)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    #  Originál
    axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Původní obrázek")
    axes[0].axis('off')
    
    #  Výsledek
    axes[1].imshow(edit_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Upraveno modelem")
    axes[1].axis('off')

    im_diff = axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title("Mapa rozdílů (Absolutní změna)")
    axes[2].axis('off')
    
    # Přidání barevné škály
    fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def main():
    checkpoint_path = "checkpoints/cfm_best_ema.pt"
    # ┌─ REVIEW ─────────────────────────────────────────────
    #   parametry běhu se čtou přes input(). Skript nejde spustit
    #   dávkově, v cronu ani reprodukovat (nikde se neuloží, co bylo zadáno) - to je ale jen detail
    # └──────────────────────────────────────────────────────
    ref_image_path = input("Zadej cestu k referenčnímu obrázku (např. 'data/ref_image.png'): ")
    source_pH = float(input(f"Zadej výchozí pH referenčního obrázku (mezi {PH_MIN} a {PH_MAX}): "))
    target_pH = float(input(f"Zadej cílové pH pro úpravu (mezi {PH_MIN} a {PH_MAX}): "))
    
    # Necháme uživatele zadat i sílu (výchozí dáme na 0.65 pro změnu tvaru)
    strength_input = input("Zadej sílu úpravy [0.1 - 0.9] (Enter pro 0.65): ")
    denoising_strength = float(strength_input) if strength_input.strip() else 0.65
    cfg = float(input("Classifier-Free Guidance (např. 4.0): ") or 4.0)
    num_steps = int(input("Počet kroků pro úpravu (např. 100): ") or 100)
    
    if not os.path.exists(checkpoint_path):
        print(f"Chyba: Checkpoint {checkpoint_path} neexistuje.")
        return
        
    if not os.path.exists(ref_image_path):
        print(f"Referenční obrázek {ref_image_path} nebyl nalezen. Uprav cestu ve skriptu.")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs_img2img", exist_ok=True)
    
    ref_image, original_size = load_and_preprocess_image(ref_image_path)
    print(f"Načten obrázek s původním rozlišením: {original_size[0]}x{original_size[1]}")
    
    edited_img = edit_image(
        model, 
        ref_image, 
        source_pH=source_pH,  
        target_pH=target_pH, 
        denoising_strength=denoising_strength, 
        num_steps=num_steps,
        cfg_scale=cfg,         
        seed=None
    )
    
    visualize_difference(ref_image, edited_img, original_size)
    
    orig_w, orig_h = original_size
    edited_crop_for_save = edited_img[:, :, :orig_h, :orig_w]
    
    save_path = f"outputs_img2img/edited_pH_{target_pH}_str_{denoising_strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Výsledek uložen do: {save_path}")

if __name__ == "__main__":
    main()