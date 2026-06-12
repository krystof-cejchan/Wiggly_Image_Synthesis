import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from model import ConditionalUNet

#matplotlib.use('Qt5Agg')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    # Zjištění aktuálních rozměrů a výpočet vycpávky (paddingu) do nejbližšího násobku 16
    _, _, h, w = img_tensor.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=-1.0)
    
    return img_tensor, original_size

@torch.no_grad()
def edit_image(model, ref_image, target_pH, denoising_strength=0.5, num_steps=50, cfg_scale=3.0, seed=42):
    """Upraví referenční obrázek podle cílového pH."""
    if seed is not None:
        torch.manual_seed(seed)
        
    pH_norm = normalize_pH(torch.tensor([target_pH])).to(DEVICE)
    pH_null = torch.tensor([float("nan")], device=DEVICE)
    
    t_start = 1.0 - denoising_strength
    noise = torch.randn_like(ref_image)
    x = (1 - t_start) * noise + t_start * ref_image 
    
    start_step = int(t_start * num_steps)
    print(f"Začínám úpravu od kroku {start_step}/{num_steps} (t={t_start:.2f}) směrem k t=1.0")
    
    for i in range(start_step, num_steps):
        t = torch.full((1,), i / num_steps, device=DEVICE)
        
        v_cond = model(x, t, pH_norm)
        v_uncond = model(x, t, pH_null)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        x = x + v_cfg * (1.0 / num_steps)
    
    return (x.clamp(-1, 1) + 1) / 2

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
    
    # 1. Originál
    axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Původní obrázek")
    axes[0].axis('off')
    
    # 2. Výsledek
    axes[1].imshow(edit_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Upraveno modelem")
    axes[1].axis('off')
    
    # 3. Mapa rozdílů (Heatmap)
    # Použijeme barvy (cmap='hot' nebo 'inferno'), aby byl rozdíl dobře vidět.
    # Černá = žádná změna, Červená/Žlutá/Bílá = velká změna.
    im_diff = axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title("Mapa rozdílů (Absolutní změna)")
    axes[2].axis('off')
    
    # Přidání barevné škály
    fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    print("Otevírám okno s vizualizací. Pro pokračování okno zavři.")
    plt.show()

def main():
    checkpoint_path = "checkpoints/cfm_best_ema.pt"
    ref_image_path = "data/cropped/cropped_output/6.8/20260219_003_Ch1_pos1_pH6.8_frame0000_crop01.png"
    
    target_pH = float(input(f"Zadej cílové pH pro úpravu (mezi {PH_MIN} a {PH_MAX}): "))
    denoising_strength = 0.9
    num_steps = 1000         
    
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
    
    # Načtení dat (ref_image je za-padovaný tenzor v rozsahu [-1, 1])
    ref_image, original_size = load_and_preprocess_image(ref_image_path)
    print(f"Načten obrázek s původním rozlišením: {original_size[0]}x{original_size[1]}")
    
    print(f"Generuji úpravu na pH {target_pH} se silou {denoising_strength}...")
    edited_img = edit_image(
        model, 
        ref_image, 
        target_pH=target_pH, 
        denoising_strength=denoising_strength,
        num_steps=num_steps
    )
    
    # 1. Zobrazení interaktivní vizualizace
    visualize_difference(ref_image, edited_img, original_size)
    
    # 2. Uložení samotného výsledku (bez heatmapy) na disk
    orig_w, orig_h = original_size
    edited_crop_for_save = edited_img[:, :, :orig_h, :orig_w]
    
    save_path = f"outputs_img2img/edited_pH_{target_pH}_str_{denoising_strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Výsledek uložen do: {save_path}")

if __name__ == "__main__":
    main()