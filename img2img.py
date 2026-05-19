import os
import torch
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
from model import ConditionalUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PH_MIN, PH_MAX = 5.8, 8.8

def normalize_pH(pH):
    """Normalizuje pH na interval [-1, 1]."""
    return 2 * (pH - PH_MIN) / (PH_MAX - PH_MIN) - 1

def load_and_preprocess_image(image_path):
    """Načte referenční obrázek a připraví ho pro síť."""
    transform = T.Compose([
        T.ToImage(),
        T.Resize((128, 128), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image)
    return image.unsqueeze(0).to(DEVICE) # Přidá batch dimenzi (1, 1, 128, 128)

@torch.no_grad()
def edit_image(model, ref_image, target_pH, denoising_strength=0.5, num_steps=50, cfg_scale=3.0, seed=42):
    """
    Upraví referenční obrázek podle cílového pH.
    
    Args:
        denoising_strength (float): Hodnota mezi 0.0 a 1.0.
            - 0.1 = nepatrná změna, obrázek zůstane skoro stejný.
            - 0.5 = střední změna, zachová se hrubá struktura, ale detaily se přizpůsobí.
            - 0.9 = velká změna, z původního obrázku zbudou jen základy.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    pH_norm = normalize_pH(torch.tensor([target_pH])).to(DEVICE)
    pH_null = torch.tensor([float("nan")], device=DEVICE)
    
    # Krok 1: Částečné zašumění referenčního obrázku
    # Určíme, ze kterého kroku 't' začneme integraci
    t_start = denoising_strength
    
    # Vygenerujeme šum
    noise = torch.randn_like(ref_image)
    
    # Smícháme obrázek se šumem (simulace Forward path)
    # x_t = (1 - t) * x_0 + t * x_1 
    # V Flow Matchingu x_1 jsou data, x_0 je šum
    x_t = (1 - t_start) * noise + t_start * ref_image 
    
    x = x_t
    
    # Krok 2: Odšumění směrem k novému cíli
    # Určíme, kolik kroků nám zbývá z celkového num_steps
    steps_remaining = int(num_steps * t_start)
    
    print(f"Začínám úpravu od kroku t={t_start:.2f} (zbývá {steps_remaining} integračních kroků)")
    
    for i in reversed(range(steps_remaining)):
        # Aktuální čas t jde od t_start do 0
        current_t = i / num_steps
        t = torch.full((1,), current_t, device=DEVICE)
        
        # CFG predikce (používá cílové pH)
        v_cond = model(x, t, pH_norm)
        v_uncond = model(x, t, pH_null)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Eulerův krok (zpětný chod integrace)
        # Během tréninku jsme se učili v_theta = x_1 (data) - x_0 (šum)
        # Nyní chceme jít od zašuměného x_t směrem k novým datům, takže šum odečítáme
        x = x - v_cfg * (1.0 / num_steps)
    
    # Denormalizace [-1, 1] → [0, 1]
    return (x.clamp(-1, 1) + 1) / 2

def main():
    checkpoint_path = "checkpoints/cfm_best_ema.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Chyba: Checkpoint {checkpoint_path} neexistuje.")
        return
        
    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs_img2img", exist_ok=True)
    
    # --- NASTAVENÍ ÚLOHY ---
    # Vyber si konkrétní referenční obrázek ze svých dat (zadej správnou cestu)
    # Příklad: Vem obrázek z pH 5.8 a zkus ho upravit, aby vypadal jako pH 8.8
    ref_image_path = "data/cropped/cropped_output/5.8/20260219_005_Ch3_pos1_MES_pH5_frame0000_crop00.png"
    target_pH = 8.8
    
    if not os.path.exists(ref_image_path):
        print(f"Referenční obrázek {ref_image_path} nebyl nalezen. Uprav cestu ve skriptu.")
        return
    
    # Načtení originálu
    ref_image = load_and_preprocess_image(ref_image_path)
    
    # Zkoušíme různé síly úprav, abychom viděli, co to dělá
    strengths = [0.2, 0.4, 0.6, 0.8]
    results = [ (ref_image.clamp(-1, 1) + 1) / 2 ] # Uložíme si originál na první pozici pro srovnání
    
    for strength in strengths:
        print(f"Generuji úpravu se silou {strength}...")
        edited_img = edit_image(model, ref_image, target_pH=target_pH, denoising_strength=strength)
        results.append(edited_img)
    
    # Spojíme obrázky (Originál + různé stupně úpravy) do mřížky vedle sebe
    all_images = torch.cat(results, dim=0)
    save_path = f"outputs_img2img/edit_to_pH_{target_pH}.png"
    
    # nrow = len(results) znamená, že obrázky budou v jedné řadě
    vutils.save_image(all_images, save_path, nrow=len(results))
    print(f"Výsledek uložen do: {save_path} (Zleva doprava: Originál, Síla 0.2, 0.4, 0.6, 0.8)")

if __name__ == "__main__":
    main()