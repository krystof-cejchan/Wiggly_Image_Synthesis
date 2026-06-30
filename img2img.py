import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF
from config import PH_MIN, PH_MAX, DEVICE
from model import ConditionalUNet

#matplotlib.use('Qt5Agg')


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

def load_and_preprocess_image(image_path):
    """Načte referenční obrázek a převede jej na tenzor v rozsahu [-1, 1]."""
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)
    
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return img_tensor, original_size

def create_blending_mask(window_size, device):
    """Vytvoří 2D masku (Hann window) pro plynulé prolnutí překrývajících se oken."""
    window_1d = torch.hann_window(window_size, periodic=False, device=device)
    mask_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    return mask_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

@torch.no_grad()
def edit_image(model, ref_image, source_pH, target_pH, denoising_strength=0.5, num_steps=100, contrastive_scale=3.0, seed=None, window_size=128, stride=64):
    """
    Upraví obrázek libovolné velikosti pomocí globálního Flow Matching integrátoru 
    kombinovaného s lokálním výpočtem vektorového pole (Sliding Window).
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    pH_source_norm = normalize_pH(torch.tensor([source_pH])).to(DEVICE)
    pH_target_norm = normalize_pH(torch.tensor([target_pH])).to(DEVICE)
    
    _, _, h, w = ref_image.shape
    
    # Výpočet paddingu pro okno a stride
    target_h = max(window_size, ((h + stride - 1) // stride) * stride) + stride
    target_w = max(window_size, ((w + stride - 1) // stride) * stride) + stride
    
    total_pad_h = target_h - h
    total_pad_w = target_w - w
    
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left
    
    # Použití 'replicate' pro zabránění padnutí u extrémně úzkých obrázků
    padded_ref = F.pad(ref_image, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    
    t_start = 1.0 - denoising_strength
    noise = torch.randn_like(padded_ref)
    x = (1 - t_start) * noise + t_start * padded_ref  
    
    start_step = int(t_start * num_steps)
    mask = create_blending_mask(window_size, ref_image.device)
    
    for i in range(start_step, num_steps):
        t = torch.full((1,), i / num_steps, device=DEVICE)
        
        progress = (i - start_step) / max(1, num_steps - start_step)
        current_scale = contrastive_scale * (1.0 - progress) + 1.0 * progress
        
        v_source_global = torch.zeros_like(x)
        v_target_global = torch.zeros_like(x)
        weight_global = torch.zeros_like(x)
        
        for y in range(0, target_h - window_size + 1, stride):
            for x_idx in range(0, target_w - window_size + 1, stride):
                x_patch = x[:, :, y:y+window_size, x_idx:x_idx+window_size]
                
                v_src_patch = model(x_patch, t, pH_source_norm)
                v_tgt_patch = model(x_patch, t, pH_target_norm)
                
                v_source_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += v_src_patch * mask
                v_target_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += v_tgt_patch * mask
                weight_global[:, :, y:y+window_size, x_idx:x_idx+window_size] += mask
        
        v_source = v_source_global / weight_global.clamp(min=1e-8)
        v_target = v_target_global / weight_global.clamp(min=1e-8)
        
        v_dir = v_source + current_scale * (v_target - v_source)        
        x = x + v_dir * (1.0 / num_steps)
    
    # Oříznutí paddingu a denormalizace
    x_cropped = x[:, :, pad_top:pad_top+h, pad_left:pad_left+w]
    out = (x_cropped.clamp(-1, 1) + 1) / 2
    
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
    
    # Originál
    axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Původní obrázek")
    axes[0].axis('off')
    
    # Výsledek
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
    parser = argparse.ArgumentParser(description="Image-to-Image úprava obrázku pomocí Flow Matching modelu.")
    parser.add_argument("--ref_image", type=str, required=True, help="Cesta k referenčnímu obrázku (např. 'data/ref_image.png')")
    parser.add_argument("--source_pH", type=float, required=True, help=f"Výchozí pH referenčního obrázku (mezi {PH_MIN} a {PH_MAX})")
    parser.add_argument("--target_pH", type=float, required=True, help=f"Cílové pH pro úpravu (mezi {PH_MIN} a {PH_MAX})")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cfm_best_ema.pt", help="Cesta k checkpointu modelu")
    parser.add_argument("--strength", type=float, default=0.65, help="Síla úpravy [0.0 - 1.0] (odpovídá zašumění)")
    parser.add_argument("--contrastive_scale", type=float, default=4.0, help="Síla Contrastive Guidance (dříve cfg_scale)")
    parser.add_argument("--num_steps", type=int, default=100, help="Počet kroků pro úpravu")
    parser.add_argument("--seed", type=int, default=None, help="Náhodný seed pro reprodukovatelnost (volitelné)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Chyba: Checkpoint {args.checkpoint} neexistuje.")
        return
        
    if not os.path.exists(args.ref_image):
        print(f"Referenční obrázek {args.ref_image} nebyl nalezen. Zkontroluj zadanou cestu.")
        return

    model = ConditionalUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    
    os.makedirs("outputs_img2img", exist_ok=True)
    
    ref_image, original_size = load_and_preprocess_image(args.ref_image)
    print(f"Načten obrázek s původním rozlišením: {original_size[0]}x{original_size[1]}")
    
    # Použití Sliding Window Inference
    edited_img = edit_image(
        model=model, 
        ref_image=ref_image, 
        source_pH=args.source_pH,  
        target_pH=args.target_pH, 
        denoising_strength=args.strength, 
        num_steps=args.num_steps,
        contrastive_scale=args.contrastive_scale,         
        seed=args.seed
    )
    
    visualize_difference(ref_image, edited_img, original_size)
    
    orig_w, orig_h = original_size
    edited_crop_for_save = edited_img[:, :, :orig_h, :orig_w]
    
    save_path = f"outputs_img2img/edited_pH_{args.target_pH}_str_{args.strength}.png"
    vutils.save_image(edited_crop_for_save, save_path, nrow=1)
    print(f"Výsledek uložen do: {save_path}")

if __name__ == "__main__":
    main()