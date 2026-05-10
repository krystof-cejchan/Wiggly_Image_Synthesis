import os
import numpy as np
from PIL import Image

def analyze_image_dimensions(base_dir):
    """
    Projde podsložky v zadaném adresáři, spočítá obrázky a vypíše 
    průměr (mean) a směrodatnou odchylku (std) pro jejich šířku a výšku.
    """
    print(f"Analyzuji složku: {base_dir}\n" + "="*40)
    
    # Získání seznamu složek a jejich seřazení (např. 5.8, 6.4, 6.8...)
    try:
        ph_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    except FileNotFoundError:
        print(f"Chyba: Složka '{base_dir}' nebyla nalezena.")
        return

    for ph_folder in ph_folders:
        folder_path = os.path.join(base_dir, ph_folder)
        
        widths = []
        heights = []
        
        # Procházení souborů v konkrétní pH složce
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                except Exception as e:
                    print(f"Varování: Nelze zpracovat obrázek {filename}: {e}")
        
        count = len(widths)
        
        # Výpočet a výpis statistik, pokud složka obsahuje obrázky
        if count > 0:
            mean_w = np.mean(widths)
            std_w = np.std(widths)
            mean_h = np.mean(heights)
            std_h = np.std(heights)
            
            print(f"pH hodnota: {ph_folder}")
            print(f"  • Počet obrázků: {count}")
            print(f"  • Šířka [px]   : Průměr = {mean_w:.2f}, Std = {std_w:.2f}")
            print(f"  • Výška [px]   : Průměr = {mean_h:.2f}, Std = {std_h:.2f}")
            print("-" * 40)
        else:
            print(f"pH hodnota: {ph_folder}")
            print("  • Nebyly nalezeny žádné obrázky.")
            print("-" * 40)

if __name__ == "__main__":
    # Cesta ke složkám s pH hodnotami podle tvé struktury
    target_directory = os.path.join("data", "cropped", "cropped_output")
    
    analyze_image_dimensions(target_directory)