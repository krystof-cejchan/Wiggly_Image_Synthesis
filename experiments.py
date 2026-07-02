import os
import subprocess
import itertools
import shutil
import numpy as np


# ==========================================
# 1. ZÁKLADNÍ NASTAVENÍ
# ==========================================
REF_IMAGE = "data/cropped/cropped_output/5.8/20260219_006_Ch4_pos4_MES_pH5_frame0000_crop09.png"
SOURCE_PH = 5.8
NUM_STEPS = 150  

# ==========================================
# 2. MŘÍŽKA PARAMETRŮ
# ==========================================
TARGET_PHS = np.arange(4.0, 10.0, 0.2, dtype=np.float64).tolist()
STRENGTHS = np.arange(0.4, 0.9, 0.1, dtype=np.float64).tolist()
SCALES = [3.0, 5.0, 7.0, 9.0, 11.0]

def main():
    exp_dir = "outputs_img2img/experiments"
    os.makedirs(exp_dir, exist_ok=True)
    
    combinations = list(itertools.product(TARGET_PHS, STRENGTHS, SCALES))
    total_runs = len(combinations)
    
    print(f"Zahajuji testování: {total_runs} celkových kombinací.")
    
    for i, (target_ph, strength, scale) in enumerate(combinations, 1):
        print("-" * 50)
        print(f"Experiment {i}/{total_runs} | Target pH: {target_ph} | Strength: {strength} | Contrastive Scale: {scale}")
        
        cmd = [
            "python3", "img2img.py",
            "--ref_image", REF_IMAGE,
            "--source_pH", str(SOURCE_PH),
            "--target_pH", str(target_ph),
            "--num_steps", str(NUM_STEPS),
            "--strength", str(strength),
            "--contrastive_scale", str(scale),
            "--contrast", "2.0"
        ]
        
        subprocess.run(cmd)
        
        original_save_path = f"outputs_img2img/edited_pH_{target_ph}_str_{strength}.png"
        new_filename = f"pH_{target_ph}_str_{strength}_scale_{scale}.png"
        new_save_path = os.path.join(exp_dir, new_filename)
        
        if os.path.exists(original_save_path):
            shutil.move(original_save_path, new_save_path)
            print(f"Uloženo jako: {new_filename}")
        else:
            print(f"Chyba: Výstupní soubor nebyl nalezen ({original_save_path})")

    print("-" * 50)
    print(f"Experimenty dokončeny. Výsledky jsou ve složce '{exp_dir}'.")

if __name__ == "__main__":
    main()