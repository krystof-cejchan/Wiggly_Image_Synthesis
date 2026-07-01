import os
import subprocess
import itertools
import shutil

# ==========================================
# 1. ZÁKLADNÍ NASTAVENÍ (Neměnné parametry)
# ==========================================
REF_IMAGE = "data/cropped/cropped_output/5.8/20260219_006_Ch4_pos4_MES_pH5_frame0000_crop09.png"
SOURCE_PH = 5.8
TARGET_PH = 8.8

# Pro testování doporučuji krok snížit (např. na 150), abyste na 20 obrázků nečekal celou věčnost.
# Až najdete nejlepší kombo, můžete to na finální vygenerování zvednout zpět na 500.
NUM_STEPS = 150  

# ==========================================
# 2. MŘÍŽKA PARAMETRŮ (20 kombinací)
# ==========================================
# Síla destrukce (jak moc se povolí změnit původní umístění vlákna)
STRENGTHS = [0.30, 0.35, 0.40, 0.45] 

# Síla Contrastive Guidance (jak agresivně se má aplikovat vliv cílového pH/zvlnění)
SCALES = [3.0, 5.0, 7.0, 9.0, 11.0]

def main():
    # Vytvoření složky pro výsledky experimentů
    exp_dir = "outputs_img2img/experiments"
    os.makedirs(exp_dir, exist_ok=True)
    
    combinations = list(itertools.product(STRENGTHS, SCALES))
    total_runs = len(combinations)
    
    print(f"Zahajuji testování: {total_runs} celkových kombinací.")
    
    for i, (strength, scale) in enumerate(combinations, 1):
        print("-" * 50)
        print(f"Experiment {i}/{total_runs} | Strength: {strength} | Contrastive Scale: {scale}")
        
        # Sestavení příkazu pro terminál
        cmd = [
            "python3", "img2img.py",
            "--ref_image", REF_IMAGE,
            "--source_pH", str(SOURCE_PH),
            "--target_pH", str(TARGET_PH),
            "--num_steps", str(NUM_STEPS),
            "--strength", str(strength),
            "--contrastive_scale", str(scale),
            "--checkpoint", "checkpoints/cfm_best_ema2.pt"
        ]
        
        # Spuštění inference
        subprocess.run(cmd)
        
        # Cesta, kam img2img.py automaticky ukládá výsledek
        original_save_path = f"outputs_img2img/edited_pH_{TARGET_PH}_str_{strength}.png"
        
        # Nová cesta se specifikovaným contrastive_scale, aby nevznikl konflikt
        new_filename = f"pH_{TARGET_PH}_str_{strength}_scale_{scale}.png"
        new_save_path = os.path.join(exp_dir, new_filename)
        
        # Přesun a přejmenování souboru
        if os.path.exists(original_save_path):
            shutil.move(original_save_path, new_save_path)
            print(f"✅ Uloženo jako: {new_filename}")
        else:
            print(f"❌ Chyba: Výstupní soubor nebyl nalezen ({original_save_path})")

    print("-" * 50)
    print(f"Všechny experimenty dokončeny. Prohlédněte si složku '{exp_dir}'.")

if __name__ == "__main__":
    main()