import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class MicrotubuleDataset(Dataset):
    def __init__(self, root_dir, is_train=True, val_samples_per_class=5):
        """
        Args:
            root_dir: Cesta ke kořenové složce s daty (např. 'data/cropped/cropped_output').
            is_train: Pokud je True, načte trénovací data. Pokud False, načte validační.
            val_samples_per_class: Počet obrázků z každého pH, které se vyhradí pro validaci.
        """
        self.root_dir = root_dir
        self.samples = []
        
        # Procházení struktury složek s pH
        for ph_folder in os.listdir(root_dir):
            ph_dir = os.path.join(root_dir, ph_folder)
            if os.path.isdir(ph_dir):
                try:
                    ph_val = float(ph_folder)
                    
                    # Načtení a seřazení všech .png obrázků ve složce
                    all_images = [img for img in os.listdir(ph_dir) if img.endswith('.png')]
                    all_images.sort() # Zajišťuje konzistentní rozdělení dat při každém spuštění
                    
                    # Rozdělení na validační a trénovací sadu
                    if is_train:
                        # Vezme vše kromě prvních 'val_samples_per_class'
                        selected_images = all_images[val_samples_per_class:]
                    else:
                        # Vezme pouze prvních 'val_samples_per_class'
                        selected_images = all_images[:val_samples_per_class]
                        
                    for img_name in selected_images:
                        self.samples.append((os.path.join(ph_dir, img_name), ph_val))
                except ValueError:
                    continue # Přeskočí složky, které nejsou číslem (např. skryté složky)

        # Definice augmentací pro trénink
        if is_train:
            self.transform = T.Compose([
                T.ToImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomChoice([T.RandomRotation(d) for d in [0, 90, 180, 270]]),
                T.RandomCrop(224),
                T.Resize(256, antialias=True),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        # Definice transformací pro validaci (bez náhodných změn)
        else:
            self.transform = T.Compose([
                T.ToImage(),
                T.Resize(256, antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ph = self.samples[idx]
        image = Image.open(img_path).convert('L') # Vynucení grayscale
        image = self.transform(image)
        return image, torch.tensor(ph, dtype=torch.float32)