import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class MicrotubuleDataset(Dataset):
    def __init__(self, root_dir, is_train=True, val_samples_per_class=5):
        self.root_dir = root_dir
        self.samples = []
        
        for ph_folder in os.listdir(root_dir):
            ph_dir = os.path.join(root_dir, ph_folder)
            if os.path.isdir(ph_dir):
                try:
                    ph_val = float(ph_folder)
                    all_images = [img for img in os.listdir(ph_dir) if img.endswith('.png')]
                    all_images.sort()
                    
                    if is_train:
                        selected_images = all_images[val_samples_per_class:]
                    else:
                        selected_images = all_images[:val_samples_per_class]
                        
                    for img_name in selected_images:
                        self.samples.append((os.path.join(ph_dir, img_name), ph_val))
                except ValueError:
                    continue

        if is_train:
            self.transform = T.Compose([
                T.ToImage(),
                # KLÍČOVÁ ZMĚNA: Zajišťuje, že obrázek je dostatečně velký před cropem
                # Změní kratší stranu minimálně na 136 px, zachová poměr stran
                T.Resize(136, antialias=True), 
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomChoice([T.RandomRotation(d) for d in [0, 90, 180, 270]]),
                # Teď už RandomCrop neselže, ani když byl původní obrázek menší
                T.RandomCrop(112),
                # Finální resize na cílových 128x128
                T.Resize((128, 128), antialias=True),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.ToImage(),
                # Pro validaci jen bezpečně změníme velikost na přesných 128x128
                T.Resize((128, 128), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ph = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, torch.tensor(ph, dtype=torch.float32)