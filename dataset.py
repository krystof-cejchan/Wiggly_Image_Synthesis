import os
import hashlib
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class MicrotubuleDataset(Dataset):
    def __init__(self, root_dir, is_train=True, val_split_ratio=0.2):
        self.root_dir = root_dir
        self.samples = []
        
        for ph_folder in os.listdir(root_dir):
            ph_dir = os.path.join(root_dir, ph_folder)
            if os.path.isdir(ph_dir):
                try:
                    ph_val = float(ph_folder)
                    all_images = [img for img in os.listdir(ph_dir) if img.endswith('.png')]
                    all_images.sort()
                    
                    for img_name in all_images:
                        # 1. Extrakce identifikátoru zdrojového snímku (odstraníme _cropXX)
                        # Předpokládá formát např. "exp1_frame0000_crop01.png"
                        base_name = img_name.split('_crop')[0]
                        
                        # 2. Deterministický hash (hash() se v Pythonu mění s každým spuštěním)
                        hash_hex = hashlib.md5(base_name.encode('utf-8')).hexdigest()
                        # Převedení posledních znaků hashe na int pro % 100
                        hash_val = int(hash_hex, 16) % 100
                        
                        # 3. Deterministické rozdělení do val/train
                        is_val_sample = hash_val < (val_split_ratio * 100)
                        
                        if is_train and not is_val_sample:
                            self.samples.append((os.path.join(ph_dir, img_name), ph_val))
                        elif not is_train and is_val_sample:
                            self.samples.append((os.path.join(ph_dir, img_name), ph_val))
                            
                except ValueError:
                    continue

        self.is_train = is_train
        
        self.base_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ph = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image_tensor = self.base_transform(image)
        return image_tensor, torch.tensor(ph, dtype=torch.float32)