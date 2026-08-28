import os
import hashlib
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

from waviness import waviness as measure_waviness

class MicrotubuleDataset(Dataset):
    def __init__(self, root_dir, is_train=True, val_split_ratio=0.2, min_height=0):
        """min_height drops sources shorter than this many pixels. Very short crops are a
        fibre with almost no background around it: they carry little signal, trace
        unreliably, and - because framing.py now grows short crops with synthesised
        background instead of mirroring them - they are the samples that would need more
        synthetic rows than they have real ones to donate grain from."""
        self.root_dir = root_dir
        self.samples = []

        self.base_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        for ph_folder in os.listdir(root_dir):
            ph_dir = os.path.join(root_dir, ph_folder)
            if os.path.isdir(ph_dir):
                try:
                    ph_val = float(ph_folder)
                    all_images = [img for img in os.listdir(ph_dir) if img.endswith('.png')]
                    all_images.sort()

                    for img_name in all_images:
                        if min_height and Image.open(os.path.join(ph_dir, img_name)).size[1] < min_height:
                            continue
                        base_name = img_name.split('_crop')[0]

                        hash_hex = hashlib.md5(base_name.encode('utf-8')).hexdigest()
                        hash_val = int(hash_hex, 16) % 100

                        # val/train split based on hash value
                        is_val_sample = hash_val < (val_split_ratio * 100)

                        if is_train and not is_val_sample:
                            img_path = os.path.join(ph_dir, img_name)
                            self.samples.append((img_path, ph_val, self._measure(img_path)))
                        elif not is_train and is_val_sample:
                            img_path = os.path.join(ph_dir, img_name)
                            self.samples.append((img_path, ph_val, self._measure(img_path)))

                except ValueError:
                    continue

        self.is_train = is_train

    def _measure(self, img_path):
        """Waviness of the full (uncropped) source crop, measured once at dataset-build
        time - reuses the exact same real-pixel measurement waviness.py already computes
        for calibration, not a synthetic label. NaN where trace_fibre can't confidently
        lock onto a fibre (common on the smallest crops): fed to the model as the same
        'missing' sentinel pH's own null conditioning already uses, not a separate
        mechanism.
        """
        image = Image.open(img_path).convert('L')
        image_tensor = self.base_transform(image)
        wav = measure_waviness(image_tensor.unsqueeze(0))
        return float(wav) if wav is not None else float("nan")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ph, wav = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image_tensor = self.base_transform(image)
        return (image_tensor, torch.tensor(ph, dtype=torch.float32),
               torch.tensor(wav, dtype=torch.float32))