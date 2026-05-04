import os
import re
import cv2
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path

# ================= NASTAVENÍ CEST =================
XML_PATH = 'annotations.xml'       # Cesta k anotačnímu XML souboru
DATA_DIR = './Data_package1'                # Kořenová složka s daty
OUTPUT_DIR = './cropped_output'    # Složka pro vyříznuté obrázky

# ====== ZOBRAZENÍ A VÝBĚR KANÁLŮ ======
NORMALIZE_OUTPUT = True            # True = obrázky budou projasněné (8-bit)
CHANNELS_TO_KEEP = [0]             # [0] vyexportuje pouze první frame/kanál. 
# ==================================================

def normalize_name(name):
    """Sjednotí názvy z CVATu (.mp4) a z disku (.tif, nebo bez koncovky)."""
    name = re.sub(r'\.(mp4|tif|tiff|zip|roi)$', '', name, flags=re.IGNORECASE)
    name = name.replace('.', '_')
    return name.lower()

def parse_cvat_xml(xml_path):
    """Zpracuje CVAT XML, srovná číslování framů od nuly a vrátí anotace."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    tasks = {}
    for task in root.findall('.//meta/project/tasks/task'):
        t_id = task.find('id').text
        t_name = task.find('name').text
        tasks[t_id] = normalize_name(t_name)
        
    task_base_frame = {}
    for image in root.findall('.//image'):
        t_id = image.get('task_id')
        if t_id not in tasks:
            continue
        f_idx = int(image.get('name').replace('frame_', ''))
        if t_id not in task_base_frame or f_idx < task_base_frame[t_id]:
            task_base_frame[t_id] = f_idx
            
    annotations = {}
    for image in root.findall('.//image'):
        t_id = image.get('task_id')
        if t_id not in tasks:
            continue
            
        task_norm_name = tasks[t_id]
        global_frame_idx = int(image.get('name').replace('frame_', ''))
        relative_frame_idx = global_frame_idx - task_base_frame[t_id]
        
        boxes = []
        for box in image.findall('box'):
            xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
            xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
            boxes.append((int(xtl), int(ytl), int(xbr), int(ybr)))
            
        if not boxes:
            continue
            
        if task_norm_name not in annotations:
            annotations[task_norm_name] = {}
        annotations[task_norm_name][relative_frame_idx] = boxes
        
    return annotations

def read_frame(data_source, frame_idx):
    """Vrátí konkrétní snímek ze seznamu datových zdrojů."""
    if data_source is not None and frame_idx < len(data_source):
        frame = data_source[frame_idx]
        if isinstance(frame, str):
            return cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        return frame
    return None

def adjust_contrast(img):
    """Roztáhne kontrast pro zviditelnění mikroskopických dat."""
    img_float = img.astype(np.float32)
    p1, p99 = np.percentile(img_float, (1, 99))
    
    if p99 > p1: 
        img_norm = np.clip((img_float - p1) / (p99 - p1) * 255.0, 0, 255)
    else:
        img_norm = np.clip(img_float, 0, 255)
        
    return img_norm.astype(np.uint8)

def process_data(data_dir, output_dir, annotations):
    """Projde složky, spáruje data s anotacemi a vyřízne obdélníky podle pH."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for ph_folder in data_path.iterdir():
        if not ph_folder.is_dir() or ph_folder.name == 'ROIs':
            continue
            
        # Vytáhne z názvu složky pouze hodnotu pH (např. z "HEPES 6.8" udělá "6.8")
        ph_match = re.search(r'(\d+\.\d+)', ph_folder.name)
        if ph_match:
            ph_value = ph_match.group(1)
        else:
            ph_value = "unknown_pH"
            
        print(f"\nZpracovávám složku: {ph_folder.name} -> Ukládám do pH: {ph_value}")
        ph_out_dir = out_path / ph_value
        ph_out_dir.mkdir(exist_ok=True)
        
        for item in ph_folder.iterdir():
            if item.name == 'Thumbs.db':
                continue
                
            norm_name = normalize_name(item.name)
            
            if norm_name not in annotations:
                continue
                
            task_annotations = annotations[norm_name]
            data_source = None
            
            if item.is_file():
                ret, pages = cv2.imreadmulti(str(item), flags=cv2.IMREAD_UNCHANGED)
                if ret and len(pages) > 1:
                    data_source = pages
                else:
                    try:
                        arr = tifffile.imread(str(item))
                        arr = np.squeeze(arr)
                        if arr.ndim >= 3 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                            data_source = [arr[i] for i in range(arr.shape[0])]
                        else:
                            data_source = [arr]
                    except Exception:
                        img = cv2.imread(str(item), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            data_source = [img]
            elif item.is_dir():
                data_source = sorted([str(f) for f in item.iterdir() if f.is_file() and f.name != 'Thumbs.db'])
            
            if not data_source:
                continue

            for frame_idx, boxes in task_annotations.items():
                if frame_idx not in CHANNELS_TO_KEEP:
                    continue
                    
                frame_img = read_frame(data_source, frame_idx)
                
                if frame_img is None:
                    continue
                
                max_y, max_x = frame_img.shape[:2]
                
                for box_idx, (xtl, ytl, xbr, ybr) in enumerate(boxes):
                    xtl, ytl = max(0, int(xtl)), max(0, int(ytl))
                    xbr, ybr = min(max_x, int(xbr)), min(max_y, int(ybr))
                    
                    crop = frame_img[ytl:ybr, xtl:xbr]
                    
                    if crop.size == 0:
                        continue
                    
                    if NORMALIZE_OUTPUT:
                        crop = adjust_contrast(crop)
                        
                    base_name = item.stem if item.is_file() else item.name
                    out_filename = f"{base_name}_frame{frame_idx:04d}_crop{box_idx:02d}.png"
                    out_filepath = ph_out_dir / out_filename
                    
                    cv2.imwrite(str(out_filepath), crop)

if __name__ == "__main__":
    print("1/2 Načítám a rovnám indexy v XML anotacích...")
    cvat_annotations = parse_cvat_xml(XML_PATH)
    
    print("\n2/2 Začínám řezat obrázky...")
    process_data(DATA_DIR, OUTPUT_DIR, cvat_annotations)
    
    print("\nHotovo! Vyříznuté snímky organizované podle pH najdeš ve složce:", OUTPUT_DIR)