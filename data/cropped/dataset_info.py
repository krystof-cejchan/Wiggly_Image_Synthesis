import os
import numpy as np
from PIL import Image

def analyze_image_dimensions(base_dir):
    """
    Walk the subfolders of the given directory, count the images, and print the
    mean and standard deviation of their width and height.
    """
    print(f"Analyzing folder: {base_dir}\n" + "="*40)
    
    # Collect the folder list and sort it (e.g. 5.8, 6.4, 6.8...)
    try:
        ph_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    except FileNotFoundError:
        print(f"Error: folder '{base_dir}' was not found.")
        return

    for ph_folder in ph_folders:
        folder_path = os.path.join(base_dir, ph_folder)
        
        widths = []
        heights = []
        
        # Walk the files inside this particular pH folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                except Exception as e:
                    print(f"Warning: cannot process image {filename}: {e}")
        
        count = len(widths)
        
        # Compute and print the statistics, if the folder contains any images
        if count > 0:
            mean_w = np.mean(widths)
            std_w = np.std(widths)
            mean_h = np.mean(heights)
            std_h = np.std(heights)
            
            print(f"pH value: {ph_folder}")
            print(f"  • Image count : {count}")
            print(f"  • Width  [px] : Mean = {mean_w:.2f}, Std = {std_w:.2f}")
            print(f"  • Height [px] : Mean = {mean_h:.2f}, Std = {std_h:.2f}")
            print("-" * 40)
        else:
            print(f"pH value: {ph_folder}")
            print("  • No images were found.")
            print("-" * 40)

if __name__ == "__main__":
    # Path to the pH-value folders, matching the project layout
    target_directory = os.path.join("data", "cropped", "cropped_output")
    
    analyze_image_dimensions(target_directory)