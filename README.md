Wiggly Image Synthesis
======================



The main inference script allows you to take an existing microtubule image, define its current pH, and synthesize how it would look at a target pH. It uses a global ODE integrator combined with a sliding window approach for the vector field to handle arbitrarily large and wide aspect ratios.

cmd:
```bash
python3 img2img.py --ref_image data/cropped/cropped_output/5.8/20260219_005_Ch3_pos2_MES_pH5_frame0000_crop00.png --source_pH 5.8 --target_pH 9.8 --num_steps 100 --strength 0.85 --contrast 2 --contrastive_scale 3.0
```


Important Arguments:
```bash
    --strength (default: 0.65): Controls the denoising strength (0.0 to 1.0). For thin microtubule structures, lower values (e.g., 0.35 - 0.45) are highly recommended to prevent the structure from breaking into disconnected segments.

    --contrastive_scale (default: 3.0): Controls how aggressively the target pH morphology (curviness) is applied.

    --num_steps (default: 100): Number of Euler integration steps. Higher = better quality but slower.

    --contrast (default: 1.0): Post-processing histogram stretch to restore deep blacks that might wash out during Flow Matching. (Values like 1.5 or 2.0 will darken the cells).
```

The script automatically generates a visual comparison plot (Original vs. Edited vs. Absolute Difference Map) and saves the final edited crop into the outputs_img2img/ directory.