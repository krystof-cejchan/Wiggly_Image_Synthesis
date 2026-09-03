"""Edit one reference crop to a whole range of pH values, saving every result to disk.

Headless by design: nothing is ever displayed. img2img.py's main() ends with a
plt.show() (its difference plot is meant for interactive use), so this calls
ph_warp.edit_to_pH directly instead of shelling out to that CLI. That also means the
checkpoint is loaded ONCE for the whole sweep rather than once per pH - at ~240MB per
load, shelling out 17 times is most of the runtime.

    python3 sweep_ph.py --ref_image data/cropped/cropped_output/5.8/<file>.png --source_pH 5.8

Writes into --out_dir (default outputs_img2img/ph_sweep/<image name>/):

    pH_00.0.png ... pH_16.0.png   one file per pH, captioned: the untouched source on
                              top, that pH's edit below, each labelled with its pH and
                              measured waviness
    contact_sheet.png         the source followed by every edit in order, same captions,
                              unless --no_contact_sheet
    raw/pH_00.0.png ...       the same edits with NO caption band, pixel-exact, for
                              measurement or further processing
    original.png              the source on its own, uncaptioned
    sweep.csv                 requested pH, target waviness, measured waviness

Most of the requested range is far outside the trained band (pH 5.8-8.8). That is the
point of the sweep, but see ph_control.py: the mapping past the ends is an extrapolation
of a fitted law, not evidence about real chemistry there.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import, including img2img's - never open a window

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from config import CHECKPOINT_PATH, DEVICE, PH_MAX, PH_MIN
from img2img import load_and_preprocess_image
from model import from_state_dict
from ph_control import describe as describe_pH
from ph_warp import edit_to_pH
from waviness import waviness as measure_waviness


def _font(size=15, bold=False):
    """DejaVu, which ships with matplotlib, so no system font has to be assumed.
    Falls back to PIL's bitmap default if that is somehow missing - the labels get
    small, but nothing crashes mid-sweep."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = os.path.join(os.path.dirname(font_manager.__file__),
                        "mpl-data", "fonts", "ttf", name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def label_panel(img01, heading, detail="", band=26):
    """Render one image with a caption band above it, as a PIL RGB image.

    Drawn with PIL rather than plotted through matplotlib on purpose: the pixels are
    copied through 1:1, so the fibre geometry the sweep exists to compare is never
    resampled, stretched to an axes aspect, or shrunk by figure margins. The caption is
    added as extra rows above the frame, never painted over the data.
    """
    array = (img01[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    frame = Image.fromarray(array, mode="L").convert("RGB")

    head_font, detail_font = _font(15, bold=True), _font(13)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    head_w = measure.textlength(heading, font=head_font)
    detail_w = measure.textlength(detail, font=detail_font) if detail else 0
    # The caption may well be wider than the crop - the median source here is under 300px
    # while a full caption runs to ~800px - so the canvas grows to fit the text rather than
    # letting it run off the edge. The image itself is never scaled to match.
    width = max(frame.width, int(6 + head_w + (10 + detail_w if detail else 0) + 6))

    canvas = Image.new("RGB", (width, frame.height + band), "white")
    canvas.paste(frame, (0, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), heading, fill=(0, 0, 0), font=head_font)
    if detail:
        draw.text((6 + head_w + 10, 6), detail, fill=(70, 70, 70), font=detail_font)
    return canvas


def stack_panels(panels, gap=6):
    """Stack labelled panels vertically, left-aligned, padding width to the widest.

    Only the width is padded. Heights are left exactly as they are, because the
    difference is real: an above-range edit legitimately comes back TALLER than the
    source, since the warp path grows the canvas to fit the waviness.
    """
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + gap
    return canvas


def ph_values(args):
    """The pH list to sweep: an explicit --pH wins, otherwise min..max in --pH_step."""
    if args.pH:
        return list(args.pH)
    values, ph = [], args.pH_min
    while ph <= args.pH_max + 1e-9:
        values.append(round(ph, 4))
        ph += args.pH_step
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref_image", required=True, help="the crop to edit")
    ap.add_argument("--source_pH", type=float, required=True,
                    help=f"the reference image's own pH (its folder name; trained range "
                         f"{PH_MIN}-{PH_MAX})")
    ap.add_argument("--pH_min", type=float, default=4.0)
    ap.add_argument("--pH_max", type=float, default=18.0)
    ap.add_argument("--pH_step", type=float, default=2)
    ap.add_argument("--pH", type=float, nargs="+",
                    help="explicit pH values, overriding --pH_min/--pH_max/--pH_step")
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs_img2img/ph_sweep/<reference image name>")
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--strength", type=float, default=0.65,
                    help="ignored by source-conditioned checkpoints, which start from noise")
    ap.add_argument("--contrastive_scale", type=float, default=3.0)
    ap.add_argument("--num_steps", type=int, default=200)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--contrast_mode", default="linear", choices=["linear", "gamma"])
    ap.add_argument("--solver", default="heun", choices=["euler", "heun"])
    ap.add_argument("--seed", type=int, default=None,
                    help="fix it to compare pH values without sampling noise between them")
    ap.add_argument("--geometry_mode", default="native", choices=["auto", "warp", "native"])
    ap.add_argument("--waviness_mode", default="relative", choices=["relative", "absolute"])
    ap.add_argument("--no_contact_sheet", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found "
              f"(see checkpoints/download_trained_model.txt).")
        return
    if not os.path.exists(args.ref_image):
        print(f"Reference image {args.ref_image} not found.")
        return

    out_dir = args.out_dir or os.path.join(
        "outputs_img2img", "ph_sweep",
        os.path.splitext(os.path.basename(args.ref_image))[0])
    os.makedirs(out_dir, exist_ok=True)

    model = from_state_dict(torch.load(args.checkpoint, map_location=DEVICE), DEVICE)
    model.eval()

    ref_image, original_size = load_and_preprocess_image(args.ref_image)
    source_waviness = measure_waviness(ref_image)
    print(f"source: {args.ref_image}  {original_size[0]}x{original_size[1]}  pH {args.source_pH}"
          f"  measured waviness "
          f"{f'{source_waviness:.2f}px' if source_waviness is not None else 'untraceable'}")

    original01 = (ref_image + 1) / 2
    vutils.save_image(original01, os.path.join(out_dir, "original.png"))
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    source_note = (f"waviness {source_waviness:.2f}px" if source_waviness is not None
                   else "waviness untraceable")
    source_panel = label_panel(
        original01, f"SOURCE  -  pH {args.source_pH:g}",
        f"{source_note}   {os.path.basename(args.ref_image)}")

    targets = ph_values(args)
    print(f"sweeping {len(targets)} pH values -> {out_dir}\n")

    rows, sheet = [], []
    for ph in targets:
        try:
            edited, info = edit_to_pH(
                model=model, ref_image=ref_image,
                source_pH=args.source_pH, target_pH=ph,
                denoising_strength=args.strength, num_steps=args.num_steps,
                contrastive_scale=args.contrastive_scale, seed=args.seed,
                contrast=args.contrast, contrast_mode=args.contrast_mode,
                solver=args.solver, extend_frame=True,
                geometry_mode=args.geometry_mode, waviness_mode=args.waviness_mode,
            )
        except Exception as exc:  # one bad pH should not abandon the rest of the sweep
            print(f"  pH {ph:>5}: FAILED - {type(exc).__name__}: {exc}")
            rows.append({"pH": ph, "target_waviness": "", "measured_waviness": "",
                         "mode": "failed", "note": f"{type(exc).__name__}: {exc}"})
            continue

        # crop back to the source width; an above-range edit may legitimately be TALLER
        # than the source, because the warp path grows the canvas to fit the waviness
        saved = edited[:, :, :, :original_size[0]]
        vutils.save_image(saved, os.path.join(raw_dir, f"pH_{ph:04.1f}.png"))

        measured = measure_waviness(saved * 2 - 1)
        target = info.get("target_waviness", info.get("target"))
        mode = info.get("mode", "?")
        band = ("inside trained range" if PH_MIN <= ph <= PH_MAX
                else f"extrapolated, outside {PH_MIN:g}-{PH_MAX:g}")
        detail = (f"target {target:.1f}px" if isinstance(target, (int, float)) else "target -")
        detail += (f"   measured {measured:.2f}px" if measured is not None
                   else "   measured: untraceable")
        detail += f"   {mode}   {band}"
        edit_panel = label_panel(
            saved, f"EDITED  ->  pH {ph:g}   (from pH {args.source_pH:g})", detail)

        path = os.path.join(out_dir, f"pH_{ph:04.1f}.png")
        stack_panels([source_panel, edit_panel]).save(path)

        print(f"  pH {ph:>5}: {mode:>11} | target "
              f"{f'{target:5.1f}px' if isinstance(target, (int, float)) else '    -  '} | "
              f"measured {f'{measured:5.2f}px' if measured is not None else 'untraceable':>11} | "
              f"{os.path.basename(path)}")
        rows.append({"pH": ph,
                     "target_waviness": f"{target:.3f}" if isinstance(target, (int, float)) else "",
                     "measured_waviness": f"{measured:.3f}" if measured is not None else "",
                     "mode": mode, "note": ""})
        sheet.append(edit_panel)

    csv_path = os.path.join(out_dir, "sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, ["pH", "target_waviness", "measured_waviness", "mode", "note"])
        writer.writeheader()
        writer.writerows(rows)

    n_edits = len(sheet)
    if sheet and not args.no_contact_sheet:
        stack_panels([source_panel] + sheet).save(
            os.path.join(out_dir, "contact_sheet.png"))

    print(f"\nwrote {n_edits} source+edit pairs + original.png + sweep.csv to {out_dir}")


if __name__ == "__main__":
    main()
