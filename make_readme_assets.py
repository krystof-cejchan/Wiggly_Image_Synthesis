"""Regenerate the animations and figures the README embeds, into assets/.

Everything here is derived from the checkpoint at config.CHECKPOINT_PATH, so re-run it
after a retrain rather than leaving stale pictures in the README - the numbers printed
into the captions are measured off the frames being shown, and a figure whose caption
disagrees with the current model is worse than no figure.

    python3 make_readme_assets.py                 # all three assets
    python3 make_readme_assets.py --only denoise  # just one

Produces:
    assets/denoise.gif    the reverse-time ODE running from pure Gaussian noise to a
                          microtubule, filmed at three pH values from ONE shared seed, so
                          the only thing separating the three rows is the conditioning
    assets/ph_sweep.gif   one real pH 5.8 crop edited across pH 4-16 and back, with a dial
                          marking where each request sits relative to the trained band
    assets/ph_ladder.png  the same sweep as a static strip, for READMEs rendered without
                          animation

GIFs are written with a reduced palette and a deliberately uneven frame schedule (sparse
through the noisy opening, dense once structure appears): grain is close to incompressible
in GIF's LZW, so a uniform schedule spends most of the file on frames that all look like
the same static.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # before anything drags in pyplot - nothing here may open a window

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import CHECKPOINT_PATH, DEVICE, PH_MAX, PH_MIN
from img2img import load_and_preprocess_image
from model import from_state_dict
from ph_warp import edit_to_pH
from sample import sample
# Shared with sweep_ph on purpose: the sweep figures and the README figures should be set
# in the same face at the same sizes, so a reader moving between them sees one house style.
from sweep_ph import _font
from waviness import waviness as measure_waviness

ASSETS = "assets"
INK, MUTED, RULE = (17, 17, 17), (105, 105, 105), (208, 208, 208)
ACCENT, TRAINED = (196, 62, 34), (108, 152, 196)

# A straight, wide, cleanly-traceable pH 5.8 crop (478x55, 1.84px of centreline rms
# against the 4.12px population mean at that pH). Chosen for the demo because it starts
# near-flat: a source that is already wavy shows nothing when it is asked to get wavier.
DEMO_SOURCE = ("data/cropped/cropped_output/5.8/"
               "20260219_005_Ch3_pos3_MES_pH5_frame0000_crop00.png")
DEMO_SOURCE_PH = 5.8


def to_gray(img01, scale=2):
    """(1,1,H,W) in [0,1] -> uint8 HxW, upscaled by nearest neighbour.

    Nearest, never bilinear: these figures exist to show fibre geometry and film grain,
    and a smooth resample invents intermediate pixels in exactly the structures the
    reader is being asked to judge.
    """
    array = (img01[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    frame = Image.fromarray(array, mode="L")
    if scale != 1:
        frame = frame.resize((frame.width * scale, frame.height * scale), Image.NEAREST)
    return frame


def text(draw, xy, string, size=14, bold=False, fill=INK):
    draw.text(xy, string, font=_font(size, bold=bold), fill=fill)


def text_w(string, size=14, bold=False):
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(
        string, font=_font(size, bold=bold))


def ph_dial(width, ph, lo=4.0, hi=16.0, height=42):
    """A pH ruler with the TRAINED band shaded and a marker at the current request.

    The whole point of the sweep is that most of it is extrapolation, so the figure says
    so continuously rather than in a footnote: the shaded segment is the only stretch the
    model has ever seen data from.
    """
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    pad, y = 58, 26

    def x_of(value):
        return pad + (width - 2 * pad) * (value - lo) / (hi - lo)

    draw.rectangle([x_of(PH_MIN), y - 5, x_of(PH_MAX), y + 5], fill=(226, 236, 246))
    draw.line([pad, y, width - pad, y], fill=RULE, width=3)
    draw.line([x_of(PH_MIN), y, x_of(PH_MAX), y], fill=TRAINED, width=3)

    for tick in range(int(lo), int(hi) + 1, 2):
        x = x_of(tick)
        draw.line([x, y + 6, x, y + 10], fill=RULE, width=1)
        text(draw, (x - text_w(str(tick), 11) / 2, y + 11), str(tick), 11, fill=MUTED)

    text(draw, (x_of((PH_MIN + PH_MAX) / 2) - text_w("trained", 11, True) / 2, y - 22),
         "trained", 11, bold=True, fill=TRAINED)
    text(draw, (6, y - 6), "pH", 12, bold=True, fill=MUTED)

    x = x_of(min(max(ph, lo), hi))
    draw.polygon([(x, y - 7), (x - 6, y - 17), (x + 6, y - 17)], fill=ACCENT)
    draw.line([x, y - 7, x, y + 7], fill=ACCENT, width=3)
    return panel


def progress_bar(width, fraction, label, height=34):
    """Where the ODE is along its trajectory, as a filled track plus a caption."""
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    text(draw, (6, 5), label, 14, bold=True)
    x0 = int(text_w(label, 14, True)) + 20
    x1 = width - 10
    if x1 - x0 > 40:
        draw.rectangle([x0, 12, x1, 20], fill=(233, 233, 233))
        draw.rectangle([x0, 12, x0 + (x1 - x0) * fraction, 20], fill=ACCENT)
    return panel


def stack(panels, width=None, gap=0, bg="white"):
    """Stack panels vertically, left-aligned, on a canvas of a fixed width."""
    width = width or max(p.width for p in panels)
    canvas = Image.new("RGB", (width, sum(p.height for p in panels) + gap * (len(panels) - 1)), bg)
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + gap
    return canvas


def shared_palette(frames, colors, reserved):
    """One palette for the whole animation, with the UI colours held out of the vote.

    Two problems solved at once. An adaptive palette is chosen by pixel count, and these
    frames are ~99.9% grayscale film grain, so the red marker and the blue trained-band
    rule simply lose the election and come back gray - the accents that make the figure
    readable are exactly the pixels a frequency-based quantiser discards. And a GIF with
    one global palette stores it once instead of per frame.

    So the grain gets `colors - len(reserved)` adaptive entries, sampled from a montage of
    frames spanning the whole trajectory (the opening is nearly all mid-gray noise and the
    end is not, and a palette fit to either alone posterises the other), and the chrome
    colours are appended verbatim.
    """
    picks = [frames[i] for i in
             sorted({0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1})]
    montage = Image.new("RGB", (picks[0].width, sum(p.height for p in picks)))
    y = 0
    for frame in picks:
        montage.paste(frame, (0, y))
        y += frame.height

    n_free = colors - len(reserved)
    values = montage.quantize(colors=n_free, method=Image.MEDIANCUT).getpalette()[:3 * n_free]
    for colour in reserved:
        values += list(colour)
    values += [0] * (768 - len(values))
    palette = Image.new("P", (1, 1))
    palette.putpalette(values)
    return palette


def write_gif(path, frames, durations, colors=40):
    """Save an animation against one shared, accent-preserving palette.

    Film grain is close to incompressible under LZW, so a trajectory that starts in pure
    noise costs a roughly flat per-frame price however it is encoded. Measured on the
    denoise animation: dropping the palette from 256 to 32 colours saves 24%, while
    halving the frame count saves 50%. Frames are the lever that matters, which is what
    capture_schedule spends carefully; the palette is chosen for legibility instead.
    """
    palette = shared_palette(frames, colors,
                             [ACCENT, TRAINED, RULE, MUTED, INK, (255, 255, 255),
                              (226, 236, 246), (233, 233, 233)])
    quantised = [f.convert("RGB").quantize(palette=palette, dither=Image.NONE)
                 for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:],
                      duration=durations, loop=0, optimize=True, disposal=1)
    print(f"  {path}  {len(frames)} frames  {os.path.getsize(path) / 1e6:.2f} MB  "
          f"{frames[0].width}x{frames[0].height}")


def capture_schedule(num_steps, n_frames, bias=0.45):
    """Step indices to film, sparse early and dense late.

    The opening of the trajectory is pure noise that barely changes frame to frame while
    being the most expensive thing in the file to store; the structure appears late. So
    the schedule is stretched towards the end (u**bias with bias<1) instead of uniform.
    """
    u = np.linspace(0.0, 1.0, n_frames)
    steps = np.unique(np.round(np.clip(u ** bias, 0, 1) * num_steps).astype(int))
    return set(int(s) for s in steps if s > 0)


def make_denoise_gif(model, args):
    """Film the from-noise ODE at three pH values sharing one seed."""
    phs = [5.8, 7.3, 8.8]
    # 48x384 and not a square: a config.TRAIN_SIZES entry, i.e. frame geometry the model
    # was actually trained on. The shortest of them, because three stacked rows of grain
    # is what sets this file's size.
    height, width, scale = 48, 384, 2
    wanted = capture_schedule(args.denoise_steps, args.denoise_frames)

    rows, finals = {}, {}
    for ph in phs:
        captured = {}

        def grab(step, x, store=captured):
            if step in wanted:
                store[step] = to_gray((x.clamp(-1, 1) + 1) / 2, scale)

        final = sample(model, pH_query=ph, num_steps=args.denoise_steps,
                       seed=args.denoise_seed, height=height, width=width,
                       geometry_mode="embedding", on_step=grab)
        captured[args.denoise_steps] = to_gray(final, scale)
        rows[ph], finals[ph] = captured, measure_waviness(final * 2 - 1)
        rms = f"{finals[ph]:.2f}px" if finals[ph] is not None else "untraceable"
        print(f"  pH {ph}: {len(captured)} frames captured, final rms {rms}")

    order = sorted(set().union(*(set(r) for r in rows.values())))
    canvas_w = width * scale
    labels = {}
    for ph in phs:
        panel = Image.new("RGB", (canvas_w, 22), "white")
        draw = ImageDraw.Draw(panel)
        head = f"pH {ph}"
        text(draw, (6, 3), head, 14, bold=True)
        # The measured rms on each row, so the progression is legible as a number and not
        # only as a shape - pH 5.8 and 7.3 genuinely differ by very little here, which is
        # what the real crops do too (bucket means 3.92px at 5.8 against 4.31px at 7.2).
        note = (f"finishes at {finals[ph]:.2f}px rms" if finals[ph] is not None
                else "final centreline untraceable")
        if ph == phs[0]:
            note += "   -   all three rows share one noise seed"
        text(draw, (6 + text_w(head, 14, True) + 12, 4), note, 12, fill=MUTED)
        labels[ph] = panel

    frames, durations = [], []
    for step in order:
        fraction = step / args.denoise_steps
        parts = [progress_bar(canvas_w, fraction,
                              f"noise  ->  microtubule     t = {fraction:.2f}")]
        for ph in phs:
            parts += [labels[ph], rows[ph][step]]
        frames.append(stack(parts, canvas_w))
        durations.append(90)
    # Hold on the finished image: without a pause the loop snaps back to static the
    # instant the picture resolves, which is the one frame a reader wants to look at.
    frames.append(frames[-1])
    durations[-1] = 60
    durations.append(2200)
    write_gif(os.path.join(ASSETS, "denoise.gif"), frames, durations)


def edit_across_pH(model, args):
    """Run the editor over every requested pH once.

    Returns (source in [0,1], its measured waviness, [(pH, edit, info, measured), ...]).
    Both the sweep animation and the static ladder are built from one call, because the
    edits are the only expensive part and they must agree frame for frame.
    """
    ref, size = load_and_preprocess_image(args.ref_image)
    source_waviness = measure_waviness(ref)
    print(f"  source {os.path.basename(args.ref_image)}  {size[0]}x{size[1]}  "
          f"pH {args.source_pH}  waviness {source_waviness:.2f}px")

    results = []
    for ph in args.pH:
        edited, info = edit_to_pH(
            model=model, ref_image=ref, source_pH=args.source_pH, target_pH=ph,
            num_steps=args.edit_steps, contrastive_scale=3.0, seed=args.seed,
            contrast=1.0, contrast_mode="linear", solver="heun", extend_frame=True,
            geometry_mode="auto", waviness_mode="relative")
        edited = edited[:, :, :, :size[0]]
        measured = measure_waviness(edited * 2 - 1)
        results.append((ph, edited, info, measured))
        print(f"  pH {ph:>5}: {info.get('mode', '?'):>7} | "
              f"target {target_of(info):5.1f}px | "
              f"measured {measured:5.2f}px")
    return (ref + 1) / 2, source_waviness, results


def target_of(info):
    """The requested centreline rms, whichever key this edit path reported it under.

    The geometry-channel path calls it "target" (it comes straight out of
    plan_target_line); the older scalar path calls it "target_waviness".
    """
    value = info.get("target", info.get("target_waviness"))
    return float(value) if isinstance(value, (int, float)) else float("nan")


def pad_to(frame, width, height):
    """Centre a rendered strip in a fixed canvas, so the animation never jumps.

    An above-range edit legitimately comes back taller than the source (the warp path
    grows the canvas to fit the excursion), and a GIF needs one size throughout.
    """
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
    return canvas


def sweep_panels(source01, results, scale=2):
    """Common layout for the sweep GIF and the static ladder."""
    source = to_gray(source01, scale)
    edits = {ph: to_gray(img, scale) for ph, img, _, _ in results}
    width = max([source.width] + [e.width for e in edits.values()])
    height = max([source.height] + [e.height for e in edits.values()])
    return source, edits, width, height


def make_sweep_gif(model, args, cached=None):
    source01, source_waviness, results = cached or edit_across_pH(model, args)
    source, edits, width, height = sweep_panels(source01, results)

    header = Image.new("RGB", (width, 26), "white")
    draw = ImageDraw.Draw(header)
    text(draw, (6, 5), f"REAL CROP   pH {args.source_pH:g}", 14, bold=True)
    text(draw, (6 + text_w(f'REAL CROP   pH {args.source_pH:g}', 14, True) + 14, 6),
         f"centreline rms {source_waviness:.2f}px   (untouched, same in every frame)",
         12, fill=MUTED)

    frames, durations = [], []
    # Ping-pong rather than a hard cut back to pH 4: the loop then reads as one filament
    # relaxing and buckling, which is the physical claim, instead of a slideshow.
    for ph, edited, info, measured in results + results[-2:0:-1]:
        caption = Image.new("RGB", (width, 26), "white")
        d = ImageDraw.Draw(caption)
        head = f"EDITED   pH {args.source_pH:g}  ->  {ph:g}"
        text(d, (6, 5), head, 14, bold=True, fill=ACCENT)
        text(d, (6 + text_w(head, 14, True) + 14, 6),
             f"asked {target_of(info):.1f}px   "
             f"got {measured:.2f}px   {info.get('mode', '?')}", 12, fill=MUTED)
        frames.append(stack([header, pad_to(source, width, source.height),
                             caption, pad_to(edits[ph], width, height),
                             ph_dial(width, ph)], width))
        durations.append(520)
    durations[len(results) - 1] = 1400   # pause at the alkaline extreme
    durations[0] = 1000                  # ...and at the acidic one
    write_gif(os.path.join(ASSETS, "ph_sweep.gif"), frames, durations)
    return source01, source_waviness, results


def make_ladder(model, args, cached=None):
    """A static strip of the same sweep, for renderers that do not animate GIFs."""
    source01, source_waviness, results = cached or edit_across_pH(model, args)
    # --ladder_pH selects FROM the sweep rather than adding to it, so a value that is not
    # in --pH would otherwise vanish from the figure without a word - which is how pH 8.8
    # went missing from the first version of this strip.
    missing = [ph for ph in args.ladder_pH if ph not in args.pH]
    if missing:
        print(f"  warning: {missing} not in --pH, so not in the ladder "
              f"(available: {args.pH})")
    shown = [r for r in results if r[0] in args.ladder_pH] or results
    source, edits, width, height = sweep_panels(source01, results, scale=1)

    def row(img, head, detail, colour=INK):
        band = Image.new("RGB", (width, 22), "white")
        d = ImageDraw.Draw(band)
        text(d, (4, 3), head, 13, bold=True, fill=colour)
        text(d, (4 + text_w(head, 13, True) + 12, 4), detail, 11, fill=MUTED)
        return [band, pad_to(img, width, img.height)]

    parts = row(source, f"pH {args.source_pH:g}  REAL",
                f"centreline rms {source_waviness:.2f}px   the input, untouched")
    for ph, _, info, measured in shown:
        band = "trained" if PH_MIN <= ph <= PH_MAX else "extrapolated"
        parts += row(edits[ph], f"pH {ph:g}",
                     f"rms {measured:.2f}px   (asked {target_of(info):.1f}px)   {band}",
                     ACCENT if band == "extrapolated" else INK)
    path = os.path.join(ASSETS, "ph_ladder.png")
    stack(parts, width, gap=4).save(path)
    print(f"  {path}  {os.path.getsize(path) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--only", choices=["denoise", "sweep", "ladder"], nargs="+",
                    help="default: all of them")
    ap.add_argument("--ref_image", default=DEMO_SOURCE)
    ap.add_argument("--source_pH", type=float, default=DEMO_SOURCE_PH)
    ap.add_argument("--pH", type=float, nargs="+",
                    default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ap.add_argument("--ladder_pH", type=float, nargs="+", default=[4, 7, 8, 12, 16],
                    help="which of --pH to show in the static ladder")
    ap.add_argument("--denoise_steps", type=int, default=400)
    ap.add_argument("--denoise_frames", type=int, default=28)
    ap.add_argument("--edit_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7, help="seed for the pH sweep edits")
    # Not an arbitrary pick. Free generation is a lottery - the model is drawing a fibre
    # from nothing - so 48 (frame size, seed) combinations were scored on whether every
    # row holds ONE dark unbroken filament and whether waviness rises with pH across the
    # rows. Seed 7 at this size, the previous choice, was neither: 78% single-fibre columns
    # (two overlapping filaments in the acidic rows) and NON-monotone, with pH 7.3 landing
    # at 5.65px against pH 5.8's 5.84px - the middle row contradicted the figure's point.
    # Seed 15 scores 94% single, 0.81 depth, 0.99 continuity, and rises 1.50/1.62/4.35px.
    ap.add_argument("--denoise_seed", type=int, default=15,
                    help="seed for the from-noise animation; see the note in the source")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found "
              f"(see checkpoints/download_trained_model.txt).")
        return
    os.makedirs(ASSETS, exist_ok=True)
    wanted = set(args.only or ["denoise", "sweep", "ladder"])

    model = from_state_dict(torch.load(args.checkpoint, map_location=DEVICE), DEVICE)
    model.eval()
    print(f"checkpoint {args.checkpoint}  in_channels={model.conv_in.weight.shape[1]}  "
          f"device {DEVICE}")

    if "denoise" in wanted:
        print("denoise.gif")
        make_denoise_gif(model, args)

    cached = None
    if "sweep" in wanted:
        print("ph_sweep.gif")
        cached = make_sweep_gif(model, args)
    if "ladder" in wanted:
        print("ph_ladder.png")
        make_ladder(model, args, cached)


if __name__ == "__main__":
    main()
