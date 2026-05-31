"""Build the README hero animation from a typical detection result.

Composes an animated WEBP that walks through the defect-detection pipeline
stages for a single sample:

    DC + CLAHE (morphology base)  ->  global mask  ->  annotated overlay

Run before the local ``defect_detection_outputs/`` directory is cleaned up::

    python scripts/make_demo_webp.py --sample 2006_left_cycle

Images are 8-bit PNGs (grayscale ``L`` or ``RGB``) at 2560x2160; they are
downscaled to ``--width`` px and saved as a looping WEBP plus a couple of
static stills under ``docs/assets/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

STAGES: Tuple[Tuple[str, str], ...] = (
    ("dc_clahe", "1 - DC + CLAHE (microstructure morphology)"),
    ("global_mask", "2 - Global threshold mask"),
    ("overlay", "3 - Classified defects (scratch / pit / crash)"),
)


def _load_rgb(path: Path, width: int) -> Image.Image:
    """Load a PNG, convert to RGB, and resize to the target width (keep aspect)."""
    img = Image.open(path).convert("RGB")
    height = round(img.height * width / img.width)
    return img.resize((width, height), Image.LANCZOS)


def _label(img: Image.Image, text: str) -> Image.Image:
    """Draw a caption banner across the bottom of a copy of ``img``."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, out.width // 36))
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    pad = 8
    bar_h = (box[3] - box[1]) + 3 * pad
    draw.rectangle([0, out.height - bar_h, out.width, out.height], fill=(0, 0, 0))
    draw.text((pad, out.height - bar_h + pad), text, fill=(255, 255, 255), font=font)
    return out


def build_demo(sample_dir: Path, out_dir: Path, width: int, frame_ms: int) -> Path:
    """Assemble the animated WEBP and stills; return the WEBP path."""
    tag = sample_dir.name
    frames: List[Image.Image] = []
    overlay_img: Image.Image | None = None
    for suffix, caption in STAGES:
        src = sample_dir / f"{tag}_{suffix}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing stage image: {src}")
        rgb = _load_rgb(src, width)
        if suffix == "overlay":
            overlay_img = rgb
        frames.append(_label(rgb, caption))

    out_dir.mkdir(parents=True, exist_ok=True)
    webp_path = out_dir / "pipeline_demo.webp"
    frames[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        quality=80,
        method=6,
    )
    if overlay_img is not None:
        overlay_img.save(out_dir / "example_overlay.webp", format="WEBP", quality=82)
    return webp_path


def _find_sample(root: Path, preferred: str) -> Path:
    """Return the preferred sample dir, or the first dir holding all stages."""
    candidate = root / preferred
    if candidate.is_dir():
        return candidate
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((d / f"{d.name}_{s}.png").exists() for s, _ in STAGES):
            return d
    raise FileNotFoundError(f"No complete sample found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("defect_detection_outputs"))
    parser.add_argument("--sample", type=str, default="2006_left_cycle")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--frame-ms", type=int, default=1200)
    args = parser.parse_args()

    sample_dir = _find_sample(args.results_root, args.sample)
    webp_path = build_demo(sample_dir, args.out_dir, args.width, args.frame_ms)
    size_kb = webp_path.stat().st_size / 1024
    print(f"Built {webp_path} from {sample_dir.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
