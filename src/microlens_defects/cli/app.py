"""Unified CLI entry point."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import typer
import yaml

from microlens_defects.data.db import ImageStackLoader
from microlens_defects.detection.threshold import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    ThresholdParams,
    run_threshold_detection,
)
from microlens_defects.features.five_step_phase import compute_phase, load_five_images, save_phase_result
from microlens_defects.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DB_FILE = Path("microlens_metadata.db")
DEFAULT_IMAGE_ROOT = Path("organized_tiffs")

app = typer.Typer(add_completion=False, help="Microlens defect detection toolkit CLI.")


def load_params(config: Optional[Path]) -> ThresholdParams:
    params = replace(DEFAULT_PARAMS)
    if config:
        cfg_data = yaml.safe_load(config.read_text(encoding="utf-8"))
        if cfg_data:
            for k, v in cfg_data.items():
                if hasattr(params, k):
                    setattr(params, k, v)
    params.ensure_valid()
    return params


@app.command()
def detect(
    db: Path = typer.Option(DEFAULT_DB_FILE, "--db", help="SQLite database path."),
    img_root: Path = typer.Option(DEFAULT_IMAGE_ROOT, "--img-root", help="Image root directory."),
    glasses: Optional[str] = typer.Option(None, "--glasses", help="Glasses ID (e.g., 2006)."),
    side: Optional[str] = typer.Option(None, "--side", help="Lens side (left/right)."),
    grating: Optional[str] = typer.Option(None, "--grating", help="Grating type (heng/zong/cycle)."),
    all: bool = typer.Option(False, "--all", help="Process all combinations in DB."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max combinations when using --all."),
    num_frames: int = typer.Option(DEFAULT_NUM_FRAMES, "--num-frames", help="Max frames per stack."),
    save_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--save-dir", help="Output root directory."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config for threshold params."),
) -> None:
    """Run classical threshold detection and export masks/COCO/metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    params = load_params(config)
    logger.info("当前参数：\n%s", json.dumps(asdict(params), ensure_ascii=False, indent=2))

    if limit and limit > 0 and not all and not any([glasses, side, grating]):
        logger.info("--limit specified without single-sample args, enabling --all.")
        all = True
    if all and any([glasses, side, grating]):
        typer.echo("--all 不可与 --glasses/--side/--grating 同时使用。", err=True)
        raise typer.Exit(code=2)
    if not all and not all([glasses, side, grating]):
        missing = [name for name, val in (("glasses", glasses), ("side", side), ("grating", grating)) if val is None]
        msg = f"缺少参数: {', '.join('--' + m for m in missing)}。如需批量，请使用 --all。"
        typer.echo(msg, err=True)
        raise typer.Exit(code=2)

    loader = ImageStackLoader(db, img_root)
    if all:
        combinations = loader.list_combinations(min_frames=num_frames)
        if limit and limit > 0:
            combinations = combinations[:limit]
    else:
        combinations = [{"glasses_id": glasses, "lens_side": side, "grating_type": grating}]

    if not combinations:
        typer.echo("数据库中没有满足帧数要求的组合。")
        raise typer.Exit(code=0)

    run_threshold_detection(loader, combinations, params, save_dir=save_dir, num_frames=num_frames)


@app.command("phase5")
def phase5(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory containing 5 phase-shifted images."),
    pattern: str = typer.Option("*.tif", "--pattern", "-p", help="Glob pattern for input images."),
    output: Path = typer.Option(Path("phase_result.npz"), "--output", "-o", help="Output NPZ path."),
    amp_threshold: float = typer.Option(1.0, "--amp-threshold", help="Amplitude threshold for valid mask."),
) -> None:
    """Compute wrapped phase/DC/amplitude using the 5-step phase-shifting method."""
    frames = load_five_images(input_dir, pattern)
    result = compute_phase(frames, amp_threshold=amp_threshold)
    save_phase_result(output, result)
    typer.echo(f"Saved phase result to {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
