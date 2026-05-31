"""Threshold detection parameters and constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ThresholdParams:
    """关键可调参数（匹配原阈值法实现）。"""

    # Background/blur parameters
    bg_kernel_ratio: float = 0.001
    bg_max_sigma: float = 60.0

    # Morphological parameters
    open_kernel: int = 3
    close_kernel: int = 5

    # Adaptive threshold parameters
    adaptive_block: int = 41
    adaptive_c: int = 9

    # Density detection (crash) parameters
    density_kernel_ratio: float = 0.08
    density_threshold: float = 0.2
    dense_min_area: int = 2000

    # Component classification parameters
    min_area: int = 20
    scratch_min_len: int = 40
    scratch_min_aspect: float = 4.0
    scratch_extend_ratio: float = 0.25
    scratch_extend_min_px: int = 12
    scratch_extend_width_ratio: float = 1.0
    scratch_extend_min_width: int = 4
    pit_max_area: int = 2600
    pit_circularity: float = 0.65
    scratch_merge_gray_tol: float = 20.0
    prominence_min_value: float = 20.0

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    def ensure_valid(self) -> None:
        """Validate and fix parameters to ensure they are within valid ranges."""
        # Kernel sizes must be positive odd numbers
        if self.open_kernel < 1:
            self.open_kernel = 1
        if self.open_kernel % 2 == 0:
            self.open_kernel += 1

        if self.close_kernel < 1:
            self.close_kernel = 1
        if self.close_kernel % 2 == 0:
            self.close_kernel += 1

        # Adaptive block must be odd and >= 3
        if self.adaptive_block < 3:
            self.adaptive_block = 3
        if self.adaptive_block % 2 == 0:
            self.adaptive_block += 1

        # Positive constraints
        if self.bg_max_sigma <= 0:
            self.bg_max_sigma = 60.0
        if self.scratch_extend_min_px < 1:
            self.scratch_extend_min_px = 1
        if self.scratch_extend_min_width < 1:
            self.scratch_extend_min_width = 1
        if self.scratch_extend_ratio <= 0:
            self.scratch_extend_ratio = 0.1
        if self.scratch_extend_width_ratio <= 0:
            self.scratch_extend_width_ratio = 1.0
        if self.scratch_merge_gray_tol <= 0:
            self.scratch_merge_gray_tol = 0.05
        if self.prominence_min_value <= 0:
            self.prominence_min_value = 1.0


# Default parameter instance
DEFAULT_PARAMS = ThresholdParams()

# Default paths and settings
DEFAULT_OUTPUT_DIR = Path.cwd() / "defect_detection_outputs"
DEFAULT_NUM_FRAMES = 28

# COCO category definitions
CATEGORIES: List[Dict] = [
    {"id": 1, "name": "scratch"},
    {"id": 2, "name": "pit"},
    {"id": 3, "name": "crash"},
    {"id": 4, "name": "anomaly"},
]

CATEGORY_NAME_TO_ID: Dict[str, int] = {c["name"]: c["id"] for c in CATEGORIES}
CATEGORY_SHORT: Dict[str, str] = {
    "scratch": "Scr",
    "pit": "Pit",
    "crash": "Crsh",
    "anomaly": "Anm",
}
