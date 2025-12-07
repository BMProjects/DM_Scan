"""Five-step phase-shifting method for microlens surface measurement.

Implements the classic 5-step PSI with equal phase increments (72 degrees).
Given five intensity frames I1..I5 with phase shifts of 0, 72, 144, 216, 288 degrees,
the wrapped phase φ is estimated by:

    num = 2 * (I2 - I4)
    den = I1 - 2 * I3 + I5
    φ = atan2(num, den)

The DC term (bias) and modulation amplitude are:

    dc = (I1 + I2 + I3 + I4 + I5) / 5
    amp = (2/5) * sqrt(num^2 + den^2)

Reference: Hariharan et al., Five-step phase-shifting algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class PhaseResult:
    phase: np.ndarray  # wrapped phase in [-pi, pi]
    dc: np.ndarray  # average intensity
    amplitude: np.ndarray  # modulation amplitude
    mask: np.ndarray  # 0/255 mask where amplitude exceeds threshold


def compute_phase(frames: np.ndarray, *, amp_threshold: float = 1.0) -> PhaseResult:
    """Compute wrapped phase, DC, and amplitude from a 5-frame stack (H x W x 5)."""
    if frames.ndim != 3 or frames.shape[2] != 5:
        raise ValueError("Expected frames shape HxWx5 for five-step PSI.")
    f1, f2, f3, f4, f5 = [frames[:, :, i].astype(np.float32) for i in range(5)]
    num = 2.0 * (f2 - f4)
    den = f1 - 2.0 * f3 + f5
    phase = np.arctan2(num, den)  # wrapped phase in [-pi, pi]
    dc = (f1 + f2 + f3 + f4 + f5) / 5.0
    amplitude = (2.0 / 5.0) * np.sqrt(num * num + den * den)
    mask = (amplitude >= amp_threshold).astype(np.uint8) * 255
    return PhaseResult(phase=phase.astype(np.float32), dc=dc.astype(np.float32), amplitude=amplitude.astype(np.float32), mask=mask)


def load_five_images(folder: Path, pattern: str = "*.tif") -> np.ndarray:
    """Load five grayscale frames from a folder using filename order."""
    files: List[Path] = sorted(folder.glob(pattern))
    if len(files) < 5:
        raise FileNotFoundError(f"Need at least 5 images in {folder} matching {pattern}, found {len(files)}")
    files = files[:5]
    imgs = []
    for fp in files:
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read {fp}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img.astype(np.float32))
    return np.stack(imgs, axis=2)


def save_phase_result(out_path: Path, result: PhaseResult) -> None:
    """Persist phase/DC/amplitude/mask into a compressed NPZ."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        phase=result.phase,
        dc=result.dc,
        amplitude=result.amplitude,
        mask=result.mask,
    )
