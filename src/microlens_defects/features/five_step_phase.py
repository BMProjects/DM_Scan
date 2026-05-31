"""Five-step phase-shifting method for microlens surface measurement.

Implements 5-step phase-shifting interferometry with equal phase increments of
72 degrees (one full period sampled by five frames). Given intensity frames
``I_k`` acquired at shifts ``δ_k = 2π·k/5`` (k = 0..4), the wrapped phase φ is
recovered by the standard discrete-Fourier demodulation:

    num = -Σ_k I_k · sin(δ_k)
    den =  Σ_k I_k · cos(δ_k)
    φ   = atan2(num, den)

The DC term (bias) and modulation amplitude are:

    dc  = (1/5) · Σ_k I_k
    amp = (2/5) · sqrt(num² + den²)

Note: the Hariharan five-step formula assumes 90-degree increments and is *not*
used here, since acquisition uses equal 72-degree steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class PhaseResult:
    phase: np.ndarray  # wrapped phase in [-pi, pi]
    dc: np.ndarray  # average intensity
    amplitude: np.ndarray  # modulation amplitude
    mask: np.ndarray  # 0/255 mask where amplitude exceeds threshold


def compute_phase(frames: np.ndarray, *, amp_threshold: float = 1.0) -> PhaseResult:
    """Compute wrapped phase, DC, and amplitude from a 5-frame stack (H x W x 5).

    Frames are assumed to be acquired at equal phase shifts of 72 degrees
    (δ_k = 2π·k/5, k = 0..4). Demodulation uses the discrete-Fourier sums over
    these shifts rather than the 90-degree Hariharan formula.
    """
    if frames.ndim != 3 or frames.shape[2] != 5:
        raise ValueError("Expected frames shape HxWx5 for five-step PSI.")
    stack = frames.astype(np.float32)
    shifts = 2.0 * np.pi * np.arange(5, dtype=np.float32) / 5.0
    cos_c = np.cos(shifts)
    sin_c = np.sin(shifts)
    num = -np.tensordot(stack, sin_c, axes=([2], [0]))  # -Σ I_k sin δ_k
    den = np.tensordot(stack, cos_c, axes=([2], [0]))  # Σ I_k cos δ_k
    phase = np.arctan2(num, den)  # wrapped phase in [-pi, pi]
    dc = stack.mean(axis=2)
    amplitude = (2.0 / 5.0) * np.sqrt(num * num + den * den)
    mask = (amplitude >= amp_threshold).astype(np.uint8) * 255
    return PhaseResult(
        phase=phase.astype(np.float32),
        dc=dc.astype(np.float32),
        amplitude=amplitude.astype(np.float32),
        mask=mask,
    )


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
