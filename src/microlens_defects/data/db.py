"""SQLite-backed image stack loader used by detection pipelines."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class ImageStackLoader:
    """Load fringe stacks from SQLite metadata plus an image root directory."""

    def __init__(self, database: Path, image_root: Path, *, validate_paths: bool = True):
        self.database = Path(database)
        self.image_root = Path(image_root)
        if validate_paths:
            if not self.database.exists():
                raise FileNotFoundError(f"Database not found: {self.database}")
            if not self.image_root.exists():
                raise FileNotFoundError(f"Image root not found: {self.image_root}")

    def _resolve_path(self, relative_path: str) -> Optional[Path]:
        candidate = self.image_root / relative_path
        if candidate.exists():
            return candidate
        abs_path = Path(relative_path)
        if abs_path.exists():
            return abs_path
        return None

    def list_combinations(self, min_frames: Optional[int] = None) -> List[Dict]:
        """Return all glasses/side/grating tuples that meet the frame count requirement."""
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT glasses_id, lens_side, grating_type, COUNT(*) AS frame_count
            FROM images
            GROUP BY glasses_id, lens_side, grating_type
            ORDER BY glasses_id, lens_side, grating_type
            """
        )
        combos: List[Dict] = []
        for row in cur.fetchall():
            count = row["frame_count"]
            if min_frames is not None and count < min_frames:
                continue
            combos.append(
                {
                    "glasses_id": row["glasses_id"],
                    "lens_side": row["lens_side"],
                    "grating_type": row["grating_type"],
                    "frame_count": count,
                }
            )
        conn.close()
        return combos

    def load_stack(self, glasses_id: str, lens_side: str, grating_type: str, max_frames: int) -> Optional[np.ndarray]:
        """Query image paths for a combination and load as HxWxN float32 stack."""
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT file_path, phase_index
            FROM images
            WHERE glasses_id = ? AND lens_side = ? AND grating_type = ?
            ORDER BY phase_index ASC
            """,
            (glasses_id, lens_side, grating_type),
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None

        paths: List[Path] = []
        for row in rows:
            resolved = self._resolve_path(row["file_path"] or "")
            if resolved is None:
                print(f"[WARN] Missing image file for {glasses_id}/{lens_side}/{grating_type}: {row['file_path']}")
                continue
            paths.append(resolved)
        if not paths:
            return None

        imgs = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Unable to read image: {path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img.astype(np.float32))

        stack = np.stack(imgs, axis=2)
        if max_frames > 0 and stack.shape[2] > max_frames:
            stack = stack[:, :, :max_frames]
        return stack
