"""Rendering and result export functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .params import CATEGORIES, CATEGORY_SHORT


def render_overlay(roi: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """Render colored overlay for annotations.
    
    Args:
        roi: Grayscale ROI image
        annotations: List of annotation dicts with bbox, category_name, rotated_points
        
    Returns:
        BGR overlay image
    """
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        category = ann["category_name"]
        
        # Color mapping
        if category == "scratch":
            color = (0, 255, 255)  # Yellow
        elif category == "pit":
            color = (0, 255, 0)    # Green
        elif category == "crash":
            color = (0, 0, 255)    # Red
        else:
            color = (255, 0, 255)  # Magenta
        
        rotated = ann.get("rotated_points")
        if category == "scratch" and rotated:
            pts = np.array(rotated, dtype=np.float32).reshape(-1, 2)
            cv2.polylines(overlay, [pts.astype(np.int32)], True, color, 2)
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        cv2.putText(
            overlay,
            CATEGORY_SHORT.get(category, category[:1].upper()),
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    
    return overlay


def record_metadata(save_root: Path, entry: Dict[str, Any]) -> None:
    """Append metadata entry to CSV and JSONL files.
    
    Args:
        save_root: Root directory for saving
        entry: Metadata dictionary to record
    """
    csv_path = save_root / "metadata_summary.csv"
    jsonl_path = save_root / "metadata_summary.jsonl"
    
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(entry)
    
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_results(
    image_path: Path,
    dc_full_uint8: np.ndarray,
    dc_full_clahe: np.ndarray,
    roi_vis: np.ndarray,
    global_mask_full: np.ndarray,
    bbox: Tuple[int, int, int, int],
    result: Dict,
    save_dir: Path,
    base_name: str,
) -> None:
    """Write overlay, CLAHE DC, global mask, and COCO annotations.
    
    Args:
        image_path: Original image path (for COCO file_name)
        dc_full_uint8: Full DC image as uint8
        dc_full_clahe: Full CLAHE-enhanced DC image
        roi_vis: ROI visualization image
        global_mask_full: Full global defect mask
        bbox: (x, y, width, height) of ROI
        result: Detection result dict with 'annotations' key
        save_dir: Directory to save outputs
        base_name: Base filename for outputs
    """
    x, y, w, h = bbox
    ann_roi = result["annotations"]
    
    # Create overlay
    overlay_roi = render_overlay(roi_vis, ann_roi)
    overlay_full = np.zeros_like(cv2.cvtColor(dc_full_uint8, cv2.COLOR_GRAY2BGR))
    overlay_full[y : y + h, x : x + w] = overlay_roi
    
    # Save images
    overlay_path = save_dir / f"{base_name}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_full)
    cv2.imwrite(str(save_dir / f"{base_name}_dc_clahe.png"), dc_full_clahe)
    cv2.imwrite(str(save_dir / f"{base_name}_global_mask.png"), global_mask_full)
    
    # Save COCO annotations
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": str(image_path.name),
                "width": int(dc_full_uint8.shape[1]),
                "height": int(dc_full_uint8.shape[0]),
            }
        ],
        "annotations": ann_roi,
        "categories": CATEGORIES,
    }
    (save_dir / f"{base_name}_annotations.json").write_text(
        json.dumps(coco, indent=2), encoding="utf-8"
    )
