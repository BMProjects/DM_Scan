"""Threshold-based baseline detection pipeline (28-frame fringe stack).

This module provides the ThresholdDetector class implementing the BaseDetector
interface, as well as convenience functions for batch processing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from microlens_defects.data.db import ImageStackLoader
from microlens_defects.logging import get_logger

from .base import BaseDetector, DetectionResult
from .features import build_feature_bundle, crop_bundle
from .masks import build_temporal_masks, classify_components
from .params import (
    CATEGORIES,
    CATEGORY_NAME_TO_ID,
    CATEGORY_SHORT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    ThresholdParams,
)
from .rendering import record_metadata, render_overlay, save_results

# Re-export for backward compatibility
__all__ = [
    "ThresholdDetector",
    "ThresholdParams",
    "DetectionResult",
    "DEFAULT_PARAMS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_NUM_FRAMES",
    "CATEGORIES",
    "CATEGORY_NAME_TO_ID",
    "CATEGORY_SHORT",
    "run_detection_for_stack",
    "run_threshold_detection",
]

logger = get_logger(__name__)


class ThresholdDetector(BaseDetector):
    """Threshold-based defect detector using classical image processing.
    
    This detector uses CLAHE enhancement, adaptive thresholding, and 
    morphological operations to detect defects in fringe image stacks.
    
    Attributes:
        params: ThresholdParams instance controlling detection behavior
    """

    def __init__(self, params: ThresholdParams | None = None) -> None:
        """Initialize the detector.
        
        Args:
            params: Detection parameters, defaults to DEFAULT_PARAMS
        """
        self.params = params if params is not None else ThresholdParams()
        self.params.ensure_valid()

    @property
    def name(self) -> str:
        """Detector name."""
        return "ThresholdDetector"

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters as dictionary."""
        return asdict(self.params)

    def detect(self, stack: np.ndarray) -> DetectionResult:
        """Run detection on an image stack.
        
        Args:
            stack: Image stack with shape (H, W, N) where N >= 1
            
        Returns:
            DetectionResult containing mask, annotations, and metadata
        """
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
        
        # Build features
        bundle = build_feature_bundle(stack)
        cropped = crop_bundle(bundle)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            self.params.clahe_clip_limit,
            (self.params.clahe_tile_size, self.params.clahe_tile_size),
        )
        dc_roi_clahe = clahe.apply(cropped["roi_uint8"])
        
        # Build masks
        mask_base, _, global_mask = build_temporal_masks(
            dc_roi_clahe,
            cropped["temporal_crop"],
            cropped["roi_mask"],
            self.params,
        )
        
        # Classify components
        annotations, final_mask = classify_components(
            mask_base,
            global_mask,
            dc_roi_clahe,
            self.params,
        )
        
        # Reconstruct full-size mask
        dc_full_uint8 = cv2.normalize(
            bundle["dc_map"], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        full_mask = np.zeros_like(dc_full_uint8, dtype=np.uint8)
        x, y, w_box, h_box = cropped["bbox"]
        mask_crop = np.clip(final_mask, 0, 255).astype(np.uint8)
        full_mask[y : y + h_box, x : x + w_box] = mask_crop
        
        # Build metadata
        category_counts = {name: 0 for name in CATEGORY_SHORT}
        for ann in annotations:
            cat_name = ann["category_name"]
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        metadata = {
            "detector": self.name,
            "total_defects": len(annotations),
            "category_counts": category_counts,
            "bbox": cropped["bbox"],
            "param_snapshot": self.get_params(),
        }
        
        return DetectionResult(
            mask=full_mask,
            annotations=annotations,
            metadata=metadata,
        )


def run_detection_for_stack(
    stack: np.ndarray,
    tag: str,
    meta: Dict[str, Any],
    params: ThresholdParams,
    save_root: Path,
) -> Dict[str, Any]:
    """Execute end-to-end detection on a single stack and persist outputs.
    
    Args:
        stack: Image stack (H, W, N)
        tag: Sample identifier string
        meta: Metadata dict with glasses_id, lens_side, grating_type
        params: Detection parameters
        save_root: Root directory for saving outputs
        
    Returns:
        Dict with annotations, mask, and metadata
    """
    stack_out_dir = save_root / tag
    stack_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run detection
    detector = ThresholdDetector(params)
    result = detector.detect(stack)
    
    # Prepare full images for saving
    bundle = build_feature_bundle(stack)
    cropped = crop_bundle(bundle)
    
    dc_full_uint8 = cv2.normalize(
        bundle["dc_map"], None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    clahe = cv2.createCLAHE(
        params.clahe_clip_limit,
        (params.clahe_tile_size, params.clahe_tile_size),
    )
    dc_full_clahe = clahe.apply(dc_full_uint8)
    dc_roi_clahe = clahe.apply(cropped["roi_uint8"])
    
    # Save results
    save_results(
        Path(tag),
        dc_full_uint8,
        dc_full_clahe,
        dc_roi_clahe,
        result.mask,
        cropped["bbox"],
        {"annotations": result.annotations, "mask": result.mask},
        stack_out_dir,
        tag,
    )
    
    # Record metadata
    entry = {
        "sample_tag": tag,
        "glasses_id": meta.get("glasses_id", ""),
        "lens_side": meta.get("lens_side", ""),
        "grating_type": meta.get("grating_type", ""),
        "total_defects": len(result.annotations),
        "count_scratch": result.metadata["category_counts"].get("scratch", 0),
        "count_pit": result.metadata["category_counts"].get("pit", 0),
        "count_crash": result.metadata["category_counts"].get("crash", 0),
        "count_anomaly": result.metadata["category_counts"].get("anomaly", 0),
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "param_snapshot": json.dumps(asdict(params), ensure_ascii=False),
    }
    record_metadata(save_root, entry)
    
    return {
        "annotations": result.annotations,
        "mask": result.mask,
        "metadata": entry,
    }


def run_threshold_detection(
    loader: ImageStackLoader,
    combinations: List[Dict[str, Any]],
    params: ThresholdParams,
    *,
    save_dir: Path = DEFAULT_OUTPUT_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> None:
    """Iterate over combinations from ImageStackLoader and run detection.
    
    Args:
        loader: ImageStackLoader instance
        combinations: List of dicts with glasses_id, lens_side, grating_type
        params: Detection parameters
        save_dir: Output directory
        num_frames: Maximum frames to use from each stack
    """
    params.ensure_valid()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for combo in combinations:
        glasses_id = combo["glasses_id"]
        lens_side = combo["lens_side"]
        grating = combo["grating_type"]
        tag = f"{glasses_id}_{lens_side}_{grating}"
        
        logger.info("Processing: %s", tag)
        
        stack = loader.load_stack(glasses_id, lens_side, grating, num_frames)
        if stack is None:
            logger.warning("Missing stack for %s; skipped.", tag)
            continue
        
        logger.info("Loaded stack %dx%dx%d", stack.shape[0], stack.shape[1], stack.shape[2])
        
        meta = {
            "glasses_id": glasses_id,
            "lens_side": lens_side,
            "grating_type": grating,
        }
        
        try:
            run_detection_for_stack(stack, tag, meta, params, save_dir)
        except Exception as exc:  # pragma: no cover - runtime robustness
            logger.error("Error processing %s: %s", tag, exc)
