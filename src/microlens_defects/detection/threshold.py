"""Threshold-based baseline detection pipeline (28-frame fringe stack)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from microlens_defects.data.db import ImageStackLoader


@dataclass
class ThresholdParams:
    """关键可调参数（匹配原阈值法实现）。"""

    bg_kernel_ratio: float = 0.001
    bg_max_sigma: float = 60.0
    open_kernel: int = 3
    close_kernel: int = 5
    adaptive_block: int = 41
    adaptive_c: int = 9

    density_kernel_ratio: float = 0.08
    density_threshold: float = 0.2
    dense_min_area: int = 2000

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

    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    def ensure_valid(self) -> None:
        if self.open_kernel < 1:
            self.open_kernel = 1
        if self.open_kernel % 2 == 0:
            self.open_kernel += 1
        if self.close_kernel < 1:
            self.close_kernel = 1
        if self.close_kernel % 2 == 0:
            self.close_kernel += 1
        if self.adaptive_block < 3:
            self.adaptive_block = 3
        if self.adaptive_block % 2 == 0:
            self.adaptive_block += 1
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


DEFAULT_PARAMS = ThresholdParams()
DEFAULT_OUTPUT_DIR = Path.cwd() / "defect_detection_outputs"
DEFAULT_NUM_FRAMES = 28

CATEGORIES = [
    {"id": 1, "name": "scratch"},
    {"id": 2, "name": "pit"},
    {"id": 3, "name": "crash"},
    {"id": 4, "name": "anomaly"},
]
CATEGORY_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}
CATEGORY_SHORT = {"scratch": "Scr", "pit": "Pit", "crash": "Crsh", "anomaly": "Anm"}


def gaussian_with_cap(image: np.ndarray, sigma: float, sigma_cap: float) -> np.ndarray:
    """Apply Gaussian blur with downscaling when sigma is large to avoid stalls."""
    if sigma <= sigma_cap:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    scale = max(sigma_cap / float(sigma), 0.1)
    h, w = image.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred_small = cv2.GaussianBlur(down, (0, 0), sigmaX=sigma_cap, sigmaY=sigma_cap)
    return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)


def estimate_roi_mask(dc_map: np.ndarray) -> np.ndarray:
    """Estimate ROI from DC map using band gradients."""
    img = cv2.normalize(dc_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    h, w = enhanced.shape
    band = max(5, int(0.08 * min(h, w)))
    band_mask = np.zeros_like(enhanced, dtype=bool)
    band_mask[:band, :] = True
    band_mask[-band:, :] = True
    band_mask[:, :band] = True
    band_mask[:, -band:] = True
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)
    border_grad = grad_mag[band_mask]
    inner_mask = np.zeros_like(enhanced, dtype=bool)
    inner_margin = max(2, band)
    inner_mask[inner_margin : h - inner_margin, inner_margin : w - inner_margin] = True
    inner_mask &= ~band_mask
    inner_grad = grad_mag[inner_mask] if np.any(inner_mask) else np.array([0], dtype=np.float32)
    border_mean = float(border_grad.mean()) if border_grad.size else 0.0
    inner_mean = float(inner_grad.mean()) if inner_grad.size else 1.0
    has_frame = border_grad.size > 0 and border_mean > inner_mean * 1.2 and (border_mean - inner_mean) > 5.0
    if has_frame and band_mask.any():
        band_samples = enhanced[band_mask]
        sample_img = band_samples.reshape(-1, 1)
        band_thresh, _ = cv2.threshold(sample_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        band_thresh = float(np.clip(band_thresh, 5, 200))
        global_thresh = band_thresh + 5.0
        _, initial_mask = cv2.threshold(enhanced, int(global_thresh), 255, cv2.THRESH_BINARY)
    else:
        perc = np.percentile(enhanced, 60)
        _, initial_mask = cv2.threshold(enhanced, int(perc), 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num <= 1:
        roi = cleaned
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_idx = 1 + int(np.argmax(areas))
        roi = (labels == main_idx).astype(np.uint8) * 255
    roi = ndi.binary_fill_holes(roi > 0).astype(np.uint8) * 255
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi = cv2.erode(roi, erode_kernel, iterations=1)
    return (roi > 0).astype(np.uint8)


def build_feature_bundle(stack: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate DC, temporal std, and ROI mask from raw stack."""
    dc_map = np.mean(stack, axis=2)
    temporal_std_map = np.std(stack, axis=2)
    roi_mask = estimate_roi_mask(dc_map) * 255
    return {
        "dc_map": dc_map.astype(np.float32),
        "temporal_std_map": temporal_std_map.astype(np.float32),
        "roi_mask": roi_mask.astype(np.uint8),
    }


def crop_bundle(bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Crop feature maps based on ROI to reduce computation."""
    roi_mask = (bundle["roi_mask"] > 0).astype(np.uint8)
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("ROI mask is empty; cannot crop.")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    dc_crop = bundle["dc_map"][y0:y1, x0:x1]
    temporal_crop = bundle["temporal_std_map"][y0:y1, x0:x1]
    roi_crop = roi_mask[y0:y1, x0:x1] * 255
    dc_uint8 = cv2.normalize(dc_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        "dc_crop": dc_crop,
        "temporal_crop": temporal_crop,
        "roi_mask": roi_crop,
        "roi_uint8": dc_uint8,
        "bbox": (x0, y0, x1 - x0, y1 - y0),
    }


def build_temporal_masks(
    detection_img: np.ndarray,
    temporal_roi: np.ndarray,
    roi_mask: np.ndarray,
    params: ThresholdParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine CLAHE DC (optionally temporal) into a global mask."""
    mask_base = (roi_mask > 0).astype(np.uint8) * 255
    source = detection_img.astype(np.uint8)
    source_inv = cv2.bitwise_not(source)
    diff = cv2.normalize(source_inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_inv = cv2.bitwise_not(diff)
    adaptive_mask = cv2.adaptiveThreshold(
        diff_inv,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        params.adaptive_block,
        params.adaptive_c,
    )
    adaptive_mask = cv2.bitwise_not(adaptive_mask)
    adaptive_mask = cv2.bitwise_and(adaptive_mask, mask_base)
    mask = cv2.morphologyEx(
        adaptive_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.open_kernel, params.open_kernel)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close_kernel, params.close_kernel)),
    )
    num_tmp, labels_tmp, stats_tmp, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for idx in range(1, num_tmp):
        if stats_tmp[idx, cv2.CC_STAT_AREA] >= 100:
            filtered[labels_tmp == idx] = 255
    return mask_base, source, filtered


def _merge_bboxes(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0


def classify_components(
    mask_base: np.ndarray,
    global_mask: np.ndarray,
    clahe_roi: np.ndarray,
    params: ThresholdParams,
) -> Tuple[List[Dict], np.ndarray]:
    """Split global mask into crash/scratch/pit/anomaly annotations."""
    annotations: List[Dict] = []
    ann_id = 1
    clahe_roi_inv = cv2.bitwise_not(clahe_roi)
    h_img, w_img = clahe_roi_inv.shape

    def add_ann(category: str, bbox, area, polygon=None, rotated=None, prominence: float | None = None) -> None:
        nonlocal ann_id
        ann = {
            "id": ann_id,
            "category_id": CATEGORY_NAME_TO_ID[category],
            "category_name": category,
            "bbox": [float(v) for v in bbox],
            "area": float(area),
            "segmentation": [polygon] if polygon else [],
            "rotated_points": rotated,
            "iscrowd": 0,
        }
        if prominence is not None:
            ann["prominence"] = float(prominence)
        annotations.append(ann)
        ann_id += 1

    def extend_component(mask: np.ndarray, rect, long_side: float, short_side: float) -> None:
        extend_len = max(int(long_side * params.scratch_extend_ratio), params.scratch_extend_min_px)
        extend_width = max(int(short_side * params.scratch_extend_width_ratio), params.scratch_extend_min_width)
        width, height = rect[1]
        if width <= 0 or height <= 0:
            return
        if width >= height:
            new_width = max(1, int(width + 2 * extend_len))
            new_height = max(1, int(height + 2 * extend_width))
        else:
            new_width = max(1, int(width + 2 * extend_width))
            new_height = max(1, int(height + 2 * extend_len))
        extended_rect = (rect[0], (new_width, new_height), rect[2])
        box = cv2.boxPoints(extended_rect)
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)

    def sample_pixel(px: float, py: float) -> float:
        px_i = int(min(max(round(px), 0), w_img - 1))
        py_i = int(min(max(round(py), 0), h_img - 1))
        return float(clahe_roi_inv[py_i, px_i])

    def compute_prominence(bbox: Tuple[int, int, int, int]) -> float:
        x, y, w_box, h_box = bbox
        if w_box <= 0 or h_box <= 0:
            return 0.0
        x0 = x + w_box / 2.0
        y0 = y + h_box / 2.0
        center_vals = [
            sample_pixel(x0, y0),
            sample_pixel(x0 + 1, y0),
            sample_pixel(x0 - 1, y0),
            sample_pixel(x0, y0 + 1),
            sample_pixel(x0, y0 - 1),
        ]
        center_mean = sum(center_vals) / len(center_vals)
        corners = [
            sample_pixel(x, y),
            sample_pixel(x + w_box - 1, y),
            sample_pixel(x, y + h_box - 1),
            sample_pixel(x + w_box - 1, y + h_box - 1),
        ]
        corner_mean = sum(corners) / len(corners)
        return center_mean - corner_mean

    h, w = global_mask.shape
    k = max(3, int(min(h, w) * params.density_kernel_ratio))
    if k % 2 == 0:
        k += 1
    density = cv2.GaussianBlur(global_mask, (k, k), 0)
    thresh_val = params.density_threshold * 255.0
    _, dense_mask = cv2.threshold(density, thresh_val, 255, cv2.THRESH_BINARY)
    dense_mask = cv2.bitwise_and(dense_mask, mask_base)
    dense_mask = cv2.morphologyEx(
        dense_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close_kernel, params.close_kernel)),
    )
    dense_mask_filtered = np.zeros_like(dense_mask)
    if np.any(dense_mask):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(dense_mask, connectivity=8)
        for idx in range(1, num):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area < params.dense_min_area:
                continue
            component_mask = (labels == idx).astype(np.uint8) * 255
            dense_mask_filtered = cv2.bitwise_or(dense_mask_filtered, component_mask)
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w_box = int(stats[idx, cv2.CC_STAT_WIDTH])
            h_box = int(stats[idx, cv2.CC_STAT_HEIGHT])
            add_ann("crash", (x, y, w_box, h_box), area)
    dense_mask = dense_mask_filtered

    residual_mask = cv2.subtract(global_mask, dense_mask)
    candidates: List[Dict] = []
    if np.any(residual_mask):
        num_seed, labels_seed, stats_seed, _ = cv2.connectedComponentsWithStats(residual_mask, connectivity=8)
        for idx in range(1, num_seed):
            area = stats_seed[idx, cv2.CC_STAT_AREA]
            if area < params.min_area:
                continue
            component_mask = (labels_seed == idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width <= 0 or height <= 0:
                continue
            long_side, short_side = max(width, height), min(width, height)
            aspect = long_side / max(short_side, 1e-5)
            x = int(stats_seed[idx, cv2.CC_STAT_LEFT])
            y = int(stats_seed[idx, cv2.CC_STAT_TOP])
            w_box = int(stats_seed[idx, cv2.CC_STAT_WIDTH])
            h_box = int(stats_seed[idx, cv2.CC_STAT_HEIGHT])
            bbox = (x, y, w_box, h_box)
            prominence = compute_prominence(bbox)
            if long_side >= params.scratch_min_len and aspect >= params.scratch_min_aspect:
                category = "scratch"
            else:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0.0
                if perimeter > 1e-3:
                    component_area = cv2.contourArea(cnt)
                    circularity = 4.0 * np.pi * component_area / (perimeter * perimeter)
                if area <= params.pit_max_area and circularity >= params.pit_circularity:
                    category = "pit"
                else:
                    category = "anomaly"
                if prominence < params.prominence_min_value:
                    continue
            candidates.append(
                {
                    "mask": component_mask,
                    "bbox": bbox,
                    "area": area,
                    "rect": rect,
                    "category": category,
                    "prominence": prominence,
                    "removed": False,
                }
            )

    for i, cand in enumerate(candidates):
        if cand["removed"] or cand["category"] != "scratch":
            continue
        rect = cand["rect"]
        width, height = rect[1]
        if width <= 0 or height <= 0:
            continue
        long_side, short_side = max(width, height), min(width, height)
        search_mask = np.zeros_like(residual_mask)
        extend_component(search_mask, rect, long_side, short_side)
        for j, other in enumerate(candidates):
            if j == i or other["removed"] or other["category"] != "scratch":
                continue
            prominence_diff = abs(other["prominence"] - cand["prominence"])
            if prominence_diff > params.scratch_merge_gray_tol:
                continue
            overlap = cv2.countNonZero(cv2.bitwise_and(search_mask, other["mask"]))
            if overlap == 0:
                continue
            cand["mask"] = cv2.bitwise_or(cand["mask"], other["mask"])
            cand["bbox"] = _merge_bboxes(cand["bbox"], other["bbox"])
            cand["area"] = int(cv2.countNonZero(cand["mask"]))
            contours, _ = cv2.findContours(cand["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt_all = np.vstack(contours)
                cand["rect"] = cv2.minAreaRect(cnt_all)
            cand["prominence"] = compute_prominence(cand["bbox"])
            other["removed"] = True

    retained_small_mask = np.zeros_like(global_mask)
    classification_map = np.zeros_like(residual_mask)
    for cand in candidates:
        if cand["removed"]:
            continue
        mask = cand["mask"]
        retained_small_mask = cv2.bitwise_or(retained_small_mask, mask)
        prominence = cand["prominence"]
        if cand["category"] == "scratch":
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = None
            if contours:
                cnt_all = np.vstack(contours)
                rect = cv2.minAreaRect(cnt_all)
                if rect[1][0] > 0 and rect[1][1] > 0:
                    box = cv2.boxPoints(rect)
                    polygon = [float(v) for point in box for v in point]
            add_ann("scratch", cand["bbox"], cand["area"], polygon, polygon, prominence)
            classification_map[mask > 0] = 180
        elif cand["category"] == "pit":
            add_ann("pit", cand["bbox"], cand["area"], prominence=prominence)
            classification_map[mask > 0] = 120
        else:
            add_ann("anomaly", cand["bbox"], cand["area"], prominence=prominence)
            classification_map[mask > 0] = 255

    final_mask = cv2.bitwise_or(retained_small_mask, dense_mask)
    final_mask = cv2.bitwise_and(final_mask, mask_base)
    return annotations, final_mask


def render_overlay(roi: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """Render colored overlay for annotations."""
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        category = ann["category_name"]
        if category == "scratch":
            color = (0, 255, 255)
        elif category == "pit":
            color = (0, 255, 0)
        elif category == "crash":
            color = (0, 0, 255)
        else:
            color = (255, 0, 255)
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


def _record_metadata(save_root: Path, entry: Dict[str, Any]) -> None:
    csv_path = save_root / "metadata_summary.csv"
    jsonl_path = save_root / "metadata_summary.jsonl"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        import csv

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
    """Write overlay, CLAHE DC, global mask, and COCO annotations."""
    x, y, w, h = bbox
    ann_roi = result["annotations"]
    overlay_roi = render_overlay(roi_vis, ann_roi)
    overlay_full = np.zeros_like(cv2.cvtColor(dc_full_uint8, cv2.COLOR_GRAY2BGR))
    overlay_full[y : y + h, x : x + w] = overlay_roi
    overlay_path = save_dir / f"{base_name}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_full)
    cv2.imwrite(str(save_dir / f"{base_name}_dc_clahe.png"), dc_full_clahe)
    cv2.imwrite(str(save_dir / f"{base_name}_global_mask.png"), global_mask_full)
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
    (save_dir / f"{base_name}_annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")


def run_detection_for_stack(
    stack: np.ndarray,
    tag: str,
    meta: Dict[str, Any],
    params: ThresholdParams,
    save_root: Path,
) -> Dict[str, Any]:
    """Execute end-to-end detection on a single stack and persist outputs."""
    stack_out_dir = save_root / tag
    stack_out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_feature_bundle(stack)
    cropped = crop_bundle(bundle)
    clahe = cv2.createCLAHE(params.clahe_clip_limit, (params.clahe_tile_size, params.clahe_tile_size))
    dc_roi_clahe = clahe.apply(cropped["roi_uint8"])
    mask_base, _, global_mask = build_temporal_masks(
        dc_roi_clahe,
        cropped["temporal_crop"],
        cropped["roi_mask"],
        params,
    )
    annotations, final_mask = classify_components(
        mask_base,
        global_mask,
        dc_roi_clahe,
        params,
    )
    dc_full_uint8 = cv2.normalize(bundle["dc_map"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe_full = cv2.createCLAHE(params.clahe_clip_limit, (params.clahe_tile_size, params.clahe_tile_size))
    dc_full_clahe = clahe_full.apply(dc_full_uint8)
    full_mask = np.zeros_like(dc_full_uint8, dtype=np.uint8)
    x, y, w_box, h_box = cropped["bbox"]
    mask_crop = np.clip(final_mask, 0, 255).astype(np.uint8)
    full_mask[y : y + h_box, x : x + w_box] = mask_crop
    save_results(
        Path(tag),
        dc_full_uint8,
        dc_full_clahe,
        dc_roi_clahe,
        full_mask,
        cropped["bbox"],
        {"annotations": annotations, "mask": final_mask},
        stack_out_dir,
        tag,
    )
    category_counts = {name: 0 for name in CATEGORY_SHORT}
    for ann in annotations:
        category_counts[ann["category_name"]] = category_counts.get(ann["category_name"], 0) + 1
    entry = {
        "sample_tag": tag,
        "glasses_id": meta.get("glasses_id", ""),
        "lens_side": meta.get("lens_side", ""),
        "grating_type": meta.get("grating_type", ""),
        "total_defects": len(annotations),
        "count_scratch": category_counts.get("scratch", 0),
        "count_pit": category_counts.get("pit", 0),
        "count_crash": category_counts.get("crash", 0),
        "count_anomaly": category_counts.get("anomaly", 0),
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "param_snapshot": json.dumps(asdict(params), ensure_ascii=False),
    }
    _record_metadata(save_root, entry)
    return {
        "annotations": annotations,
        "mask": full_mask,
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
    """Iterate over combinations from ImageStackLoader and run detection."""
    params.ensure_valid()
    save_dir.mkdir(parents=True, exist_ok=True)
    for combo in combinations:
        glasses_id = combo["glasses_id"]
        lens_side = combo["lens_side"]
        grating = combo["grating_type"]
        tag = f"{glasses_id}_{lens_side}_{grating}"
        print(f"\n[RUN] {tag}")
        stack = loader.load_stack(glasses_id, lens_side, grating, num_frames)
        if stack is None:
            print(f"[WARN] Missing stack for {tag}; skipped.")
            continue
        print(f"    Loaded stack {stack.shape[0]}x{stack.shape[1]}x{stack.shape[2]}")
        meta = {"glasses_id": glasses_id, "lens_side": lens_side, "grating_type": grating}
        try:
            run_detection_for_stack(stack, tag, meta, params, save_dir)
        except Exception as exc:  # pragma: no cover - runtime robustness
            print(f"[ERROR] {tag}: {exc}")
