"""Mask building and component classification functions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .params import CATEGORY_NAME_TO_ID, ThresholdParams


def build_temporal_masks(
    detection_img: np.ndarray,
    temporal_roi: np.ndarray,
    roi_mask: np.ndarray,
    params: ThresholdParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine CLAHE DC (optionally temporal) into a global mask.

    Args:
        detection_img: CLAHE-enhanced DC image
        temporal_roi: Temporal standard deviation map (unused currently)
        roi_mask: Binary ROI mask
        params: Detection parameters

    Returns:
        Tuple of (mask_base, source, filtered_mask)
    """
    mask_base = (roi_mask > 0).astype(np.uint8) * 255
    source: np.ndarray = detection_img.astype(np.uint8)
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

    # Morphological cleanup
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

    # Filter small components
    num_tmp, labels_tmp, stats_tmp, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for idx in range(1, num_tmp):
        if stats_tmp[idx, cv2.CC_STAT_AREA] >= 100:
            filtered[labels_tmp == idx] = 255

    return mask_base, source, filtered


def _merge_bboxes(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Merge two bounding boxes into their union."""
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
    """Split global mask into crash/scratch/pit/anomaly annotations.

    Args:
        mask_base: Base ROI mask
        global_mask: Filtered global defect mask
        clahe_roi: CLAHE-enhanced ROI image
        params: Detection parameters

    Returns:
        Tuple of (annotations list, final mask)
    """
    annotations: List[Dict] = []
    ann_id = 1
    clahe_roi_inv = cv2.bitwise_not(clahe_roi)
    h_img, w_img = clahe_roi_inv.shape

    def add_ann(
        category: str,
        bbox: Tuple,
        area: float,
        polygon: List[float] | None = None,
        rotated: List[float] | None = None,
        prominence: float | None = None,
    ) -> None:
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

    def extend_component(mask: np.ndarray, rect: Tuple, long_side: float, short_side: float) -> None:
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

    # Density-based crash detection
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
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < params.dense_min_area:
                continue
            component_mask: np.ndarray = (labels == idx).astype(np.uint8) * 255
            dense_mask_filtered = cv2.bitwise_or(dense_mask_filtered, component_mask)
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w_box = int(stats[idx, cv2.CC_STAT_WIDTH])
            h_box = int(stats[idx, cv2.CC_STAT_HEIGHT])
            add_ann("crash", (x, y, w_box, h_box), area)
    dense_mask = dense_mask_filtered

    # Process residual components
    residual_mask = cv2.subtract(global_mask, dense_mask)
    candidates: List[Dict] = []

    if np.any(residual_mask):
        num_seed, labels_seed, stats_seed, _ = cv2.connectedComponentsWithStats(residual_mask, connectivity=8)
        for idx in range(1, num_seed):
            area = int(stats_seed[idx, cv2.CC_STAT_AREA])
            if area < params.min_area:
                continue

            component_mask: np.ndarray = (labels_seed == idx).astype(np.uint8) * 255
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

            # Classify component
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

    # Merge scratches
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

    # Build final output
    retained_small_mask = np.zeros_like(global_mask)

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
        elif cand["category"] == "pit":
            add_ann("pit", cand["bbox"], cand["area"], prominence=prominence)
        else:
            add_ann("anomaly", cand["bbox"], cand["area"], prominence=prominence)

    final_mask = cv2.bitwise_or(retained_small_mask, dense_mask)
    final_mask = cv2.bitwise_and(final_mask, mask_base)

    return annotations, final_mask
