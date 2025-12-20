"""Feature extraction functions for threshold-based detection."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def gaussian_with_cap(image: np.ndarray, sigma: float, sigma_cap: float) -> np.ndarray:
    """Apply Gaussian blur with downscaling when sigma is large to avoid stalls.
    
    Args:
        image: Input image
        sigma: Desired sigma for Gaussian blur
        sigma_cap: Maximum sigma before downscaling is applied
        
    Returns:
        Blurred image with same shape as input
    """
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
    """Estimate ROI from DC map using band gradients.
    
    Args:
        dc_map: DC (mean) image from stack
        
    Returns:
        Binary ROI mask (0/1 uint8)
    """
    img = cv2.normalize(dc_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    h, w = enhanced.shape
    band = max(5, int(0.08 * min(h, w)))
    
    # Create band mask for border region
    band_mask = np.zeros_like(enhanced, dtype=bool)
    band_mask[:band, :] = True
    band_mask[-band:, :] = True
    band_mask[:, :band] = True
    band_mask[:, -band:] = True
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)
    
    border_grad = grad_mag[band_mask]
    
    # Inner mask excluding border
    inner_mask = np.zeros_like(enhanced, dtype=bool)
    inner_margin = max(2, band)
    inner_mask[inner_margin : h - inner_margin, inner_margin : w - inner_margin] = True
    inner_mask &= ~band_mask
    inner_grad = grad_mag[inner_mask] if np.any(inner_mask) else np.array([0], dtype=np.float32)
    
    border_mean = float(border_grad.mean()) if border_grad.size else 0.0
    inner_mean = float(inner_grad.mean()) if inner_grad.size else 1.0
    
    # Detect if frame exists
    has_frame = (
        border_grad.size > 0
        and border_mean > inner_mean * 1.2
        and (border_mean - inner_mean) > 5.0
    )
    
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
    
    # Morphological cleanup
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    
    # Keep only largest component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num <= 1:
        roi = cleaned
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_idx = 1 + int(np.argmax(areas))
        roi = (labels == main_idx).astype(np.uint8) * 255
    
    # Fill holes and erode slightly
    roi = ndi.binary_fill_holes(roi > 0).astype(np.uint8) * 255
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi = cv2.erode(roi, erode_kernel, iterations=1)
    
    return (roi > 0).astype(np.uint8)


def build_feature_bundle(stack: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate DC, temporal std, and ROI mask from raw stack.
    
    Args:
        stack: Image stack with shape (H, W, N)
        
    Returns:
        Dictionary containing:
            - dc_map: Mean image (float32)
            - temporal_std_map: Temporal standard deviation (float32)
            - roi_mask: Binary ROI mask (uint8, 0/255)
    """
    dc_map = np.mean(stack, axis=2)
    temporal_std_map = np.std(stack, axis=2)
    roi_mask = estimate_roi_mask(dc_map) * 255
    
    return {
        "dc_map": dc_map.astype(np.float32),
        "temporal_std_map": temporal_std_map.astype(np.float32),
        "roi_mask": roi_mask.astype(np.uint8),
    }


def crop_bundle(bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Crop feature maps based on ROI to reduce computation.
    
    Args:
        bundle: Dictionary from build_feature_bundle
        
    Returns:
        Dictionary containing cropped features and bounding box:
            - dc_crop: Cropped DC map
            - temporal_crop: Cropped temporal std map
            - roi_mask: Cropped ROI mask
            - roi_uint8: Cropped DC as uint8 for visualization
            - bbox: (x, y, width, height) of crop region
            
    Raises:
        RuntimeError: If ROI mask is empty
    """
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
