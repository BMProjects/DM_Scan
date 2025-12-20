"""Tests for feature extraction functions."""

from __future__ import annotations

import numpy as np
import pytest

from microlens_defects.detection.features import (
    build_feature_bundle,
    crop_bundle,
    estimate_roi_mask,
    gaussian_with_cap,
)


class TestGaussianWithCap:
    """Tests for gaussian_with_cap function."""

    def test_output_shape_matches_input(self, sample_dc_map):
        """Output should have same shape as input."""
        result = gaussian_with_cap(sample_dc_map, sigma=5.0, sigma_cap=10.0)
        assert result.shape == sample_dc_map.shape

    def test_small_sigma_no_downscale(self, sample_dc_map):
        """Small sigma should not trigger downscaling."""
        result = gaussian_with_cap(sample_dc_map, sigma=5.0, sigma_cap=60.0)
        assert result.shape == sample_dc_map.shape
        # Result should be blurred (different from input)
        assert not np.array_equal(result, sample_dc_map)

    def test_large_sigma_with_downscale(self, sample_dc_map):
        """Large sigma should trigger downscaling but still work."""
        result = gaussian_with_cap(sample_dc_map, sigma=100.0, sigma_cap=60.0)
        assert result.shape == sample_dc_map.shape


class TestEstimateRoiMask:
    """Tests for estimate_roi_mask function."""

    def test_output_is_binary(self, sample_dc_map):
        """Output should be binary (0 or 1)."""
        mask = estimate_roi_mask(sample_dc_map)
        unique_vals = np.unique(mask)
        assert all(v in [0, 1] for v in unique_vals)

    def test_output_shape_matches_input(self, sample_dc_map):
        """Output shape should match input."""
        mask = estimate_roi_mask(sample_dc_map)
        assert mask.shape == sample_dc_map.shape

    def test_handles_uniform_image(self):
        """Should handle uniform images without crashing."""
        uniform = np.ones((100, 100), dtype=np.float32) * 128
        mask = estimate_roi_mask(uniform)
        assert mask.shape == uniform.shape


class TestBuildFeatureBundle:
    """Tests for build_feature_bundle function."""

    def test_returns_expected_keys(self, sample_stack):
        """Bundle should contain expected keys."""
        bundle = build_feature_bundle(sample_stack)
        assert "dc_map" in bundle
        assert "temporal_std_map" in bundle
        assert "roi_mask" in bundle

    def test_dc_map_shape(self, sample_stack):
        """DC map should have HxW shape."""
        bundle = build_feature_bundle(sample_stack)
        assert bundle["dc_map"].shape == sample_stack.shape[:2]

    def test_dc_map_is_mean(self, small_stack):
        """DC map should be temporal mean."""
        bundle = build_feature_bundle(small_stack)
        expected_mean = np.mean(small_stack, axis=2)
        np.testing.assert_array_almost_equal(bundle["dc_map"], expected_mean)

    def test_temporal_std_shape(self, sample_stack):
        """Temporal std map should have HxW shape."""
        bundle = build_feature_bundle(sample_stack)
        assert bundle["temporal_std_map"].shape == sample_stack.shape[:2]


class TestCropBundle:
    """Tests for crop_bundle function."""

    def test_returns_expected_keys(self, sample_stack):
        """Cropped bundle should contain expected keys."""
        bundle = build_feature_bundle(sample_stack)
        cropped = crop_bundle(bundle)
        assert "dc_crop" in cropped
        assert "temporal_crop" in cropped
        assert "roi_mask" in cropped
        assert "roi_uint8" in cropped
        assert "bbox" in cropped

    def test_bbox_format(self, sample_stack):
        """Bbox should be (x, y, width, height)."""
        bundle = build_feature_bundle(sample_stack)
        cropped = crop_bundle(bundle)
        bbox = cropped["bbox"]
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w > 0 and h > 0

    def test_raises_on_empty_roi(self):
        """Should raise RuntimeError if ROI is empty."""
        bundle = {
            "dc_map": np.zeros((100, 100), dtype=np.float32),
            "temporal_std_map": np.zeros((100, 100), dtype=np.float32),
            "roi_mask": np.zeros((100, 100), dtype=np.uint8),
        }
        with pytest.raises(RuntimeError, match="ROI mask is empty"):
            crop_bundle(bundle)
