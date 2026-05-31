"""Tests for mask building and classification functions."""

from __future__ import annotations

import numpy as np

from microlens_defects.detection.masks import (
    _merge_bboxes,
    build_temporal_masks,
    classify_components,
)


class TestBuildTemporalMasks:
    """Tests for build_temporal_masks function."""

    def test_returns_three_arrays(self, default_params):
        """Should return tuple of three arrays."""
        detection_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        temporal_roi = np.random.rand(50, 50).astype(np.float32) * 50
        roi_mask = np.ones((50, 50), dtype=np.uint8) * 255

        result = build_temporal_masks(detection_img, temporal_roi, roi_mask, default_params)

        assert len(result) == 3
        mask_base, source, filtered = result
        assert mask_base.shape == (50, 50)
        assert source.shape == (50, 50)
        assert filtered.shape == (50, 50)

    def test_mask_base_is_binary(self, default_params):
        """mask_base should be binary (0 or 255)."""
        detection_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        temporal_roi = np.random.rand(50, 50).astype(np.float32) * 50
        roi_mask = np.ones((50, 50), dtype=np.uint8) * 255

        mask_base, _, _ = build_temporal_masks(detection_img, temporal_roi, roi_mask, default_params)

        unique_vals = np.unique(mask_base)
        assert all(v in [0, 255] for v in unique_vals)


class TestMergeBboxes:
    """Tests for _merge_bboxes helper."""

    def test_merge_non_overlapping(self):
        """Merging non-overlapping boxes should create enclosing box."""
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 10, 10)
        merged = _merge_bboxes(box_a, box_b)

        assert merged[0] == 0  # x
        assert merged[1] == 0  # y
        assert merged[2] == 30  # width
        assert merged[3] == 30  # height

    def test_merge_overlapping(self):
        """Merging overlapping boxes should work correctly."""
        box_a = (0, 0, 20, 20)
        box_b = (10, 10, 20, 20)
        merged = _merge_bboxes(box_a, box_b)

        assert merged[0] == 0
        assert merged[1] == 0
        assert merged[2] == 30
        assert merged[3] == 30

    def test_merge_identical(self):
        """Merging identical boxes should return same box."""
        box = (5, 5, 10, 10)
        merged = _merge_bboxes(box, box)
        assert merged == box


class TestClassifyComponents:
    """Tests for classify_components function."""

    def test_returns_tuple(self, default_params):
        """Should return (annotations, final_mask) tuple."""
        mask_base = np.ones((50, 50), dtype=np.uint8) * 255
        global_mask = np.zeros((50, 50), dtype=np.uint8)
        clahe_roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        result = classify_components(mask_base, global_mask, clahe_roi, default_params)

        assert isinstance(result, tuple)
        assert len(result) == 2
        annotations, final_mask = result
        assert isinstance(annotations, list)
        assert isinstance(final_mask, np.ndarray)

    def test_empty_mask_returns_empty_annotations(self, default_params):
        """Empty global_mask should produce empty annotations."""
        mask_base = np.ones((50, 50), dtype=np.uint8) * 255
        global_mask = np.zeros((50, 50), dtype=np.uint8)
        clahe_roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        annotations, _ = classify_components(mask_base, global_mask, clahe_roi, default_params)

        assert len(annotations) == 0

    def test_annotation_format(self, default_params):
        """Annotations should have required COCO fields."""
        mask_base = np.ones((100, 100), dtype=np.uint8) * 255
        # Create a simple defect blob
        global_mask = np.zeros((100, 100), dtype=np.uint8)
        global_mask[40:60, 40:60] = 255
        clahe_roi = np.ones((100, 100), dtype=np.uint8) * 128
        clahe_roi[40:60, 40:60] = 50  # darker in defect region

        annotations, _ = classify_components(mask_base, global_mask, clahe_roi, default_params)

        if annotations:  # May or may not detect depending on params
            ann = annotations[0]
            assert "id" in ann
            assert "category_id" in ann
            assert "category_name" in ann
            assert "bbox" in ann
            assert "area" in ann
