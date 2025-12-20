"""Tests for detector interface and ThresholdDetector."""

from __future__ import annotations

import numpy as np
import pytest

from microlens_defects.detection import (
    BaseDetector,
    DetectionResult,
    ThresholdDetector,
    ThresholdParams,
)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation_with_mask(self):
        """Should create with just a mask."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = DetectionResult(mask=mask)
        
        assert result.mask is mask
        assert result.annotations == []
        assert result.metadata == {}

    def test_creation_full(self):
        """Should create with all fields."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        annotations = [{"id": 1, "category_id": 1}]
        metadata = {"detector": "test"}
        
        result = DetectionResult(mask=mask, annotations=annotations, metadata=metadata)
        
        assert result.mask is mask
        assert result.annotations == annotations
        assert result.metadata == metadata

    def test_requires_mask(self):
        """Should raise if mask is None."""
        with pytest.raises(ValueError, match="mask cannot be None"):
            DetectionResult(mask=None)


class TestThresholdDetector:
    """Tests for ThresholdDetector class."""

    def test_implements_base_detector(self):
        """ThresholdDetector should be subclass of BaseDetector."""
        assert issubclass(ThresholdDetector, BaseDetector)

    def test_default_initialization(self):
        """Should initialize with default params."""
        detector = ThresholdDetector()
        assert isinstance(detector.params, ThresholdParams)

    def test_custom_params(self):
        """Should accept custom params."""
        params = ThresholdParams(open_kernel=5, close_kernel=7)
        detector = ThresholdDetector(params)
        assert detector.params.open_kernel == 5
        assert detector.params.close_kernel == 7

    def test_name_property(self):
        """Should have correct name."""
        detector = ThresholdDetector()
        assert detector.name == "ThresholdDetector"

    def test_get_params(self):
        """get_params should return dict."""
        detector = ThresholdDetector()
        params = detector.get_params()
        
        assert isinstance(params, dict)
        assert "open_kernel" in params
        assert "clahe_clip_limit" in params

    def test_detect_returns_detection_result(self, sample_stack):
        """detect should return DetectionResult."""
        detector = ThresholdDetector()
        result = detector.detect(sample_stack)
        
        assert isinstance(result, DetectionResult)
        assert isinstance(result.mask, np.ndarray)
        assert isinstance(result.annotations, list)
        assert isinstance(result.metadata, dict)

    def test_detect_mask_shape(self, sample_stack):
        """Output mask should have same HxW as input."""
        detector = ThresholdDetector()
        result = detector.detect(sample_stack)
        
        assert result.mask.shape == sample_stack.shape[:2]

    def test_detect_metadata_has_expected_keys(self, sample_stack):
        """Metadata should contain expected keys."""
        detector = ThresholdDetector()
        result = detector.detect(sample_stack)
        
        assert "detector" in result.metadata
        assert "total_defects" in result.metadata
        assert "category_counts" in result.metadata

    def test_detect_raises_on_wrong_dimensions(self):
        """Should raise ValueError for non-3D input."""
        detector = ThresholdDetector()
        
        with pytest.raises(ValueError, match="Expected 3D stack"):
            detector.detect(np.zeros((50, 50), dtype=np.float32))

    def test_params_are_validated(self):
        """Params should be validated on init."""
        params = ThresholdParams(open_kernel=2)  # Even, invalid
        detector = ThresholdDetector(params)
        # Should be fixed to odd
        assert detector.params.open_kernel % 2 == 1


class TestBackwardCompatibility:
    """Tests for backward compatibility of imports."""

    def test_import_from_threshold_module(self):
        """Old import paths should still work."""
        from microlens_defects.detection.threshold import (
            DEFAULT_NUM_FRAMES,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_PARAMS,
            ThresholdParams,
            run_threshold_detection,
        )
        
        assert ThresholdParams is not None
        assert DEFAULT_PARAMS is not None

    def test_import_from_detection_package(self):
        """New import paths should work."""
        from microlens_defects.detection import (
            BaseDetector,
            DetectionResult,
            ThresholdDetector,
            ThresholdParams,
        )
        
        assert BaseDetector is not None
        assert DetectionResult is not None
        assert ThresholdDetector is not None
        assert ThresholdParams is not None
