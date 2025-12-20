"""Tests for detection parameters."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from microlens_defects.detection.params import (
    CATEGORIES,
    CATEGORY_NAME_TO_ID,
    DEFAULT_PARAMS,
    ThresholdParams,
)


class TestThresholdParams:
    """Tests for ThresholdParams dataclass."""

    def test_default_values(self):
        """Default params should have expected values."""
        params = ThresholdParams()
        assert params.open_kernel == 3
        assert params.close_kernel == 5
        assert params.adaptive_block == 41

    def test_ensure_valid_fixes_even_kernels(self):
        """ensure_valid should fix even kernel sizes to odd."""
        params = ThresholdParams(open_kernel=2, close_kernel=4, adaptive_block=10)
        params.ensure_valid()
        assert params.open_kernel % 2 == 1
        assert params.close_kernel % 2 == 1
        assert params.adaptive_block % 2 == 1

    def test_ensure_valid_fixes_negative_values(self):
        """ensure_valid should fix negative values."""
        params = ThresholdParams(bg_max_sigma=-1, prominence_min_value=0)
        params.ensure_valid()
        assert params.bg_max_sigma > 0
        assert params.prominence_min_value > 0

    def test_ensure_valid_fixes_small_block(self):
        """ensure_valid should ensure adaptive_block >= 3."""
        params = ThresholdParams(adaptive_block=1)
        params.ensure_valid()
        assert params.adaptive_block >= 3

    def test_serialization(self):
        """Params should serialize to dict correctly."""
        params = ThresholdParams()
        d = asdict(params)
        assert isinstance(d, dict)
        assert "open_kernel" in d
        assert d["clahe_clip_limit"] == 2.0


class TestCategories:
    """Tests for category constants."""

    def test_categories_structure(self):
        """CATEGORIES should have expected structure."""
        assert len(CATEGORIES) == 4
        for cat in CATEGORIES:
            assert "id" in cat
            assert "name" in cat

    def test_category_name_to_id(self):
        """CATEGORY_NAME_TO_ID should map correctly."""
        assert CATEGORY_NAME_TO_ID["scratch"] == 1
        assert CATEGORY_NAME_TO_ID["pit"] == 2
        assert CATEGORY_NAME_TO_ID["crash"] == 3
        assert CATEGORY_NAME_TO_ID["anomaly"] == 4

    def test_default_params_singleton(self):
        """DEFAULT_PARAMS should be a valid instance."""
        assert isinstance(DEFAULT_PARAMS, ThresholdParams)
