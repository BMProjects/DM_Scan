"""Shared pytest fixtures for microlens-defects tests."""

from __future__ import annotations

import numpy as np
import pytest

from microlens_defects.detection.params import ThresholdParams


@pytest.fixture
def sample_stack() -> np.ndarray:
    """Generate a mock image stack (100x100x28) for testing."""
    np.random.seed(42)
    return np.random.randint(0, 255, (100, 100, 28), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def small_stack() -> np.ndarray:
    """Generate a small mock image stack (50x50x5) for faster tests."""
    np.random.seed(42)
    return np.random.randint(0, 255, (50, 50, 5), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def sample_dc_map() -> np.ndarray:
    """Generate a mock DC map (100x100)."""
    np.random.seed(42)
    return np.random.randint(50, 200, (100, 100), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Generate a mock binary mask with an ROI region."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return mask


@pytest.fixture
def default_params() -> ThresholdParams:
    """Get default ThresholdParams instance."""
    return ThresholdParams()
