"""Tests for ThresholdParams validation.

NOTE: This file is kept for backward compatibility. New tests should be added
to tests/test_params.py instead.
"""

from microlens_defects.detection.params import ThresholdParams


def test_params_validation():
    """Test basic parameter validation."""
    params = ThresholdParams(
        open_kernel=2, 
        close_kernel=0, 
        adaptive_block=2, 
        bg_max_sigma=-1
    )
    params.ensure_valid()
    
    assert params.open_kernel % 2 == 1
    assert params.close_kernel % 2 == 1
    assert params.adaptive_block % 2 == 1
    assert params.bg_max_sigma > 0
