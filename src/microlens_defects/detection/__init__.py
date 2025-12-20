"""Detection module for microlens defects.

This module provides both classical threshold-based and ML-based (planned)
defect detection capabilities.
"""

from .base import BaseDetector, DetectionResult
from .params import (
    CATEGORIES,
    CATEGORY_NAME_TO_ID,
    CATEGORY_SHORT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    ThresholdParams,
)
from .threshold import (
    ThresholdDetector,
    run_detection_for_stack,
    run_threshold_detection,
)

__all__ = [
    # Abstract interface
    "BaseDetector",
    "DetectionResult",
    # Parameters and constants
    "ThresholdParams",
    "DEFAULT_PARAMS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_NUM_FRAMES",
    "CATEGORIES",
    "CATEGORY_NAME_TO_ID",
    "CATEGORY_SHORT",
    # Threshold detector
    "ThresholdDetector",
    "run_detection_for_stack",
    "run_threshold_detection",
]
