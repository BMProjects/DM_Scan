"""Microlens schlieren toolkit: microstructure morphology and defect detection.

Two classical, dependency-light pillars for defocus/phase-shift microlens
inspection:

* **Morphology** -- five-step phase-shifting (:mod:`microlens_defects.features`)
  recovers wrapped phase / DC / modulation amplitude from fringe images.
* **Defect detection** -- a 28-frame threshold baseline
  (:mod:`microlens_defects.detection`) exports masks, COCO annotations, and
  overlays for scratches, pits, crashes, and anomalies.

Learning-based detection lives in a separate project; this package is meant to
slot into a larger microstructure-lens recognition system as its classical
morphology/defect front end.
"""

from microlens_defects.exceptions import (
    ConfigurationError,
    DatabaseError,
    DetectionError,
    ImageLoadError,
    MicrolensError,
)
from microlens_defects.logging import get_logger, set_log_level

__all__ = [
    "__version__",
    # Exceptions
    "MicrolensError",
    "DatabaseError",
    "ImageLoadError",
    "DetectionError",
    "ConfigurationError",
    # Logging
    "get_logger",
    "set_log_level",
]

__version__ = "0.4.0"
