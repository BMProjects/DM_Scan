"""Abstract base classes for defect detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DetectionResult:
    """Container for detection results.
    
    Attributes:
        mask: Binary defect mask (H x W, uint8, 0/255)
        annotations: List of COCO-format annotation dicts
        metadata: Additional metadata (params snapshot, timing, etc.)
    """

    mask: np.ndarray
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask is None:
            raise ValueError("mask cannot be None")


class BaseDetector(ABC):
    """Abstract base class for defect detectors.
    
    All detector implementations (threshold-based, ML-based, etc.) should
    inherit from this class to ensure consistent interface.
    """

    @abstractmethod
    def detect(self, stack: np.ndarray) -> DetectionResult:
        """Run detection on an image stack.
        
        Args:
            stack: Image stack with shape (H, W, N) where N is frame count
            
        Returns:
            DetectionResult containing mask, annotations, and metadata
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get current detector parameters.
        
        Returns:
            Dictionary of parameter names to values
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name for identification.
        
        Returns:
            Human-readable detector name
        """
        pass
