# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-20

### Added
- **Logging framework**: Centralized logging via `microlens_defects.logging.get_logger()`
- **Exception hierarchy**: Custom exceptions (`MicrolensError`, `DatabaseError`, `ImageLoadError`, `DetectionError`, `ConfigurationError`)
- **Abstract detector interface**: `BaseDetector` and `DetectionResult` for pluggable detectors
- **Comprehensive unit tests**: Added 5 new test modules with pytest fixtures
  - `tests/conftest.py`: Shared fixtures
  - `tests/test_params.py`: Parameter validation tests
  - `tests/test_features.py`: Feature extraction tests
  - `tests/test_masks.py`: Mask building tests
  - `tests/test_detector.py`: Detector interface tests
- **CI/CD**: GitHub Actions workflow for linting, type checking, and testing on Python 3.9-3.11

### Changed
- **detection module refactoring**: Split `threshold.py` (630 lines) into 6 focused modules:
  - `detection/base.py`: Abstract interface
  - `detection/params.py`: Parameter definitions
  - `detection/features.py`: Feature extraction functions
  - `detection/masks.py`: Mask building and classification
  - `detection/rendering.py`: Visualization and export
  - `detection/threshold.py`: `ThresholdDetector` class implementation
- **CLI improvements**: Replace `print()` calls with structured logging
- **Import paths**: Added re-exports for backward compatibility
- **Version bump**: 0.2.0 → 0.3.0

### Fixed
- Import organization in CLI (moved dataclass imports to top)
- Parameter validation now uses centralized `ThresholdParams.ensure_valid()`

## [0.2.0] - Previous Release

### Added
- 28-frame threshold detection baseline
- 5-step phase-shifting method
- CLI with Typer
- YAML configuration support
- COCO format output
- SQLite-based data management

### Documentation
- Initial README, architecture, and pipeline documentation
- Handover documentation for defect detection

[0.3.0]: https://github.com/YOUR_ORG/DM_Scan/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/YOUR_ORG/DM_Scan/releases/tag/v0.2.0
