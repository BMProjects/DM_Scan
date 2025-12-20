# Architecture Overview (v0.3.0)

- **Data layer**: SQLite (`microlens_metadata.db`) remains the single source of truth for metadata; images live under `organized_tiffs/` (or user-specified root). Validation scripts ensure paths exist and frame counts meet expectations.

- **Infrastructure**: 
  - `logging.py`: Centralized logging via `get_logger()` for consistent output formatting
  - `exceptions.py`: Custom exception hierarchy (`MicrolensError`, `DatabaseError`, `ImageLoadError`, `DetectionError`, `ConfigurationError`)

- **Detection module** (refactored in v0.3.0):
  - `detection/base.py`: Abstract interface (`BaseDetector`, `DetectionResult`) for pluggable detectors
  - `detection/params.py`: Parameter definitions and validation
  - `detection/features.py`: Feature extraction (DC, temporal std, ROI estimation)
  - `detection/masks.py`: Mask building and component classification (scratch/pit/crash/anomaly)
  - `detection/rendering.py`: Visualization and result export
  - `detection/threshold.py`: `ThresholdDetector` implementation of classical 28-frame CLAHE + threshold + morphology + geometric classification

- **Feature export**: `features/five_step_phase.py` provides 5-step phase-shifting method for phase/amplitude/DC computation, accessible via CLI `microlens-defects phase5`.

- **ML stage (planned)**: `ml/` will host DINOv3 encoder + YOLO12 detection/segmentation heads, dataset conversion, training, and evaluation CLI. Experiment tracking recommended via MLflow/W&B. All ML detectors will implement `BaseDetector` interface for seamless swapping.

- **Semi-supervised (planned)**: `semi/` manages pseudo-label generation, confidence filtering, manual review loop, and maintains frozen golden test sets.

- **Presentation**: CLI unified via Typer; GUI可由 `gui_frontend.py` 或后续 Streamlit 前端承载。

- **Testing**: Unit tests with pytest, shared fixtures in `conftest.py`, module-specific tests cover params, features, masks, and detector interface.

- **CI/CD**: GitHub Actions workflow (`.github/workflows/ci.yml`) runs ruff, mypy, and pytest on Python 3.9-3.11.

目录摘要：
- `src/microlens_defects/`：库代码（logging, exceptions, data/io, detection, features, ml, semi, viz, cli）
- `configs/`：可版本化的 YAML 参数
- `docs/`：架构、流水线、测试大纲
- `examples/`：最小样例或指向 LFS/DVC 的数据引用
- `tests/`：单元与黄金样本测试

