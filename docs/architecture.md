# Architecture & Pipeline

`microlens_defects` (v0.4.0) is a classical, dependency-light toolkit with two
pillars — **microstructure morphology** (five-step phase-shifting) and
**defect detection** (28-frame threshold baseline).

## Components

- **Data layer** — [data/db.py](../src/microlens_defects/data/db.py): SQLite
  (`microlens_metadata.db`) is the single source of truth for metadata; images
  live under `organized_tiffs/` (or a user-specified `--img-root`). The loader
  resolves `file_path` relative to the root first, then as an absolute path,
  and lists frame-complete sample combinations.
- **Infrastructure**:
  - [logging.py](../src/microlens_defects/logging.py): centralized `get_logger()`.
  - [exceptions.py](../src/microlens_defects/exceptions.py): `MicrolensError`,
    `DatabaseError`, `ImageLoadError`, `DetectionError`, `ConfigurationError`.
- **Detection module** ([detection/](../src/microlens_defects/detection/)):
  - `base.py`: `BaseDetector` / `DetectionResult` interface.
  - `params.py`: `ThresholdParams` definitions and validation.
  - `features.py`: feature extraction (DC, temporal std, ROI estimation).
  - `masks.py`: mask building and component classification (scratch/pit/crash/anomaly).
  - `rendering.py`: overlay, COCO, and metadata export.
  - `threshold.py`: `ThresholdDetector` + `run_threshold_detection` (CLAHE +
    adaptive threshold + morphology + geometric classification).
- **Morphology** — [features/five_step_phase.py](../src/microlens_defects/features/five_step_phase.py):
  five-step phase-shifting → phase / DC / amplitude / valid mask.
- **CLI** — [cli/app.py](../src/microlens_defects/cli/app.py): Typer app with
  `detect` and `phase5`.
- **Testing**: pytest with shared fixtures in `tests/conftest.py`; CI runs ruff,
  mypy, and pytest on Python 3.9–3.11 (`.github/workflows/ci.yml`).

## Inputs

- **SQLite** table `images(glasses_id, lens_side, grating_type, phase_index,
  file_path, …)`; `phase_index` 0..28, the threshold baseline uses the first 28 frames.
- **Image root**: default `organized_tiffs/`; missing frames are skipped with a
  WARN log.

## Outputs (threshold baseline)

`save_dir/<glasses>_<side>_<grating>/`:

| File | Content |
| --- | --- |
| `<tag>_dc_clahe.png` | DC + CLAHE base image (morphology) |
| `<tag>_global_mask.png` | pixel-level global mask |
| `<tag>_overlay.png` | colored defect overlay |
| `<tag>_annotations.json` | COCO annotations (scratch/pit/crash/anomaly) |

Plus `save_dir/metadata_summary.csv|jsonl` — one row per sample with stats and a
parameter snapshot for reproducibility.

## Configuration

YAML under [configs/](../configs/); each key maps to a `ThresholdParams` field
and overrides the default. Resolved parameters are logged and written to the
metadata snapshot.

## Scope

Learning-based detection is maintained in a separate project. The threshold
baseline still produces COCO/mask outputs suitable as training data for that
work, but no training/inference code lives here.
