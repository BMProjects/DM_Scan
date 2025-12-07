# Architecture Overview

- **Data layer**: SQLite (`microlens_metadata.db`) remains the single source of truth for metadata; images live under `organized_tiffs/` (or user-specified root). Validation scripts ensure paths exist and frame counts meet expectations.
- **Classical baseline**: `microlens_defects.detection.threshold` implements the 28-frame CLAHE +阈值 + 形态学 + 几何分类流程，输出全图蒙版、COCO 标注、元数据摘要。
- **Feature export (legacy/29 帧)**: 旧脚本已移除（见 `docs/legacy_cleanup.md`），如需恢复可在 Git 历史取用；后续会以 `features/` 模块形式提供 DC/temporal/residual/ROI `.npz`。
- **五步相移测量（新增）**: `features/five_step_phase.py` 提供五步相移法相位/幅值/DC 计算，CLI 子命令 `microlens-defects phase5` 可直接从 5 张条纹图生成结果。
- **ML stage (planned)**: `ml/` 将承载 DINOv3 编码 + YOLO12 检测/分割头，数据集转换、训练、评估 CLI。实验记录建议用 MLflow/W&B。
- **半监督 (planned)**: `semi/` 管理伪标签生成、置信度过滤、人工审校回流，维护冻结的黄金测试集。
- **Presentation**: CLI 统一由 Typer 提供；GUI 可由 `gui_frontend.py` 或后续 Streamlit 前端承载。

目录摘要：
- `src/microlens_defects/`：库代码（data/io, detection, ml, semi, viz, cli）。
- `configs/`：可版本化的 YAML 参数。
- `docs/`：架构、流水线、测试大纲。
- `examples/`：最小样例或指向 LFS/DVC 的数据引用。
- `tests/`：单元与黄金样本测试。
