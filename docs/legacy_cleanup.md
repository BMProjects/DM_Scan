# Legacy Cleanup & Iteration Notes

为聚焦当前目标（数据库规范化 + 阈值基线 + 即将到来的 DINOv3/YOLO12 + 半监督流程），以下文件已清理：

- `defect_segmentation_threshold.py`：已完全模块化为 `src/microlens_defects/detection/threshold.py`，并由 `microlens-defects detect` CLI 替代。
- `detect_defects_29stack.py`：旧 29 帧直接检测实现，核心思路（DC 局部异常 + 拟合残差 + 形态学）已在 `docs/architecture.md` 留档，后续如需可在 `features/` 重新引入。
- `process_lens_defects.py`：29 帧特征导出脚本，功能将由未来的 `features/` 与 `ml/` 数据准备脚本接手。
- 其他实验/工具脚本（已删除，留档以便追溯思路）：`circle_grid_extract.py`、`compute_from_db.py`、`defect_interactive.py`、`prepare_test_images.py`、`pthtr2DB.py`、`view_npz_features.py`、`gui_frontend.py`；GUI 说明 `README_GUI.md` 同步移除。若需恢复，可从 Git 历史提取并迁移到新的包模块或 notebooks。

已恢复/保留的重要功能：
- 五步相移法：现以 `src/microlens_defects/features/five_step_phase.py` + CLI `microlens-defects phase5` 提供。

保留的文档（如 `project_summary.md`、`algorithm_test_plan.md`）记录了更早期的设计与测试计划，可作为迭代参考。若需恢复某段旧算法，可在 Git 历史中查找这些文件。
