# 开发与评测 SOP（详细）

## 开发与协作
- 环境：Python >=3.9，建议 `pip install -e .[dev]`；遵循 `.gitignore`，提交前执行格式化/lint/测试。
- 分支/提交：功能分支 + 小步提交，提交信息包含模块与要点（如 `cli: add eval command stub`）。
- 依赖管理：默认依赖为最小集，ML/YOLO/DINO 等重型组件放入 extras 或文档提示。

## 数据准备
- SQLite 必含表 `images(glasses_id, lens_side, grating_type, phase_index, file_path, …)`；路径优先在 `--img-root` 下解析。
- 最小样例：准备 5 张条纹图或 28 张 fringe 图 + 对应 SQLite，用于本地冒烟和 CI。
- 数据版本化：划分清单放置于 `splits/`（或 DVC/LFS），明确版本号/日期；黄金测试集只读。

## 阈值基线与五步相移
- 阈值法：`microlens-defects detect --glasses ...` 或 `--all --limit 1` 进行冒烟；输出蒙版/COCO/统计，参数快照记录到 `metadata_summary.*`。
- 五步相移：`microlens-defects phase5 <dir> --pattern '*.tif' --output phase_result.npz`；主要用于光学量测验证。

## ML / 半监督（规划中）
- ML 训练/推理：在 `ml/` 增加数据转换、特征提取（DINOv3）、检测头训练（YOLO12），统一入口预留为 `microlens-defects ml <subcommand>`。
- 半监督回流：在 `semi/` 实现教师伪标签 → 置信度/面积/一致性过滤 → 人审校 → 写回 SQLite/COCO，需记录数据版本与阈值。

## 评测与报告
- 统一入口：计划中的 `microlens-defects eval`（或脚本）接受模型权重、数据划分，输出 mAP/mIoU、FPR、时延等指标；生成 CSV/JSON/图表（PR、混淆矩阵）。
- 追溯：每次评测记录模型版本、提交哈希、数据划分、参数与运行时间；报告存档在 `artifacts/` 或 MLflow。

## 回归与质量闸口
- CI：lint + pytest + 最小样例跑通（阈值法/phase5）。
- 质量闸口：对于回流的伪标签或新模型，至少满足设定的 mAP/IoU/FPR 阈值；未达标不得覆盖黄金集。
