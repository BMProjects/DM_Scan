# Pipeline & Data Conventions

## Inputs
- **SQLite**: table `images(glasses_id, lens_side, grating_type, phase_index, file_path, …)`; phase_index 默认 0..28，当前阈值法使用前 28 帧。
- **Image root**: default `organized_tiffs/`; `file_path` 优先在该根路径下解析，若为绝对路径则直接使用。

## Outputs (threshold baseline)
- `save_dir/<glasses>_<side>_<grating>/`
  - `<tag>_dc_clahe.png`：CLAHE DC 全图
  - `<tag>_global_mask.png`：像素级全局蒙版
  - `<tag>_overlay.png`：彩色叠加
  - `<tag>_annotations.json`：COCO 标注（scratch/pit/crash/anomaly）
- `save_dir/metadata_summary.csv|jsonl`：每样本一行的统计与参数快照

## CLI (new)
```bash
microlens-defects detect --db microlens_metadata.db --img-root organized_tiffs \
  --all --limit 10 --save-dir defect_detection_outputs \
  --config configs/detect_threshold.yaml
```
- 单样本：`--glasses 2006 --side left --grating cycle`
- 若仅给 `--limit`，CLI 自动启用 `--all`。

## Config
- YAML 位于 `configs/`; 每个键对应 `ThresholdParams` 字段，可覆盖默认值。
- 运行时打印当前参数并写入 `param_snapshot` 以便复现。

## Roadmap hooks
- **ML 数据准备**：阈值法输出可直接生成 COCO/掩码训练集；后续在 `ml/` 中增加转换、切分、数据版本化脚本。
- **半监督**：`semi/` 将接受教师模型伪标签 + 置信度过滤 + 人审校；结果回写 DB/COCO。
- **测试大纲**：标准化评测脚本将在 `cli eval` 中落地，指标覆盖 mAP/mIoU/FPR/时延。
