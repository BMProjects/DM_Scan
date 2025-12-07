# Defect Detection Test Plan (baseline → ML)

## Scope
- 阈值法与后续 ML 模型（DINOv3+YOLO12）在冻结测试集上的标准化评测。
- 目标覆盖：分割精度、分类/检测精度、鲁棒性（边缘/微小/低对比）、无缺陷片误报率、性能（时延）。

## Dataset policy
- 测试集：未参与训练/调参，来自高置信度与人工复审的“黄金标注”。
- 划分清单版本化（`splits/`），保持可复现。
- 标注格式：COCO + 像素掩码；必要时导出 YOLO 格式用于对比。

## Metrics
- Segmentation: IoU、Dice、mIoU。
- Detection/Classification: mAP@50/50:95、Precision/Recall/F1、Confusion Matrix。
- Robustness: FPR on “无缺陷”样本；针对微小/边缘缺陷的子集统计。
- Performance: per-image latency（CPU/GPU）；模型尺寸与显存占用（记录项）。

## Execution steps
1) 冻结测试集与黄金标注。
2) 运行 `cli eval`（待实现）加载模型权重，批量推理测试集。
3) 生成指标表、PR 曲线、混淆矩阵；输出 JSON/CSV 报告。
4) 失败案例挖掘：列出 FP/FN 样例清单与截图，反馈给数据/算法迭代。

## Traceability
- 每次评测绑定：模型版本/提交哈希、超参快照、数据划分版本、运行时间戳。
- 报告与快照存档于 `artifacts/` 或 MLflow 记录。
