# 缺陷检测与标注（数据库→特征→分割/标注→数据集）

面向非计算机背景用户，提供开箱即用的 Python 包 `microlens-defects` (v0.3.0)：
- 28 帧阈值法基线：一键生成全图蒙版、COCO 标注、统计汇总；
- 五步相移法：计算相位/DC/幅值；
- 统一检测器接口：为机器学习检测器预留扩展；
- 单一 CLI + YAML 配置，配套文档集中在 `docs/index.md`；
- 完善的日志和异常处理，便于调试和错误追踪。

---

## 快速开始

1) 安装（复制即用）
```bash
python -m venv .venv && source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .   # 安装 CLI
```

2) 批量检测（最多 10 个样本）
```bash
microlens-defects detect --all --limit 10 \
  --db microlens_metadata.db --img-root organized_tiffs \
  --save-dir defect_detection_outputs \
  --config configs/detect_threshold.yaml
```

3) 单样本检测
```bash
microlens-defects detect --glasses 2006 --side left --grating cycle \
  --db microlens_metadata.db --img-root organized_tiffs \
  --save-dir defect_detection_outputs
```

4) 五步相移法（输入 5 张条纹图）
```bash
microlens-defects phase5 ./my_five_frames --pattern "*.tif" --output phase_result.npz
```

提示：仅传 `--limit` 会自动启用 `--all`；`--all` 不可与单样本参数同时使用。

更多资料：集中在 `docs/index.md`；若遇安装/运行问题，可查 `docs/troubleshooting.md`。

---

## 目录结构
- `src/microlens_defects/`：库代码  
  - `logging.py`：统一日志框架
  - `exceptions.py`：自定义异常层次
  - `data/db.py`：SQLite + 图像根目录加载器  
  - `detection/`：缺陷检测模块（已模块化）
    - `base.py`：抽象检测器接口 (`BaseDetector`, `DetectionResult`)
    - `params.py`：参数定义与验证
    - `features.py`：特征提取函数
    - `masks.py`：掩码构建与组件分类
    - `rendering.py`：可视化渲染与结果导出
    - `threshold.py`：阈值检测器实现 (`ThresholdDetector`)
  - `cli/app.py`：Typer CLI 入口  
  - `features/five_step_phase.py`：五步相移法
  - `ml/`, `semi/`, `viz/`：ML/半监督/可视化占位  
- `configs/`：YAML 参数（`detect_threshold.yaml`）  
- `docs/`：集中文档（架构/流水线/测试等）  
- `tests/`：单元测试（pytest + fixtures）
  - `conftest.py`：共享 fixtures
  - `test_params.py`, `test_features.py`, `test_masks.py`, `test_detector.py`：模块测试
- `examples/`：最小数据/Notebook 占位  
- `.github/workflows/ci.yml`：GitHub Actions CI/CD（lint + type check + pytest）

---

## 阈值法检测原理（28 帧）

- 条纹相位堆栈：单次拍摄得到 28 帧；像素理想为正弦随帧变化。
- DC/temporal：每像素求均值 DC 与时间标准差（temporal std）；DC 取反 + CLAHE 作为主要分割底图。
- 29→28 帧：历史第 29 帧为平行光照，当前默认丢弃。
- 流程：
  1) CLAHE DC 自适应阈值 + 形态学 → global mask 粗分割；
  2) 高斯密度阈值 → crash（大面积破损）；
  3) 几何/突出性分类 → 划痕（延伸/合并）、pit、anomaly；
  4) 生成全图蒙版、COCO 标注与叠加图。

---

## 数据库与输入约定
- SQLite 表 `images` 至少包含：`glasses_id`、`lens_side`、`grating_type`、`phase_index`(0..28)、`file_path`。
- 图像根目录 `--img-root`（默认 `organized_tiffs`）：优先按相对路径解析，若不存在再尝试绝对路径。

---

## 输出结构
- `save-dir/<glasses>_<side>_<grating>/`
  - `<tag>_dc_clahe.png`：CLAHE DC 全图
  - `<tag>_global_mask.png`：全局蒙版
  - `<tag>_overlay.png`：叠加可视化
  - `<tag>_annotations.json`：COCO 标注（scratch/pit/crash/anomaly）
- `save-dir/metadata_summary.csv|jsonl`：每样本统计与参数快照

COCO schema：categories `scratch(1)`,`pit(2)`,`crash(3)`,`anomaly(4)`；annotations 含 `bbox/area/segmentation/rotated_points` 等。

---

## 关键参数
- 阈值/形态学：`adaptive_block` / `adaptive_c` / `open_kernel` / `close_kernel`
- 密度 crash：`density_kernel_ratio` / `density_threshold` / `dense_min_area`
- 划痕：`scratch_min_len` / `scratch_min_aspect` / 延伸与合并参数
- 突出性：`prominence_min_value`（仅 pit/anomaly 强制）

调整方法：编辑 `configs/detect_threshold.yaml` 或 CLI 覆盖；运行时打印参数并写入 `param_snapshot`。

---

## FAQ
- 报“缺少参数”：仅传 `--limit` 会自动启用 `--all`；否则补齐 `--glasses/--side/--grating`。
- crash 过多：提高 `density_threshold` 或 `dense_min_area`，或减小 `density_kernel_ratio`。
- 划痕偏少：放宽 `scratch_min_len/scratch_min_aspect` 或延伸参数；划痕不受 `prominence_min_value` 约束。
- 路径缺失：确保 `images.file_path` 能在 `--img-root` 下解析，或使用可访问的绝对路径。

---

## 路线图（结合当前进度）
1) 已完成：数据库规范化；阈值法基线 + 初始高置信度标注。  
2) 进行中：DINOv3 + YOLO12 ML 方法；半监督伪标签 + 人审校；冻结黄金测试集与标准评测（见 `docs/test_plan.md`）。  
3) 待办：`ml/` 与 `semi/` CLI/训练脚本、评测脚本 `cli eval`、样例数据与更多单测。 

---

## 开发与评测 SOP（简版）
- 环境：使用虚拟环境；`pip install -e .[dev]`，首选本地 CPU 跑通示例。
- 数据准备：创建/导入 SQLite `images` 表；准备 5/28 帧最小样例用于自检（见 `examples/minimal_data`）。
- 开发循环：单功能迭代在 notebooks 或 `src/microlens_defects`；提交前运行 `pytest` 和 CLI 冒烟（单样本 + `--limit 1`）。
- 评测：冻结黄金集，运行计划中的 `microlens-defects eval`（或现阶段脚本）生成 mAP/mIoU、FPR、时延；产出 CSV/JSON 报告与 PR 曲线。
- 回流：通过半监督/人工审校更新标注，写回 SQLite/COCO，记录数据版本与参数快照。

## 路线图进度表
| 阶段 | 状态 | 说明 | 入口 |
| --- | --- | --- | --- |
| 数据规范化 + 阈值基线 | 已完成 | 28 帧阈值法、SQLite 加载、COCO 导出 | `microlens-defects detect` |
| 五步相移法 | 已完成 | 5 帧相位/DC/幅值计算 | `microlens-defects phase5` |
| ML 检测（DINOv3+YOLO12） | 进行中 | 特征提取与检测头训练/推理脚本 | 待补 `ml/` CLI |
| 半监督回流 | 计划中 | 伪标签过滤 + 人审校回写，质量闸口 | 待补 `semi/` CLI |
| 标准化评测 | 计划中 | 黄金集 + `cli eval` 统一指标输出 | 待补 `cli eval` |

---

## 迭代与清理
- 旧脚本 `defect_segmentation_threshold.py` / `detect_defects_29stack.py` / `process_lens_defects.py` 已清理，功能分别由 `microlens_defects/detection/threshold.py`、未来 `features/` 数据导出、以及 CLI 统一入口承接。详见 `docs/legacy_cleanup.md`。
- 五步相移法已作为正式功能保留：`microlens-defects phase5`。
