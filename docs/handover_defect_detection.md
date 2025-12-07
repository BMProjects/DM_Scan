# 缺陷检测与数据库压缩转存功能交接文档

## 一、工程整体结构和角色概览

与“从原始文件夹压缩转存到数据库，并基于转存数据做缺陷检测”直接相关的主要脚本（均在项目根目录）：

- `pthtr2DB.py`：从原始 TIFF 目录扫描、压缩、整理到 `organized_tiffs`，并写入 SQLite 数据库 `microlens_metadata.db`（含缩略图、时间戳等元数据）。这是“导入+压缩+建库”的核心。
- `gui_frontend.py`：Tkinter 图形界面，用于从桌面启动并配置 `pthtr2DB.py` 导入流程，面向非技术用户。
- `defect_segmentation_threshold.py`：从 SQLite 数据库读取 28 帧条纹图像栈，直接执行缺陷检测与分割，输出 DC 图、全局掩码、叠加图和 COCO 标注，是当前推荐的“从数据库直接到缺陷结果”的主干脚本。
- `detect_defects_29stack.py`：对 29 帧条纹图像栈做频域拟合+时域/空间异常检测，输出一张缺陷掩码，可作为独立命令行工具，也被下游脚本复用。
- `process_lens_defects.py`：从数据库+文件系统读取某个镜片/光栅/左右片的 29 帧栈，调用 `detect_defects_29stack.py` 的核心函数，仅导出特征（DC、残差 STD、temporal STD、ROI）为 `.npz`，供后续交互式/更复杂的检测使用。
- `defect_interactive.py`：从 `process_lens_defects.py` / `detect_defects_29stack.py` 导出的 `.npz` 特征文件出发，做可调参数的缺陷检测（划痕/坑/密集破损），输出掩码和 COCO 标注，是基于“先导出特征再检测”的实验/交互版流程。
- `prepare_test_images.py`：从数据库里随机抽若干 `phase_index=29` 的图像到 `defect_test_images`，用于缺陷算法的快速手工测试。
- `view_npz_features.py`：查看 `lens_feature_npz/*.npz` 中的 DC/残差/temporal/ROI 等特征图，可视化调试用。

辅助/相关但非主干的模块：

- `compute_from_db.py` + `five_step_phase.py`：示例如何从 DB 取出若干帧，做五步相移相位重建（相位计算部分，与当前阈值分割缺陷检测相对独立）。
- `circle_grid_extract.py`：圆点/网格检测算法，可能用于标定或其它实验任务。

---

## 二、功能 1：从原始文件夹进行图像数据库压缩转存

### 1. 数据与目录约定

- 原始图像根目录（默认）：
  - `DEFAULT_ROOT_DIR = /run/media/bm/Data/Data/MicroLens/`，实际运行时可通过参数或 GUI 选择。
- 目录层级约定（由 `pthtr2DB.py` 解析）：
  - 结构类似：`glasses_id / lens_side / grating_type / ... / image_xx.tif`
  - `glasses_id`：眼镜编号（任意字符串，如 `2006`）。
  - `lens_side`：通过目录名包含 `left/right` 或 `lf/rf` 识别。
  - `grating_type`：仅允许 `heng`, `zong`, `cycle`。
  - 每个文件名必须带有数字索引，用于推断 `phase_index`（0–28）。
- 输出组织目录（默认）：`organized_tiffs/`
- SQLite 数据库文件（默认）：`microlens_metadata.db`

数据库表结构（由 `pthtr2DB.setup_database()` 建立，表名 `images`）：

- `id`：主键
- `glasses_id`：眼镜编号
- `lens_side`：左右片
- `grating_type`：光栅类型（`heng/zong/cycle`）
- `phase_index`：相位帧索引（整数）
- `original_path`：原始图像绝对路径（唯一）
- `file_path`：压缩/整理后的相对路径（相对 `organized_tiffs`，唯一）
- `compression_mode`：压缩方式（`none` 或 `jpeg`）
- `thumbnail`：缩略图 JPEG 二进制
- `timestamp`：从图像左上角时间戳区域 OCR 识别并解析后的时间字符串（若识别成功）

### 2. 核心程序及调用关系

#### 主流程脚本：`pthtr2DB.py`

- `main()`：
  - 解析命令行参数：
    - `--root`：原始图像根目录
    - `--db`：数据库路径
    - `--outdir`：整理后图像目录
    - `--compression`：`none` / `jpeg`
    - `--thumb`：缩略图尺寸（如 `256` 或 `256x256`）
    - `--no-timestamp`：关闭时间戳 OCR 处理
  - 设置全局变量：`ROOT_DIR`, `DATABASE_FILE`, `ORGANIZED_IMAGES_DIR`, `COMPRESSION_MODE`, `THUMBNAIL_SIZE`。
  - 调用 `setup_database()` 创建表。
  - 调用 `process_and_store_images()` 执行扫描/压缩/入库。

- `iter_source_images(ROOT_DIR)`：
  - 遍历原始根目录下的所有 `*.tif/*.tiff` 文件，生成器形式输出源路径。

- `parse_image_metadata(src_path)`：
  - 根据路径层级及文件名提取：
    - `glasses_id`, `lens_side`, `grating_type`, `phase_index` 等元数据。
  - 做各种合法性校验（未知光栅、错误相位索引等）。

- `process_timestamp_region(image)` + `get_paddle_ocr()`：
  - 在图像左上角寻找低灰度区域，粗略定位时间戳带。
  - 截取后利用 `PaddleOCR`（GPU）识别文本。
  - 使用 `parse_timestamp_text()` 从 OCR 文本中解析日期+时间。
  - 将时间戳区域涂黑（便于后续算法不受干扰），返回处理后图像及时间戳字符串。

- `create_thumbnail_bytes(im, size)`：
  - 调用 Pillow，将原图归一化到 8bit 灰度或 RGB，按给定高度生成缩略图 JPEG，返回 bytes 存入 DB。

- `save_processed_image(processed_image, src_path, metadata, original_format)`：
  - 依据 `COMPRESSION_MODE` 决定输出格式：
    - `none`：无压缩（保持 TIFF 或原始格式）；
    - `jpeg`：统一转成 JPEG 以减小体积。
  - 按 `glasses_id/lens_side/grating_type/phase_index.*` 组织到 `ORGANIZED_IMAGES_DIR` 内。
  - 若路径冲突，通过 `ensure_unique_path()` 自动加后缀避免覆盖。

- `process_and_store_images()`：
  - 循环遍历 `iter_source_images(ROOT_DIR)` 得到每一张原始图。
  - 对每张图：
    - 调用 `parse_image_metadata()` 解析元信息；
    - 跳过已处理过的 `(glasses_id, lens_side, grating_type, phase_index)` 组合（避免重复栈）；
    - 用 Pillow 打开图像，生成缩略图和经过时间戳处理后的图像；
    - 调用 `save_processed_image()` 写入压缩文件；
    - 在 `images` 表中执行 `INSERT`：
      - `original_path` = 源绝对路径；
      - `file_path` = 相对 `ORGANIZED_IMAGES_DIR` 的路径；
      - 写入缩略图、时间戳等。
    - 若 DB 已存在相同记录（唯一索引冲突），自动跳过并删除刚写出的重复文件。
  - 结束后调用 `verify_database_integrity()` 进行数量对账。

- `verify_database_integrity()`：
  - 统计：
    - DB 中图像记录数、不同 `glasses_id` 数、左右片计数。
    - 原始目录中可识别图像文件数。
  - 比较两者差异是否在允许阈值内（`COUNT_DIFF_ABS`、`COUNT_DIFF_PERC`）。
  - 打印汇总日志，供 GUI 或命令行用户确认导入是否“基本完整”。

#### 图形界面：`gui_frontend.py`

- UI 功能：
  - 选择 `ROOT_DIR`（原始 TIFF 所在位置）。
  - 选择 `Organized dir`（输出压缩图像目录）。
  - 选择 `Compression` 模式（`none/jpeg/tiff`，当前 `pthtr2DB` 实际支持 `none/jpeg`）。
  - 设置缩略图尺寸。
  - 按钮：
    - `Run Import`：启动导入流程。
    - `Verify DB`：调用 `verify_database_integrity()` 做数量校验。
    - `Clear Log`：清空日志窗口。

- 调用关系：
  - 在后台线程内动态 `importlib.reload(pthtr2DB)`，保证修改 `pthtr2DB.py` 后可即时生效。
  - 设置 `pthtr2DB.ROOT_DIR`、`ORGANIZED_IMAGES_DIR`、`COMPRESSION_MODE`、`THUMBNAIL_SIZE` 等模块级变量。
  - 顺序调用：
    - `pthtr2DB.setup_database()`
    - `pthtr2DB.migrate_add_columns()`（如果存在；用于老 DB 升级）
    - `pthtr2DB.process_and_store_images()`
  - `Verify DB` 按钮则调用 `pthtr2DB.verify_database_integrity()`，并用弹窗显示结果。

#### 实际使用建议

- 命令行批量导入（推荐先在测试子目录试跑）：

```bash
python pthtr2DB.py \
  --root /path/to/raw_tiffs \
  --db microlens_metadata.db \
  --outdir organized_tiffs \
  --compression jpeg \
  --thumb 256x256
```

- 实验室日常使用可优先通过：

```bash
python gui_frontend.py
```

由 GUI 帮忙选择源/目标目录，方便非开发人员操作。

---

## 三、功能 2：利用转存后的图像进行缺陷检测

转存后的数据由两部分组成：

- 文件系统：`organized_tiffs/` 中的压缩图像。
- 数据库：`microlens_metadata.db` 中的 `images` 表（提供 `(glasses_id, lens_side, grating_type, phase_index)` 与 `file_path` 的对应关系）。

在此基础上，目前有两条较成熟的检测流程。

### 2.1 主推荐流程：直接从数据库到缺陷分割（`defect_segmentation_threshold.py`）

入口脚本：`defect_segmentation_threshold.py`（已在 `README.md` 中详细说明）。

#### 输入

- SQLite 数据库：`--db`（默认 `microlens_metadata.db`）
- 图像根目录：`--img-root`（默认 `organized_tiffs`，脚本会用 `images.file_path` 构造完整路径）
- 指定单样本：`--glasses / --side / --grating`
- 批量模式：`--all`（遍历所有组合）+ 可选 `--limit` 限制前 N 个
- 帧数：`--num-frames`（默认 28，只取前 28 帧，丢弃历史上的第 29 帧平行光）

#### 核心组件与调用关系

- `ThresholdParams`：
  - 集中管理所有阈值、核大小等关键参数，支持 `ensure_valid()` 做基本校正（保证奇数核、正数阈值等）。
  - 包含：
    - Temporal 背景抑制参数；
    - 全局阈值+形态学清理参数；
    - 密集破损（crash）检测参数；
    - 划痕几何筛选、突出性阈值等。

- `ImageStackLoader`：
  - 持有 `database`（SQLite 文件）与 `image_root`（图像目录）。
  - `list_combinations(min_frames)`：
    - 查询 DB，按 `glasses_id/lens_side/grating_type` 分组，统计帧数。
    - 只保留帧数 ≥ `min_frames` 的组合。
  - `load_stack(glasses_id, lens_side, grating_type, max_frames)`：
    - 从 DB 查询对应组合所有 `file_path` 和 `phase_index`；
    - 将相对路径拼接到 `image_root` 上（若是绝对路径则直接使用）；
    - 用 OpenCV (`cv2.imread`) 读取为灰度 float32 数组；
    - 堆成 `H×W×N` 的 stack，如 N>max_frames，则裁剪至前 `max_frames` 帧。

- 特征构建与缺陷检测（在 `run_detection_for_stack()` 等函数中实现）：
  - 从 28 帧 stack 计算：
    - DC 图（时间平均）。
    - Temporal 标准差图。
  - 使用 CLAHE 增强 DC 图，对 ROI 区域裁剪。
  - 自适应阈值 + 形态学操作生成全局缺陷 mask。
  - 结合密度图（局部密度/面积阈值）识别大面积 crash。
  - 对连通域做几何分类：
    - 划痕（scratch）：细长、高长宽比，使用延伸/合并策略。
    - 圆坑（pit）、不规则异常（anomaly）：根据圆度、面积和“突出性”阈值筛选。
  - 生成 COCO 风格的 `annotations`（包含 `bbox/area/segmentation/rotated_points` 等）。

- `record_metadata(save_root, entry)`：
  - 将每个样本的统计信息追加到 `metadata_summary.csv` 与 `metadata_summary.jsonl`。

#### 输出目录结构

`--save-dir`（默认 `./defect_detection_outputs`）下：

- `save-dir/<glasses_id>_<lens_side>_<grating_type>/`
  - `*_dc_clahe.png`：CLAHE 处理后的 DC 全图（模型训练的输入）。
  - `*_global_mask.png`：全局缺陷蒙版（像素级标签）。
  - `*_overlay.png`：DC+检测结果叠加，用于人工抽检。
  - `*_annotations.json`：COCO 格式实例标注。
- `save-dir/metadata_summary.csv`：每个样本一行的统计与参数快照。
- `save-dir/metadata_summary.jsonl`：同上，逐行 JSON 形式，适合下游程序读取。

#### 典型调用方式

- 批量扫库（前 10 个组合）：

```bash
python defect_segmentation_threshold.py \
  --all --limit 10 \
  --db microlens_metadata.db \
  --img-root organized_tiffs \
  --save-dir defect_detection_outputs
```

- 单样本（指定镜片/侧/光栅）：

```bash
python defect_segmentation_threshold.py \
  --glasses 2006 --side left --grating cycle \
  --db microlens_metadata.db \
  --img-root organized_tiffs \
  --save-dir defect_detection_outputs
```

#### 与导入模块的关系

- 依赖 `pthtr2DB.py` 建立的：
  - `microlens_metadata.db` 中 `images` 表（`glasses_id/lens_side/grating_type/phase_index/file_path`）。
  - `organized_tiffs/` 中整理/压缩后的图像。
- 完整链路：
  - 原始目录 → `pthtr2DB.py`（压缩 & 建库）→ `microlens_metadata.db + organized_tiffs` → `defect_segmentation_threshold.py`（缺陷检测+标注+统计）。

### 2.2 实验/交互流程：“先导出特征，再进行缺陷检测”

这一条更偏研究与调参，通常在算法开发阶段使用。

#### 1）频域缺陷检测基础：`detect_defects_29stack.py`

- 直接从一个包含约 29 帧的目录读取图像栈（默认前 29 个 `*.tif`），计算：
  - DC 图；
  - 一阶 DFT 系数 `C1`；
  - 基于拟合的残差序列；
  - residual/std/temporal_std 图。
- 构造：
  - `static_mask`：DC 空间异常（静态缺陷）。
  - `residual_mask`：残差时域异常（非条纹因素、镀膜破坏等）。
  - 合并并用 `scipy.ndimage` 做形态学清理，得到最终二值缺陷掩码。
- 可单独使用：

```bash
python detect_defects_29stack.py \
  --dir /path/to/stack_dir \
  --pattern '*.tif' \
  --N 29 \
  --out defect_mask.png \
  --export-npz features_out.npz
```

- 导出的 `.npz` 特征（`dc_map/residual_std_map/temporal_std_map/roi_mask`）也是之后 `process_lens_defects.py` / `defect_interactive.py` 的数据格式基础。

#### 2）从数据库批量导出特征：`process_lens_defects.py`

- 依赖：
  - `microlens_metadata.db`（同样来自 `pthtr2DB.py`）。
  - `organized_tiffs/`。
  - `detect_defects_29stack.py` 中的 `dft1_param_estimates/reconstruct_fitted_stack/estimate_roi_mask/save_feature_npz` 等函数。

- 主要流程：
  - `ImageStackLoader`：
    - 通过 DB 查询指定 `(glasses_id, lens_side, grating_type)` 的若干帧，按 `phase_index` 排序。
    - 拼接文件路径 (`file_path` + `ORGANIZED_IMAGES_DIR`) 读取为 `H×W×N` stack。
  - `export_features_for_combination()`：
    - 将 stack 裁剪为最多 28 帧（丢弃第 29 帧平行光）。
    - 调用 `det.dft1_param_estimates()` 等函数生成：
      - `dc_map`；
      - `residual_std_map`；
      - `temporal_std_map`；
      - `roi_mask`。
    - 调用 `det.save_feature_npz()` 保存为 `lens_feature_npz/<glasses>_<side>_<grating>.npz`。
  - `export_features_for_all()`：
    - 遍历 DB 中所有符合帧数条件的组合，批量导出特征。

- 命令示例：

```bash
# 单个组合
python process_lens_defects.py \
  --glasses 2006 --side left --grating cycle \
  --db microlens_metadata.db \
  --imgdir organized_tiffs \
  --feature-dir lens_feature_npz

# 批量导出所有
python process_lens_defects.py \
  --all \
  --db microlens_metadata.db \
  --imgdir organized_tiffs \
  --feature-dir lens_feature_npz
```

#### 3）基于特征的交互式缺陷检测：`defect_interactive.py`

- 输入：
  - `lens_feature_npz/*.npz` 文件（来源于 `process_lens_defects.py` 或直接 `detect_defects_29stack.py`）。
  - 内含：
    - `dc_map`
    - `residual_std_map`
    - `temporal_std_map`
    - `roi_mask`
- `DefectParams`：
  - 配置 temporal 种子阈值、Gabor 滤波参数、几何过滤阈值、密集破损参数等，适合精细调参。
- 检测步骤：
  - 在 ROI 内利用 `temporal_std_map` 做高置信缺陷种子；
  - 使用阈值+方向滤波（Gabor）定位划痕、点状缺陷；
  - 基于局部密度 + temporal 亮度识别 crash 区；
  - 连通域分析 + 几何特征（面积、长宽比、圆度等）分类为 `scratch/pit/crash/...`；
  - 输出掩码、叠加图和 COCO 标注。

#### 4）其它配套脚本

- `prepare_test_images.py`：
  - 从 DB 中随机抽取指定相位 `phase_index` 的图像（默认 29）复制到 `defect_test_images`，便于手工/脚本快速测试缺陷检测算法。
- `view_npz_features.py`：
  - 读取单个 `.npz` 特征包并用 matplotlib 展示 `dc_map/residual_std_map/temporal_std_map/roi_mask`，用于调试与人工检视特征质量。

---

## 四、交接要点与建议

### 1. 整体链路梳理（推荐主线）

1. `pthtr2DB.py` / `gui_frontend.py`：从原始 TIFF 栈 → 压缩图像 (`organized_tiffs`) + 元数据 DB (`microlens_metadata.db`)。
2. `defect_segmentation_threshold.py`：从 DB+整理图像 → DC & Temporal 特征 → 阈值/形态学 + 几何分类 → 缺陷掩码 & COCO 标注 & 统计汇总。

### 2. 旧版/实验性链路（用于研究与扩展）

- `detect_defects_29stack.py` + `process_lens_defects.py` + `defect_interactive.py`：
  - 原始或 DB → stack → 特征 `.npz` → 交互式检测/调参 → 掩码和标注。

### 3. 接手时建议先做的验证步骤

用一小批样本（例如 1–2 个 `glasses_id`）跑一遍：

1. 执行导入：`pthtr2DB.py`（或 GUI），检查：
   - `organized_tiffs` 中目录结构和文件数量；
   - DB `images` 表中对应记录是否完整、`file_path` 是否可访问；
   - `verify_database_integrity()` 的差异是否在容许范围。
2. 执行检测：`defect_segmentation_threshold.py --limit 5`，检查：
   - `defect_detection_outputs` 下是否生成对应子目录；
   - `*_overlay.png` 可视化是否合理；
   - `metadata_summary.*` 中统计是否与肉眼观察大体一致。

### 4. 后续扩展点

- 若需改动导入流程（例如增加新字段、识别更多元数据），建议集中修改：
  - `pthtr2DB.parse_image_metadata()`、`setup_database()` 和相关 `INSERT` 语句。
- 若要调整/升级缺陷检测逻辑，优先修改：
  - `defect_segmentation_threshold.py` 中的 `ThresholdParams` 和与 `run_detection_for_stack()` 相关的阈值/形态学流程；
  - 或在实验阶段使用 `defect_interactive.py` 对 `.npz` 特征做更复杂的判别逻辑。

