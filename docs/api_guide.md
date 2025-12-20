# API 使用指南 (v0.3.0)

本文档介绍 `microlens-defects` v0.3.0 的主要 API 和使用方式。

## 快速开始

### 1. 使用 CLI（推荐用于日常检测）

```bash
# 批量检测
microlens-defects detect --all --limit 10 \
  --db microlens_metadata.db \
  --img-root organized_tiffs \
  --save-dir defect_detection_outputs

# 单样本检测
microlens-defects detect --glasses 2006 --side left --grating cycle \
  --db microlens_metadata.db --img-root organized_tiffs

# 五步相移法
microlens-defects phase5 ./my_five_frames --output phase_result.npz
```

### 2. 使用 Python API（适用于二次开发）

#### 基本检测流程

```python
from pathlib import Path
from microlens_defects.data.db import ImageStackLoader
from microlens_defects.detection import ThresholdDetector, ThresholdParams

# 1. 加载图像栈
loader = ImageStackLoader(
    database=Path("microlens_metadata.db"),
    image_root=Path("organized_tiffs")
)

stack = loader.load_stack(
    glasses_id="2006",
    lens_side="left",
    grating_type="cycle",
    max_frames=28
)

# 2. 创建检测器
params = ThresholdParams(
    adaptive_block=41,
    scratch_min_len=40,
    # ... 其他参数
)
detector = ThresholdDetector(params)

# 3. 执行检测
result = detector.detect(stack)

# 4. 处理结果
print(f"检测到 {result.metadata['total_defects']} 个缺陷")
print(f"分类统计: {result.metadata['category_counts']}")

# result.mask: 二值掩码 (H x W, uint8)
# result.annotations: COCO格式标注列表
```

#### 自定义参数

```python
from microlens_defects.detection import ThresholdParams
import yaml

# 方式1: 从 YAML 加载
with open("configs/detect_threshold.yaml") as f:
    config = yaml.safe_load(f)

params = ThresholdParams(**config)
params.ensure_valid()  # 自动修正非法值

# 方式2: 代码中设置
params = ThresholdParams(
    density_threshold=0.3,  # 提高crash检测阈值
    scratch_min_len=50,     # 更长的划痕
    prominence_min_value=25.0  # 更高的突出性要求
)
```

---

## 模块说明

### 日志框架

```python
from microlens_defects.logging import get_logger, set_log_level
import logging

# 获取logger
logger = get_logger(__name__)

# 使用
logger.info("Processing sample: %s", sample_id)
logger.warning("Missing file: %s", path)
logger.error("Detection failed: %s", error)

# 调整日志级别
set_log_level(logging.DEBUG)
```

### 异常处理

```python
from microlens_defects.exceptions import (
    MicrolensError,
    DatabaseError,
    ImageLoadError,
    DetectionError,
    ConfigurationError
)

try:
    result = detector.detect(stack)
except ImageLoadError as e:
    logger.error("图像加载失败: %s", e)
except DetectionError as e:
    logger.error("检测过程出错: %s", e)
except MicrolensError as e:
    logger.error("未知错误: %s", e)
```

### 数据加载

```python
from microlens_defects.data.db import ImageStackLoader

loader = ImageStackLoader(
    database=Path("microlens_metadata.db"),
    image_root=Path("organized_tiffs"),
    validate_paths=True  # 启动时检查路径
)

# 列出所有可用组合
combinations = loader.list_combinations(min_frames=28)
print(f"找到 {len(combinations)} 个样本")

# 加载特定组合
stack = loader.load_stack("2006", "left", "cycle", max_frames=28)
print(f"图像栈形状: {stack.shape}")  # (H, W, 28)
```

### 检测器接口

所有检测器都实现 `BaseDetector` 接口：

```python
from microlens_defects.detection import BaseDetector, DetectionResult

class BaseDetector(ABC):
    def detect(self, stack: np.ndarray) -> DetectionResult:
        """执行检测"""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        pass
    
    @property
    def name(self) -> str:
        """检测器名称"""
        pass
```

**当前实现**：
- `ThresholdDetector`: 阈值法检测器

**未来计划**：
- `MLDetector`: 机器学习检测器 (DINOv3 + YOLO12)

#### 切换检测器示例

```python
# 当前使用阈值检测
from microlens_defects.detection import ThresholdDetector

detector = ThresholdDetector(params)
result = detector.detect(stack)

# 未来可无缝切换到ML检测器
# from microlens_defects.ml import MLDetector
# detector = MLDetector(model_path="model.pth")
# result = detector.detect(stack)  # 同样的接口！
```

### 五步相移法

```python
from microlens_defects.features.five_step_phase import (
    load_five_images,
    compute_phase,
    save_phase_result
)

# 加载5张图像
frames = load_five_images(
    folder=Path("./five_frames"),
    pattern="*.tif"
)

# 计算相位
result = compute_phase(frames, amp_threshold=1.0)

# 结果包含
# result.phase: 包裹相位 [-π, π]
# result.dc: DC图
# result.amplitude: 调制幅值
# result.mask: 有效区域掩码

# 保存
save_phase_result(Path("output.npz"), result)
```

---

## 测试

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_params.py -v
pytest tests/test_features.py::TestBuildFeatureBundle -v

# 带覆盖率
pytest tests/ -v --cov=microlens_defects
```

### 使用测试 fixtures

```python
import pytest
from microlens_defects.detection import ThresholdParams

def test_my_function(sample_stack, default_params):
    """使用共享的 fixtures"""
    # sample_stack: 100x100x28 模拟图像栈
    # default_params: ThresholdParams() 实例
    
    result = my_function(sample_stack, default_params)
    assert result is not None
```

可用 fixtures (定义在 `tests/conftest.py`):
- `sample_stack`: 100x100x28 图像栈
- `small_stack`: 50x50x5 小图像栈
- `sample_dc_map`: 100x100 DC 图
- `sample_mask`: 100x100 二值掩码
- `default_params`: 默认参数

---

## 开发指南

### 添加新的检测器

1. 继承 `BaseDetector`：

```python
from microlens_defects.detection import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    def __init__(self, my_param: float):
        self.my_param = my_param
    
    @property
    def name(self) -> str:
        return "MyDetector"
    
    def get_params(self) -> Dict[str, Any]:
        return {"my_param": self.my_param}
    
    def detect(self, stack: np.ndarray) -> DetectionResult:
        # 实现检测逻辑
        mask = self._process(stack)
        annotations = self._generate_annotations(mask)
        
        return DetectionResult(
            mask=mask,
            annotations=annotations,
            metadata={"detector": self.name}
        )
```

2. 添加测试：

```python
# tests/test_my_detector.py
def test_my_detector(sample_stack):
    detector = MyDetector(my_param=1.0)
    result = detector.detect(sample_stack)
    
    assert isinstance(result, DetectionResult)
    assert result.mask.shape == sample_stack.shape[:2]
```

### 代码风格

```bash
# Lint 检查
ruff check src/

# 类型检查
mypy src/microlens_defects --ignore-missing-imports

# 格式化
black src/
```

---

## 更多资源

- [README.md](../README.md): 快速开始
- [architecture.md](architecture.md): 系统架构
- [CHANGELOG.md](../CHANGELOG.md): 版本变更历史
- [troubleshooting.md](troubleshooting.md): 常见问题
