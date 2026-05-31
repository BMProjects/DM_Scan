# 常见问题与故障排查

## 环境/安装
- **pip 过慢或失败**：尝试国内镜像（如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。确认 Python ≥3.9。
- **opencv/科学计算依赖冲突**：建议使用虚拟环境（`python -m venv .venv` 或 `uv venv`），再 `pip install -e .`。

## 数据与路径
- **提示数据库/图片根目录不存在**：检查 `--db`、`--img-root` 路径；确保 SQLite 文件可读、图像目录存在。
- **file_path 无法解析**：`images.file_path` 优先在 `--img-root` 下解析，若存绝对路径请保证文件可访问。缺帧会被跳过并在日志中 WARN。
- **帧数不足**：阈值法默认需要前 28 帧；数据库中帧数不足会被过滤。使用 `--num-frames` 可调整。

## 运行与输出
- **批量参数冲突**：`--all` 不能与 `--glasses/--side/--grating` 同时使用；仅传 `--limit` 时自动启用 `--all`。
- **输出目录权限**：`--save-dir` 默认当前目录下的 `defect_detection_outputs`，若无写权限请改到可写路径。
- **结果为空/缺陷过少**：调低 `prominence_min_value`、放宽 `scratch_min_len/scratch_min_aspect`，或减小 `density_threshold`。调整后建议先用 `--limit` 小批量验证。
- **crash 过多误检**：提高 `density_threshold` 或 `dense_min_area`，或减小 `density_kernel_ratio`。

## 五步相移法
- **输入不足 5 张图**：`phase5` 需要至少 5 张匹配 `--pattern` 的图像；文件名排序后取前 5 张。
- **相位异常/掩码为空**：检查曝光/对比度，适当降低 `--amp-threshold`。

## 参考
- 完整用法与参数说明：`README.md`、`docs/architecture.md`
- API 与二次开发：`docs/api_guide.md`
