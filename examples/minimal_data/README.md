# Minimal Example

用于快速验证安装与 CLI 的最小示例结构（不含真实图像，仅示意）：

- `sample_images.csv`：示例的元数据表头与 5 帧记录格式，可导入/转换为 SQLite。
- `sample_frames/`：请放置至少 5 张匹配 CSV 的灰度图（文件名与 `file_path` 一致）。

运行思路：
1) 准备一个最小 SQLite（或直接扩展 CSV -> SQLite），包含表 `images(glasses_id, lens_side, grating_type, phase_index, file_path)`。
2) 将 `sample_frames/` 路径作为 `--img-root`，确保 `file_path` 可解析。
3) 运行阈值法：
```bash
microlens-defects detect --glasses 0001 --side left --grating cycle \
  --db microlens_metadata.db --img-root sample_frames --save-dir outputs_demo 
```
4) 或运行五步相移法（若只准备 5 张条纹图）：
```bash
microlens-defects phase5 sample_frames --pattern "*.tif" --output phase_result.npz
```

可将真实小样本替换进来，便于 CI 或环境自检。
