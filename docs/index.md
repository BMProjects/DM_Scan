# 文档总览 (v0.3.0)

- **快速使用**：见仓库根目录 `README.md`（安装与 CLI 示例）。
- **API 使用指南**：`api_guide.md` —— Python API、模块说明、开发指南（v0.3.0 新增）。
- **版本变更**：`../CHANGELOG.md` —— 详细记录各版本新增、变更和修复内容。
- **架构概览**：`architecture.md` —— 组件、数据流、功能分层（v0.3.0 已更新模块化结构）。
- **流水线与规范**：`pipeline.md` —— 输入输出、目录/配置约定、阈值基线 CLI 用法。
- **测试大纲**：`test_plan.md` —— 评测指标、流程与冻结测试集策略。
- **开发/评测 SOP**：`dev_eval_sop.md` —— 环境、数据准备、ML/半监督计划、评测与回流流程。
- **迭代/清理记录**：`legacy_cleanup.md` —— 移除/保留的历史脚本与取回方式。
- **算法测试计划（旧版）**：`algorithm_test_plan.md` —— 早期检测方案的测试规划。
- **项目背景总结**：`project_summary.md` —— 目标与系统设计简述。
- **年度进展/待办**：`annual_progress_2025.md`，`development_todolist.md`。
- **交接说明**：`handover_defect_detection.md`。
- **常见问题与排障**：`troubleshooting.md` —— 安装/数据/运行常见错误处理。

## v0.3.0 主要改进

- ✅ 模块化重构：`detection/` 拆分为 6 个专注模块
- ✅ 抽象接口：`BaseDetector` 为 ML 集成预留扩展
- ✅ 日志框架：统一 `get_logger()` 替代 `print()`
- ✅ 异常层次：自定义异常便于错误分类
- ✅ 单元测试：新增 5 个测试模块 + pytest fixtures
- ✅ CI/CD：GitHub Actions 自动化 lint + type check + test

