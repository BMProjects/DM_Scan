# 离焦微结构镜片缺陷智能检测算法测试计划

## 1. 测试目标

定量评估AI缺陷检测模型在独立测试集上的性能、泛化能力和局限性，确保其准确性和可靠性，并为后续的模型迭代提供明确的优化方向。

## 2. 测试数据集制作

*   **数据源:** 从已标注的数据中划分出一个独立的、**从未用于训练或验证**的**测试集 (Hold-out Test Set)**。
*   **规模:** 建议至少包含 **50-100 张图像**。
*   **构成:**
    *   应覆盖所有已定义的缺陷类型，并尽可能均衡。
    *   包含一些“挑战性样本”，如：微小缺陷、边缘缺陷、多缺陷重叠图像、以及无缺陷的正常图像（用于评估误报率）。
*   **黄金标准:** 对测试集中的所有标注进行二次审核，确保其标注质量为“黄金标准 (Golden Standard)”，作为评估的绝对基准。

## 3. 测试大纲 (测试指标)

*   **A. 分割精度测试 (Segmentation Accuracy):**
    *   **IoU (Intersection over Union):** 像素级重合度的核心指标，计算每个缺陷实例的 `预测掩码` 与 `真实掩码` 的交并比。
    *   **Dice Coefficient:** 与 IoU 类似，同样衡量掩码的重合度。
    *   **mIoU (mean IoU):** 计算所有缺陷类别 IoU 的平均值，是评估模型整体分割性能的关键。
*   **B. 分类与检测精度测试 (Classification & Detection Accuracy):**
    *   **Precision (精确率):** 预测为正的样本中，有多少是真正的正样本。
    *   **Recall (召回率):** 所有真正的正样本中，有多少被成功预测。
    *   **F1-Score:** 精确率和召回率的调和平均数。
    *   **混淆矩阵 (Confusion Matrix):** 直观展示模型对不同缺陷类别的分类正确与错误情况。
*   **C. 鲁棒性/边界测试 (Robustness/Boundary Cases):**
    *   分析模型在微小缺陷、边缘缺陷、低对比度缺陷等困难样本上的表现。
    *   评估模型在正常（无缺陷）图像上的表现，统计**误报率 (False Positive Rate)**。
*   **D. 性能测试 (Performance - 记录项):**
    *   记录模型处理单张图像所需的平均推理时间。

## 4. 测试步骤规划

1.  **准备阶段:**
    *   **步骤 1:** 根据上述要求，正式划分并最终确认测试数据集及其“黄金标准”标注。
2.  **执行阶段:**
    *   **步骤 2:** 加载最终训练好的AI模型。
    *   **步骤 3:** 编写测试脚本，遍历测试集中的每一张图像，执行模型推理，并保存预测结果（预测的分割掩码、缺陷类别、置信度）。
3.  **评估阶段:**
    *   **步骤 4:** 编写评估脚本，将模型的预测结果与“黄金标准”标注进行逐一对比。
    *   **步骤 5:** 自动计算测试大纲中定义的所有量化指标（mIoU, F1-Score 等）。
    *   **步骤 6:** 生成并可视化混淆矩阵。
4.  **分析与报告阶段:**
    *   **步骤 7:** 分析测试结果，特别是识别失败的案例（False Positives/Negatives），总结模型的优势和明显弱点。
    *   **步骤 8:** 撰写正式的《算法测试报告》，内容包括：测试环境、数据集描述、量化评估结果、失败案例分析以及下一步模型优化的具体建议。


在改成“先导出特征再检测”之前，整套流程就是直接在 29 帧相位图上做频域拟合+空间/时间异常检测。实现集中在 detect_defects_29stack.py，核心步骤可以这样拆解（便于留档参考）：

读取与预处理

将某个镜片/光栅的 29 张帧图按文件名排序读入，堆成 stack (H×W×N)，通常 N=29。
若图像是 16bit/彩色，先转 float32 灰度。
时域拟合（一次傅里叶项）

对每个像素序列计算 DC = mean(stack, axis=2)。
求 DFT k=1 的复系数 C1 = Σ I_n·e^{-i2πn/N}，再得振幅 A = 2|C1|/N。
重建拟合序列 fitted = DC + (2/N)·Re(C1·e^{i2πn/N})，即假设条纹是完美正弦信号。
残差与标准差

residuals = stack - fitted；按帧方向求 residual_std = std(residuals, axis=2)。
另算原始序列的 temporal_std = std(stack, axis=2) 供后续参考（旧版主要用 residual_std）。
两个互补的异常掩码

静态 DC 掩码 static_mask：
对 DC 图做局部中值滤波（size≈15）。
diff = |DC - local_median|，用稳健阈值 th = median(diff) + k·MAD（默认 k=6）分割；
得到的是“空间亮度异常”——随相位不变的暗/亮缺陷（划痕、污点）。
残差掩码 residual_mask：
residual_std 反映正弦拟合的失真；
同样用 median + k·MAD 做阈值，挑出“随相位频谱不匹配”的区域（镀膜破坏、时域噪声）。
掩码合并与清理

combined = static_mask OR residual_mask，再调用 clean_mask:
ndi.label 计算连通域；
把面积 < min_size（默认 50px）的区域丢弃；
binary_closing + binary_fill_holes，形成干净的二值 mask。
输出

mask.png = cleaned * 255；
overlay = overlay_mask_on_image(DC, cleaned) 将掩码以红色覆盖在 DC 图上。
这一整套流程就是 process_lens_defects.py 在“直接 29 帧检测”模式下调用的算法（run_detection_for_combination 里手动重复了上述步骤）。
总结：旧算法 = DC 局部异常 + 拟合残差异常 → 形态学后处理；没有先导出特征文件，直接对 29 帧 stack 做计算。现在的特征导出（DC / residual_std / temporal_std）就是把这些中间产物“持久化”，以便在 defect_interactive.py 里做更复杂的多通道分析。