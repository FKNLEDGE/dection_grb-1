# EI论文写作完整项目说明与指南

> **用途**：将本文件导入新的Claude对话，再上传 `generate_paper_results.py` 生成的所有结果文件（outputs/paper/目录下的CSV、JSON、LaTeX、图片），即可让Claude帮你完成一篇合格的EI会议论文。

---

## 第一部分：项目完整技术说明

### 1.1 研究背景与目标

**课题**：基于迁移学习与注意力机制的智能垃圾分类系统

**核心贡献**：
1. 提出 MobileNetV2 + CBAM（通道+空间双注意力）的轻量级垃圾分类方案
2. 引入 Focal Loss 解决垃圾图像类别不平衡问题
3. 综合 Label Smoothing、Mixup/CutMix、EMA 等训练策略进一步提升性能
4. 通过完整的消融实验（9组配置）验证每个组件的边际贡献

**目标会议**：普通 EI 索引会议（非顶会），论文语言为英文。

---

### 1.2 数据集

| 项目 | 详情 |
|------|------|
| 名称 | Kaggle Garbage Classification Dataset |
| 来源 | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| 类别数 | 12 类 |
| 类别名 | battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash, white-glass, brown-glass |
| 划分比例 | Train:Val:Test = 8:1:1 |
| 输入尺寸 | 224 × 224 × 3 |
| 预处理 | Rescale 1/255；训练集做在线数据增强（旋转20°、平移0.2、翻转、缩放0.2、亮度[0.8,1.2]） |

---

### 1.3 模型架构

#### 1.3.1 总体架构（Proposed Method）

```
输入图像 (224×224×3)
    ↓
MobileNetV2 Backbone（ImageNet预训练，冻结）
    ↓ 特征图 (7×7×1280)
CBAM 注意力模块
    ├── Channel Attention（通道注意力）
    └── Spatial Attention（空间注意力）
    ↓ 加权特征图 (7×7×1280)
Global Average Pooling
    ↓ (1280,)
Dropout(0.5) → Dense(256, ReLU) → Dropout(0.25)
    ↓
Dense(12, Softmax)  →  分类输出
```

#### 1.3.2 MobileNetV2 Backbone

- 核心结构：Inverted Residual Block + Depthwise Separable Convolution
- 参数量：~3.4M（冻结，不参与训练）
- 预训练权重：ImageNet
- 优势：轻量高效，适合移动端/嵌入式部署

#### 1.3.3 CBAM（Convolutional Block Attention Module）

CBAM 由 Woo et al. (2018) 提出，包含两个顺序子模块：

**Channel Attention（通道注意力）——学习"关注什么特征"：**

$$M_c(F) = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))$$

- 对特征图 F ∈ R^{H×W×C} 分别做 Global Average Pooling 和 Global Max Pooling，得到两个 1×1×C 向量
- 通过共享的 MLP（两层全连接：C → C/r → C，r=16为压缩比）分别处理
- 将两个输出相加后经 Sigmoid 激活，得到通道注意力权重 M_c ∈ R^{1×1×C}
- 通过逐元素乘法应用到原始特征图：F' = M_c ⊗ F

**Spatial Attention（空间注意力）——学习"关注哪个位置"：**

$$M_s(F') = \sigma(f^{7\times7}([AvgPool(F'); MaxPool(F')]))$$

- 对通道加权后的特征图 F' 沿通道维度做 Average 和 Max 操作，得到两个 H×W×1 特征图
- 将两个特征图拼接得到 H×W×2 的张量
- 通过 7×7 卷积（filters=1）+ Sigmoid 激活，得到空间注意力权重 M_s ∈ R^{H×W×1}
- 应用：F'' = M_s ⊗ F'

**CBAM 在本项目中的位置**：after_encoder，即 MobileNetV2 编码器输出后、GAP 之前。

**与 SE-Net 的区别**：SE-Net (Hu et al., 2018) 仅有通道注意力（Squeeze-and-Excitation），无空间注意力。本文消融实验对比了 CBAM vs SE-Net，证明空间注意力的额外贡献。

#### 1.3.4 分类头设计

```python
# CBAM模型的分类头
Dropout(0.5)  →  Dense(256, ReLU)  →  Dropout(0.25)  →  Dense(12, Softmax)
```

- 使用两层 Dropout 防止过拟合
- 全连接层 256 单元作为特征映射
- Softmax 输出 12 类概率分布

---

### 1.4 损失函数

#### 1.4.1 Focal Loss

解决垃圾分类中类别不平衡问题。Lin et al. (2017) 提出：

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中：
- $p_t$ 是模型对真实类别的预测概率
- $\gamma = 2.0$（聚焦参数）：降低易分类样本的损失权重
- $\alpha = 0.25$（平衡参数）：平衡正负样本

当 $\gamma = 0$ 时退化为标准交叉熵。$\gamma$ 越大，对易分类样本的抑制越强。

#### 1.4.2 Label Smoothing

$$y_{smooth} = (1 - \epsilon) \cdot y_{one-hot} + \epsilon / K$$

- $\epsilon = 0.1$（平滑系数）
- $K = 12$（类别数）
- 防止模型对训练标签过度自信，提高泛化能力

---

### 1.5 训练策略

#### 1.5.1 学习率调度：Warmup + Cosine Annealing

```
阶段1 - Warmup（前5个epoch）: lr 从 1e-7 线性增长到 0.001
阶段2 - Cosine Decay（之后）: lr 从 0.001 按余弦函数衰减到 1e-7
```

$$lr(t) = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{t - t_{warmup}}{T - t_{warmup}} \pi))$$

#### 1.5.2 Mixup 数据增强

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

- $\lambda \sim Beta(\alpha, \alpha)$，$\alpha = 0.2$
- 训练时以50%概率选择 Mixup 或 CutMix

#### 1.5.3 CutMix 数据增强

将一个训练样本的矩形区域替换为另一个样本的对应区域：
- $\alpha = 1.0$（Beta 分布参数）
- 裁剪区域面积比例由 $\lambda$ 决定

#### 1.5.4 EMA（Exponential Moving Average）

$$\theta_{EMA} = \beta \cdot \theta_{EMA} + (1 - \beta) \cdot \theta$$

- $\beta = 0.999$（衰减率）
- 在验证和测试时使用 EMA 平滑后的参数，提升模型稳定性

#### 1.5.5 其他训练配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adam (lr=0.001) |
| Batch Size | 32 |
| Epochs | 30 |
| Early Stopping | patience=5（监控 val_loss） |
| 混合精度训练 | FP16（加速30-50%） |
| 随机种子 | 42 |

---

### 1.6 对比实验设计

#### 1.6.1 多模型对比（5个模型）

| 模型 | 参数量 | 特点 |
|------|--------|------|
| MobileNetV2 | ~3.4M | 轻量级，倒残差结构 |
| MobileNetV2_CBAM | ~3.4M+CBAM | 本文方案 |
| VGG16 | ~138M | 经典深层网络，3×3卷积堆叠 |
| DenseNet121 | ~8M | 密集连接，特征复用 |
| EfficientNetB0 | ~5.3M | 复合缩放策略，高效 |

所有模型统一使用：
- ImageNet 预训练 + 冻结 Backbone
- 相同的数据增强策略
- 相同的训练超参数（lr、epochs、batch_size等）
- 相同的数据划分

#### 1.6.2 消融实验（9组配置）

| 编号 | 配置名称 | 模型 | Focal Loss | Label Smoothing | Mixup/CutMix | EMA |
|------|----------|------|------------|----------------|--------------|-----|
| 1 | Baseline | MobileNetV2 | ✗ | ✗ | ✗ | ✗ |
| 2 | + CBAM | MobileNetV2_CBAM | ✗ | ✗ | ✗ | ✗ |
| 3 | + SE-Net | MobileNetV2_SE | ✗ | ✗ | ✗ | ✗ |
| 4 | + Focal Loss | MobileNetV2 | ✓ (γ=2,α=0.25) | ✗ | ✗ | ✗ |
| 5 | + Label Smoothing | MobileNetV2 | ✗ | ✓ (ε=0.1) | ✗ | ✗ |
| 6 | + Mixup/CutMix | MobileNetV2 | ✗ | ✗ | ✓ | ✗ |
| 7 | + EMA | MobileNetV2 | ✗ | ✗ | ✗ | ✓ (β=0.999) |
| 8 | + CBAM + Focal Loss | MobileNetV2_CBAM | ✓ | ✗ | ✗ | ✗ |
| 9 | Full Proposed | MobileNetV2_CBAM | ✓ | ✓ | ✓ | ✓ |

**消融设计逻辑**：
- 组1为纯 Baseline（无任何增强的 MobileNetV2）
- 组2-7为单因素消融，每组仅添加一个组件，与 Baseline 对比得到该组件的独立贡献
- 组8为双因素组合（CBAM + Focal Loss）
- 组9为完整方案（所有组件叠加）
- 组2 vs 组3：对比 CBAM（通道+空间）vs SE-Net（仅通道），验证空间注意力的价值

---

### 1.7 评估指标

所有指标均在独立测试集上计算（weighted average for multi-class）：

| 指标 | 公式/说明 |
|------|-----------|
| Accuracy | 正确预测数 / 总样本数 |
| Precision | TP / (TP + FP)，加权平均 |
| Recall | TP / (TP + FN)，加权平均 |
| F1-Score | 2 × P × R / (P + R)，加权平均 |
| Inference Time | 单张图片推理耗时（ms） |
| Model Size | 模型文件大小（MB） |
| Parameters | 模型总参数量 |

**Grad-CAM（Gradient-weighted Class Activation Mapping）**：
- 用于可解释性分析
- 对比 Baseline 和 CBAM 模型的类激活热力图
- 直观展示注意力机制如何改善模型对关键特征的关注

---

### 1.8 代码文件结构

```
dection_grb-1/
├── config.py                    # 全局配置（超参数、模型列表、消融配置）
├── data_loader.py               # 数据加载（划分、增强、类别权重）
├── models.py                    # 模型定义（MobileNetV2、CBAM、SE-Net、VGG16等）
├── trainer.py                   # 训练模块（Focal Loss、LR调度、EMA、Mixup等）
├── evaluation.py                # 评估可视化（混淆矩阵、Grad-CAM、消融图表）
├── ensemble.py                  # 模型集成（加权、投票、快照集成、TTA）
├── main.py                      # 主实验入口（CLI支持--ablation等）
├── generate_paper_results.py    # 一键论文结果生成脚本
├── download_data.py             # 数据集下载脚本
├── quick_test.py                # 环境快速测试
├── requirements.txt             # 依赖列表
└── outputs/paper/               # 论文输出目录
    ├── tables/                  # LaTeX表格
    │   ├── tab1_model_comparison.tex    # 表1：模型对比
    │   ├── tab2_ablation_results.tex    # 表2：消融实验
    │   └── tab3_per_class_metrics.tex   # 表3：分类别指标
    ├── figures/                 # 300 DPI 高清图片
    │   ├── fig1_model_comparison.png    # 模型性能对比四宫格
    │   ├── fig2_ablation_comparison.png # 消融对比柱状图
    │   ├── fig3_ablation_delta.png      # 增量贡献瀑布图
    │   ├── fig4_ablation_radar.png      # 多指标雷达图
    │   ├── fig5-9_confusion_matrix_*.png # 各模型混淆矩阵
    │   ├── fig10-14_training_curves_*.png # 训练曲线
    │   ├── gradcam_baseline/            # Baseline Grad-CAM
    │   ├── gradcam_cbam/                # CBAM Grad-CAM
    │   └── fig16_per_class_metrics.png  # 分类别指标图
    ├── data/                    # 原始数据
    │   ├── model_comparison.csv
    │   ├── ablation_results.csv
    │   └── ablation_results.json
    └── PAPER_FIGURES_GUIDE.txt  # 图表→论文章节映射
```

---

## 第二部分：论文写作详细指南

### 2.1 论文推荐结构

EI 会议论文推荐 6-8 页，结构如下：

```
Title
Abstract (150-250 words)
1. Introduction (~1 page)
2. Related Work (~0.5-1 page)
3. Proposed Method (~1.5-2 pages)
   3.1 Overall Architecture
   3.2 CBAM Attention Mechanism
   3.3 Focal Loss
   3.4 Training Strategy
4. Experiments (~2-3 pages)
   4.1 Dataset and Implementation Details
   4.2 Model Comparison
   4.3 Ablation Study
   4.4 Per-class Analysis
   4.5 Interpretability Analysis (Grad-CAM)
5. Conclusion (~0.5 page)
References (15-20 references)
```

---

### 2.2 各章节写作指导（带模板）

#### 2.2.1 Title

建议格式：方法 + 应用领域

**推荐标题**：
- "An Intelligent Waste Classification Method Based on MobileNetV2 with CBAM Attention Mechanism and Focal Loss"
- "Lightweight Garbage Classification Using Transfer Learning with Convolutional Block Attention Module"

#### 2.2.2 Abstract

**结构**：Background (1-2句) → Problem (1句) → Method (2-3句) → Results (2-3句) → Conclusion (1句)

**模板**（请根据实际结果填入数据）：

> Waste classification is essential for environmental protection and sustainable resource management. However, the visual similarity between different waste categories poses significant challenges for automated classification systems. In this paper, we propose an intelligent waste classification method based on MobileNetV2 integrated with Convolutional Block Attention Module (CBAM) and Focal Loss. The CBAM module enhances the model's ability to focus on discriminative features through both channel and spatial attention mechanisms. Focal Loss addresses the class imbalance problem by down-weighting easy samples during training. We further incorporate Label Smoothing, Mixup/CutMix data augmentation, and Exponential Moving Average (EMA) to improve generalization. Extensive experiments on a 12-class garbage classification dataset demonstrate that our proposed method achieves an accuracy of [XX.XX]% and F1-score of [XX.XX]%, outperforming the baseline MobileNetV2 by [+X.XX]%. Comprehensive ablation studies validate the contribution of each component. The proposed lightweight architecture is suitable for deployment on mobile and embedded devices for real-world waste sorting applications.

#### 2.2.3 Introduction（4段结构）

**第1段——大背景**：
- 垃圾分类的社会重要性（环保、资源回收、政策驱动）
- 传统人工分类的局限性
- 计算机视觉/深度学习自动分类的前景

**第2段——已有方法及不足**：
- 早期方法：传统机器学习（SVM、RF + 手工特征如 HOG、SIFT）→ 特征工程困难，泛化差
- CNN 方法：VGG、ResNet 等大模型 → 参数量大、推理慢、不适合移动端
- 轻量级模型：MobileNet 系列 → 精度有下降空间
- 现有不足：缺乏注意力机制引导特征学习、未充分处理类别不平衡

**第3段——本文方法概述**：
- 提出 MobileNetV2 + CBAM + Focal Loss 的方法
- CBAM 的双注意力机制增强特征表示
- Focal Loss 解决类别不平衡
- 额外训练策略（Label Smoothing, Mixup/CutMix, EMA）
- 在 12 类垃圾分类数据集上验证

**第4段——贡献总结**：

> The main contributions of this paper are summarized as follows:
> 1. We propose a lightweight garbage classification method that integrates MobileNetV2 with CBAM attention module, achieving [state accuracy] with only [X]M trainable parameters.
> 2. We employ Focal Loss combined with Label Smoothing and advanced data augmentation strategies to address class imbalance and improve generalization.
> 3. We conduct comprehensive ablation experiments with 9 configurations to validate the individual and combined contributions of each component.
> 4. We provide Grad-CAM visualization analysis to demonstrate the interpretability improvement brought by the attention mechanism.

#### 2.2.4 Related Work

**建议分3个小节**：

**(a) Transfer Learning for Image Classification**
- 介绍迁移学习的概念和优势
- 引用 MobileNetV2 (Sandler et al., 2018)、VGG (Simonyan and Zisserman, 2015)、ResNet (He et al., 2016)、DenseNet (Huang et al., 2017)、EfficientNet (Tan and Le, 2019)
- 说明 ImageNet 预训练 + 冻结 backbone + 训练分类头的范式

**(b) Attention Mechanisms in CNNs**
- SE-Net (Hu et al., 2018): Squeeze-and-Excitation，仅通道注意力
- CBAM (Woo et al., 2018): 通道+空间双注意力，计算开销小但效果显著
- 其他注意力：ECA-Net, BAM 等可简要提及
- 说明注意力在细粒度分类中的价值

**(c) Waste/Garbage Classification**
- 列举近年垃圾分类相关工作
- 指出现有方法的不足（模型过重、未用注意力、忽略类别不平衡）
- 引出本文的改进方向

#### 2.2.5 Proposed Method

**§3.1 Overall Architecture**
- 画一个整体架构图（建议用 draw.io 或 PPT 画，包含：Input → MobileNetV2 → CBAM → GAP → FC → Output）
- 简要描述信息流

**§3.2 CBAM Attention Mechanism**
- 分别写 Channel Attention 和 Spatial Attention 的公式（见 1.3.3 节）
- 配一个 CBAM 模块内部结构图
- 解释为什么先通道再空间（原论文实验验证的最优顺序）
- 说明 reduction ratio r=16 的选择

**§3.3 Focal Loss**
- 写出公式（见 1.4.1 节）
- 解释 γ 和 α 的物理意义
- 说明与标准交叉熵的关系（γ=0 时退化为 CE）
- 可画一个 Focal Loss vs CE 的对比图（不同 γ 值下的 loss 曲线）

**§3.4 Training Strategy**
- Warmup + Cosine Annealing 学习率调度（可画学习率曲线图）
- Label Smoothing（公式 + 作用说明）
- Mixup/CutMix 数据增强（公式 + 可视化示例）
- EMA（公式 + 作用说明）

#### 2.2.6 Experiments

**§4.1 Dataset and Implementation Details**

> 必写内容清单：
> - 数据集名称、来源、类别数、总样本数
> - 数据划分比例（8:1:1）
> - 输入尺寸（224×224）
> - 硬件环境（GPU型号、显存）
> - 软件环境（Python版本、TensorFlow版本）
> - 训练超参数（lr=0.001, batch=32, epochs=30, optimizer=Adam）
> - 数据增强配置
> - 评估指标（Accuracy, Precision, Recall, F1, weighted average）

**§4.2 Model Comparison**（使用 tab1 和 fig1）

写作要点：
1. 先概述实验设计（5个模型，统一配置，公平对比）
2. 放表1（tab1_model_comparison.tex）
3. 分析结果：
   - 哪个模型最优？为什么？
   - MobileNetV2_CBAM vs MobileNetV2：CBAM 带来了多少提升？
   - MobileNetV2_CBAM vs VGG16/DenseNet121/EfficientNetB0：在更轻量的参数量下性能如何？
   - 讨论精度-效率 trade-off
4. 放图1（fig1_model_comparison.png），描述四宫格图的各个方面

**§4.3 Ablation Study**（使用 tab2, fig2, fig3, fig4）

这是本文最重要的实验部分！写作要点：
1. 说明消融实验的设计原则（逐个添加组件，验证边际贡献）
2. 放表2（tab2_ablation_results.tex）
3. 逐项分析各组件的贡献：
   - CBAM vs Baseline: 注意力机制带来的提升
   - CBAM vs SE-Net: 空间注意力的额外价值
   - Focal Loss: 对类别不平衡的改善
   - Label Smoothing: 正则化效果
   - Mixup/CutMix: 数据增强效果
   - EMA: 训练稳定性提升
   - CBAM + Focal Loss（组合效果）
   - Full Proposed vs Baseline: 总体提升
4. 放图2-4，分别描述：
   - fig2: 各配置的 Accuracy 和 F1 对比
   - fig3: 各组件的增量贡献（ΔAccuracy）
   - fig4: 雷达图展示多维度指标

**§4.4 Per-class Analysis**（使用 tab3 和 fig16）

1. 分析模型在 12 个类别上的具体表现
2. 指出表现最好和最差的类别，分析原因（视觉相似性、样本量等）
3. 特别讨论 glass vs white-glass vs brown-glass（视觉相似类别）
4. 放表3 和 fig16

**§4.5 Interpretability Analysis**（使用 Grad-CAM 图）

1. 解释 Grad-CAM 的原理（1-2句）
2. 对比 Baseline 和 CBAM 模型的 Grad-CAM 热力图
3. 说明 CBAM 如何帮助模型关注更有判别力的区域
4. 选取2-3个典型样例进行分析

#### 2.2.7 Conclusion

**模板**：

> In this paper, we proposed a lightweight garbage classification method based on MobileNetV2 with CBAM attention mechanism and Focal Loss. The CBAM module effectively enhanced feature representation through channel and spatial attention, while Focal Loss addressed the class imbalance problem. Combined with Label Smoothing, Mixup/CutMix, and EMA strategies, our method achieved [XX.XX]% accuracy on a 12-class waste classification dataset, improving [+X.XX]% over the baseline. The ablation study confirmed that each component contributed positively, with CBAM and Focal Loss being the most significant contributors. Grad-CAM visualization further demonstrated the improved interpretability of the attention-enhanced model. Future work will explore deploying the model on mobile devices and extending to more waste categories.

---

### 2.3 参考文献列表（IEEE格式，直接使用）

```bibtex
% 核心引用 - 必须引用
@inproceedings{sandler2018mobilenetv2,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={ECCV},
  year={2018}
}

@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{hu2018senet,
  title={Squeeze-and-Excitation Networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={CVPR},
  year={2018}
}

% Backbone 模型
@inproceedings{simonyan2015vgg,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  booktitle={ICLR},
  year={2015}
}

@inproceedings{he2016resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{huang2017densenet,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={CVPR},
  year={2017}
}

@inproceedings{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={ICML},
  year={2019}
}

% 训练技术
@article{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David},
  journal={ICLR},
  year={2018}
}

@inproceedings{yun2019cutmix,
  title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
  author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
  booktitle={ICCV},
  year={2019}
}

@article{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  journal={CVPR},
  year={2016},
  note={Label Smoothing 首次提出}
}

@inproceedings{loshchilov2017sgdr,
  title={SGDR: Stochastic Gradient Descent with Warm Restarts},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2017},
  note={Cosine Annealing 学习率调度}
}

% 可解释性
@inproceedings{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Convolutional Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={ICCV},
  year={2017}
}

% 迁移学习
@article{deng2009imagenet,
  title={ImageNet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  journal={CVPR},
  year={2009}
}

% 垃圾分类相关（按需选用，建议搜索最新的引用）
@article{yang2016classification,
  title={Classification of Trash for Recyclability Status},
  author={Yang, Mindy and Thung, Gary},
  journal={CS229 Project Report},
  year={2016}
}
```

---

### 2.4 论文图表与论文章节的完整映射

| 论文位置 | 图表编号 | 文件名 | 内容说明 |
|----------|----------|--------|----------|
| §3.1 | 图1 (自制) | 需自己画 | 整体网络架构图 |
| §3.2 | 图2 (自制) | 需自己画 | CBAM 模块内部结构图 |
| §4.2 | 表1 | tab1_model_comparison.tex | 5个模型的性能对比 |
| §4.2 | 图3 | fig1_model_comparison.png | 模型对比四宫格可视化 |
| §4.3 | 表2 | tab2_ablation_results.tex | 9组消融实验结果 |
| §4.3 | 图4 | fig2_ablation_comparison.png | 消融对比柱状图 |
| §4.3 | 图5 | fig3_ablation_delta.png | 增量贡献瀑布图 |
| §4.3 | 图6 | fig4_ablation_radar.png | 多指标雷达图 |
| §4.4 | 表3 | tab3_per_class_metrics.tex | 12类别详细指标 |
| §4.4 | 图7 | fig16_per_class_metrics.png | 分类别P/R/F1条形图 |
| §4.4 | 图8-9 | fig5-9_confusion_matrix_*.png | 关键模型混淆矩阵（选2-3个） |
| §4.5 | 图10 | gradcam_baseline/ + gradcam_cbam/ | Grad-CAM 对比图 |
| 附录(可选) | 图A1-A5 | fig10-14_training_curves_*.png | 训练/验证曲线 |

---

### 2.5 写作注意事项

#### EI 论文基本要求
1. **语言**：学术英语，避免口语化表达，使用被动语态为主
2. **篇幅**：6-8 页（含参考文献）
3. **图片**：300 DPI，清晰可读（本项目已设置好）
4. **表格**：使用三线表（booktabs），本项目 LaTeX 表格已是 booktabs 格式
5. **数据精度**：保留4位小数（如 0.9234）
6. **引用格式**：IEEE 格式 [1], [2], [3]
7. **公式编号**：所有重要公式需编号

#### 常见错误避免
| 错误 | 正确做法 |
|------|----------|
| "We use CBAM because it is good" | "We integrate CBAM to enhance feature representation through channel and spatial attention" |
| 只报告 Accuracy | 同时报告 Accuracy, Precision, Recall, F1 |
| 消融实验不完整（只对比2-3组） | 完整9组消融，逐个验证每个组件 |
| 没有统计显著性讨论 | 提及消融实验中各组件提升的幅度 |
| 缺少 Limitation | 在 Conclusion 中简要提及局限性和未来工作 |
| 图表不引用 | 正文中必须引用每个图表："As shown in Table 1...", "Fig. 2 illustrates..." |

#### 论文中结果描述的标准表达

**描述精度提升**：
- "The proposed method achieves an accuracy of 95.23%, which is 3.15% higher than the baseline."
- "CBAM brings a notable improvement of +2.1% in accuracy over the vanilla MobileNetV2."

**描述消融结果**：
- "As shown in Table 2, each component contributes positively to the overall performance."
- "The most significant improvement comes from CBAM (+X.XX%), followed by Focal Loss (+X.XX%)."
- "The combination of all components (Full Proposed) achieves the best result of XX.XX%."

**描述 Grad-CAM**：
- "As illustrated in Fig. 10, the CBAM-enhanced model focuses on more discriminative regions of the waste objects compared to the baseline model."
- "The attention maps show that CBAM effectively guides the model to attend to the object of interest rather than background clutter."

---

## 第三部分：给新Claude的使用指令

当你将本文件导入新的 Claude 对话时，请附上以下指令：

---

**请求模板**：

> 我正在写一篇 EI 会议论文，主题是基于迁移学习与注意力机制的智能垃圾分类。
>
> 我已经附上了：
> 1. 项目完整说明文件（EI_PAPER_COMPLETE_BRIEF.md）——包含所有技术细节、模型架构、训练策略、实验设计
> 2. 实验结果数据文件（来自 outputs/paper/ 目录）：
>    - model_comparison.csv：5个模型对比的原始数据
>    - ablation_results.csv / ablation_results.json：9组消融实验结果
>    - tab1_model_comparison.tex：模型对比 LaTeX 表格
>    - tab2_ablation_results.tex：消融实验 LaTeX 表格
>    - tab3_per_class_metrics.tex：分类别指标 LaTeX 表格
>    - 各种 figures（模型对比图、消融图、Grad-CAM图等）
>
> 请根据项目说明和实际实验结果，帮我写一篇完整的 EI 会议论文。
> 要求：
> - 英文学术写作
> - 6-8 页
> - IEEE 格式
> - 所有数据使用实际实验结果（不要编造数据）
> - 包含完整的 Abstract, Introduction, Related Work, Proposed Method, Experiments, Conclusion
> - LaTeX 格式输出

---

## 附录：关键代码实现细节（供论文 Methods 部分参考）

### A.1 CBAM 实现核心代码

```python
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16):
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = GlobalMaxPooling2D(keepdims=True)
        # 共享 MLP: Dense(C/r, relu) → Dense(C)

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.global_avg_pool(inputs)))
        max_out = self.fc2(self.fc1(self.global_max_pool(inputs)))
        attention = sigmoid(avg_out + max_out)  # [1,1,C]
        return inputs * attention

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = mean(inputs, axis=-1, keepdims=True)  # [H,W,1]
        max_pool = max(inputs, axis=-1, keepdims=True)   # [H,W,1]
        attention = self.conv(concat([avg_pool, max_pool]))  # [H,W,1]
        return inputs * attention

class CBAM(layers.Layer):
    def call(self, inputs):
        x = ChannelAttention()(inputs)   # 先通道
        x = SpatialAttention()(x)        # 后空间
        return x
```

### A.2 Focal Loss 实现

```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = clip(y_pred, 1e-7, 1-1e-7)
        ce = -y_true * log(y_pred)
        weight = self.alpha * y_true * (1 - y_pred) ** self.gamma
        return sum(weight * ce, axis=-1)
```

### A.3 模型构建（Proposed Method）

```python
def build_mobilenetv2_cbam():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False  # 冻结 backbone

    inputs = Input(shape=(224,224,3))
    x = base(inputs)                              # [7,7,1280]
    x = CBAM(reduction_ratio=16, kernel_size=7)(x) # 注意力加权
    x = GlobalAveragePooling2D()(x)                # [1280]
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(12, activation='softmax')(x)
    return Model(inputs, outputs)
```
