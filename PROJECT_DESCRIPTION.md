# 项目完整说明文档：基于迁移学习与MobileNetV2的智能垃圾分类系统

> **文档用途**：本文档旨在让 Claude 或其他 AI 助手在实验结果产出后，能够完全、细致地了解本项目的研究背景、技术方案、代码架构、实验设计及各模块的实现细节，从而精准地给出论文写作建议。

---

## 1. 项目概述

### 1.1 研究主题

本项目是一个**基于深度学习迁移学习的智能垃圾分类系统**（Intelligent Waste Classification System Based on Transfer Learning and MobileNetV2），核心任务是将垃圾图像自动分类到 12 个类别中。

### 1.2 研究动机与背景

- 城市垃圾分类是环保领域的重要课题，人工分类效率低、成本高
- 深度学习在图像分类领域取得了巨大突破，但大型模型难以部署到嵌入式设备
- 迁移学习可以利用 ImageNet 预训练权重，在小数据集上快速收敛
- MobileNetV2 作为轻量级网络，适合边缘部署场景

### 1.3 研究目标

1. **对比分析**多种预训练 CNN 模型在垃圾分类任务上的性能表现
2. **提出改进方案**：在轻量级网络 MobileNetV2 上集成 CBAM 注意力机制
3. **引入 Focal Loss** 解决垃圾分类中的类别不平衡问题
4. **通过 Grad-CAM** 提供模型可解释性分析
5. **消融实验**验证各改进组件的独立贡献

### 1.4 论文目标定位

区域性 EI 会议发表水平。

---

## 2. 数据集

### 2.1 数据来源

- **数据集名称**：Garbage Classification Dataset
- **来源**：Kaggle（作者：mostafaabla）
- **链接**：`https://www.kaggle.com/datasets/mostafaabla/garbage-classification`
- **下载方式**：Kaggle API 或手动下载（项目提供 `download_data.py` 脚本自动化）

### 2.2 类别定义（12 类）

| 序号 | 英文名称 | 中文名称 | 所属大类 |
|------|----------|----------|----------|
| 1 | battery | 电池 | 有害垃圾 |
| 2 | biological | 生物垃圾 | 湿垃圾/厨余 |
| 3 | cardboard | 纸板 | 可回收物 |
| 4 | clothes | 衣物 | 可回收物 |
| 5 | glass | 玻璃 | 可回收物 |
| 6 | metal | 金属 | 可回收物 |
| 7 | paper | 纸张 | 可回收物 |
| 8 | plastic | 塑料 | 可回收物 |
| 9 | shoes | 鞋子 | 可回收物 |
| 10 | trash | 其他垃圾 | 干垃圾/其他 |
| 11 | white-glass | 白色玻璃 | 可回收物 |
| 12 | brown-glass | 棕色玻璃 | 可回收物 |

### 2.3 数据集划分策略

- **训练集 : 验证集 : 测试集 = 8 : 1 : 1**
- 按类别分层随机划分（Stratified Split），保证各划分集中类别比例一致
- 随机种子固定为 `42`，确保可复现性
- 划分通过 `data_loader.py` 中的 `create_data_split()` 函数实现，会将文件物理复制到 `data/garbage_classification/split/{train,val,test}/` 目录

### 2.4 数据预处理

- **图像尺寸归一化**：所有图像统一缩放至 `224 x 224 x 3`（适配 MobileNetV2/ResNet50/VGG16 等模型的标准输入）
- **像素归一化**：`rescale=1./255`，将像素值从 [0, 255] 映射到 [0, 1]
- **图像有效性验证**：可选启用，使用 PIL 的 `Image.verify()` 检测损坏图像

### 2.5 数据增强策略

仅对训练集应用数据增强（验证集和测试集只做归一化）：

| 增强方式 | 参数 | 说明 |
|----------|------|------|
| 旋转 (Rotation) | ±20° | 模拟拍摄角度变化 |
| 水平平移 (Width Shift) | ±20% | 模拟水平位置变化 |
| 垂直平移 (Height Shift) | ±20% | 模拟垂直位置变化 |
| 水平翻转 (Horizontal Flip) | True | 增加左右对称样本 |
| 缩放 (Zoom) | ±20% | 模拟远近距离变化 |
| 亮度调整 (Brightness) | [0.8, 1.2] | 模拟不同光照条件 |
| 填充模式 (Fill Mode) | nearest | 变换后的空白区域填充 |

此外还实现了高级数据增强：

- **Mixup**（Zhang et al., 2017）：α=0.2，线性插值混合两个样本
- **CutMix**（Yun et al., 2019）：α=1.0，裁剪粘贴矩形区域
- 两者以 50% 概率随机选择一种应用

### 2.6 类别权重计算

使用 `get_class_weights()` 函数自动计算类别权重，公式为：

```
weight_i = total_samples / (num_classes × count_i)
```

该权重在训练时传入 `model.fit()` 的 `class_weight` 参数，使模型更关注少数类。

---

## 3. 模型架构

### 3.1 迁移学习框架

所有模型采用统一的迁移学习范式：

```
ImageNet 预训练基模型 (feature extractor, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dropout(rate)
        ↓
Dense(units, activation='relu')
        ↓
Dropout(rate/2)
        ↓
Dense(12, activation='softmax')  ← 输出 12 类概率
```

- 基模型权重从 ImageNet 加载，默认冻结（`FREEZE_BASE=True`），仅训练分类头
- 支持通过 `unfreeze_base_model()` 解冻最后 N 层进行微调

### 3.2 对比模型一览

| 模型 | 参数量 | 估计大小 | 核心架构特点 | 分类头配置 |
|------|--------|----------|-------------|-----------|
| **MobileNetV2** | ~3.4M | ~14 MB | 倒残差结构 + 深度可分离卷积 | Dense(128) |
| **MobileNetV2_CBAM** | ~3.5M | ~14 MB | MobileNetV2 + CBAM 注意力 | Dense(256) |
| **VGG16** | ~138M | ~528 MB | 3x3 卷积堆叠，16 层 | Dense(512)+Dense(256) |
| **DenseNet121** | ~8M | ~30 MB | 密集连接，特征复用 | Dense(256) |
| **EfficientNetB0** | ~5.3M | ~20 MB | 复合缩放策略（NAS搜索） | Dense(128) |
| **ResNet50** | ~25.6M | ~98 MB | 残差连接，50 层 | Dense(256) |
| **MobileNetV2_SE** | ~3.5M | ~14 MB | MobileNetV2 + SE-Net 注意力 | Dense(256) |
| **Simple_CNN** | 可变 | 小 | 4 层简单 CNN（基线对照） | Dense(512) |

> 实际默认运行的对比模型为：MobileNetV2, MobileNetV2_CBAM, VGG16, DenseNet121, EfficientNetB0（共 5 个）

### 3.3 注意力机制详解

#### 3.3.1 CBAM（Convolutional Block Attention Module）

CBAM 是本项目的**核心技术创新点**之一，由 Woo et al. (2018) 提出，它将注意力分为两个子模块顺序应用：

**通道注意力（Channel Attention）：**
- 输入特征图 F ∈ R^{H×W×C}
- 分别进行全局平均池化和全局最大池化 → 得到两个 1×1×C 的向量
- 通过共享的两层 MLP（压缩比 r=16）：Dense(C/r, ReLU) → Dense(C)
- 两路输出相加后 Sigmoid → 通道注意力权重 M_c ∈ R^{1×1×C}
- F' = F ⊗ M_c（逐元素相乘）

**空间注意力（Spatial Attention）：**
- 输入 F' ∈ R^{H×W×C}
- 沿通道维度分别计算平均值和最大值 → 两个 H×W×1 的特征图
- 拼接后通过 7×7 卷积 → Sigmoid → 空间注意力权重 M_s ∈ R^{H×W×1}
- F'' = F' ⊗ M_s

**实现位置选项：**
- `after_encoder`（默认）：在基模型输出后添加 CBAM
- `before_gap`：在全局平均池化前添加
- `after_gap`：仅使用通道注意力（空间维度已被压缩）
- `multi_scale`：在 MobileNetV2 的 block_6、block_13 和最终输出三个尺度分别添加 CBAM，然后拼接多尺度特征

#### 3.3.2 SE-Net（Squeeze-and-Excitation）

- **Squeeze**：全局平均池化将 H×W×C 压缩为 1×1×C
- **Excitation**：两层全连接 Dense(C/r, ReLU) → Dense(C, Sigmoid)
- **Scale**：将学到的通道权重乘回原始特征图

#### 3.3.3 注意力可视化

项目提供 `AttentionVisualizer` 类，可以：
- 提取并可视化通道注意力权重（柱状图）
- 提取并可视化空间注意力权重（热力图叠加）
- 统计多样本的平均通道重要性（Top-K 通道分析）

### 3.4 模型构建代码架构

```
models.py
├── ChannelAttention(Layer)      # 通道注意力层
├── SpatialAttention(Layer)      # 空间注意力层
├── CBAM(Layer)                  # CBAM 组合层
├── SEBlock(Layer)               # SE-Net 层
├── AttentionVisualizer          # 注意力可视化器
├── build_transfer_model()       # 通用迁移学习模型构建（工厂模式）
├── build_mobilenetv2_cbam()     # MobileNetV2 + CBAM
├── build_mobilenetv2_se()       # MobileNetV2 + SE-Net
├── build_mobilenetv2_attention()# 统一注意力模型接口
├── build_model()                # 顶层模型分发函数
├── compile_model()              # 模型编译
├── get_model_info()             # 获取模型参数信息
└── build_simple_cnn()           # 简单 CNN 基线
```

---

## 4. 训练策略与优化技术

### 4.1 基础训练配置

| 参数 | 值 | 说明 |
|------|------|------|
| Batch Size | 32 | 每批样本数 |
| Epochs | 30 | 最大训练轮数 |
| Learning Rate | 0.001 | 初始学习率 |
| Optimizer | Adam | 自适应学习率优化器 |
| Early Stopping | patience=5 | 验证损失 5 轮无改善则停止 |
| ReduceLROnPlateau | patience=3, factor=0.5 | 学习率衰减 |
| Min LR | 1e-7 | 最小学习率 |

### 4.2 损失函数

#### 4.2.1 标准交叉熵（带标签平滑）

```python
CategoricalCrossentropy(label_smoothing=0.1)
```

标签平滑将 one-hot 标签 [0,0,...,1,...,0] 转化为 [ε/(K-1), ..., 1-ε, ..., ε/(K-1)]，其中 ε=0.1，防止模型过度自信。

#### 4.2.2 Focal Loss（核心创新点）

用于解决垃圾分类中的类别不平衡问题：

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

- **γ（gamma）= 2.0**：聚焦参数，降低易分类样本的损失权重
- **α（alpha）= 0.25**：类别平衡参数
- 参考论文：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

**直觉理解**：当模型对某样本预测概率 p_t 很高（易分类）时，(1-p_t)^γ 趋近于 0，损失几乎为 0；当 p_t 很低（难分类）时，损失保持较大。这迫使模型集中精力学习难分类样本。

### 4.3 学习率调度策略

项目实现了 4 种学习率调度策略，默认使用 Warmup + Cosine Annealing：

#### 4.3.1 Warmup + Cosine Annealing（默认）

```
阶段 1（Warmup，前 5 个 epoch）：
    lr 从 1e-7 线性增长到 0.001

阶段 2（Cosine Decay）：
    lr = min_lr + 0.5 × (initial_lr - min_lr) × (1 + cos(π × step / total_steps))
```

#### 4.3.2 OneCycleLR（Smith, 2018）

- 学习率先升后降，实现"超级收敛"
- max_lr=0.01，上升阶段占比 30%
- 初始 lr = max_lr / 25，最终 lr = max_lr / 10000

#### 4.3.3 SGDR（Loshchilov & Hutter, 2016）

- 带热重启的余弦退火
- 首次重启周期 T_0=10（以 epoch 计），之后周期倍增（T_mult=2）
- 周期性重启帮助跳出局部最优

#### 4.3.4 Constant + ReduceOnPlateau

- 固定学习率 + 当验证损失停滞时自动缩减

### 4.4 混合精度训练（Mixed Precision）

- 使用 TensorFlow 的 `mixed_float16` 策略
- 前向传播使用 FP16 加速计算
- 权重更新使用 FP32 保证精度
- 可加速 30-50%

### 4.5 指数移动平均（EMA）

- 衰减率 decay=0.999
- 训练过程中维护权重的指数移动平均：`ema_w = decay × ema_w + (1-decay) × model_w`
- 每个 epoch 结束时使用 EMA 权重进行验证评估
- 训练结束时应用 EMA 权重作为最终模型权重
- 目的：平滑权重更新，提升泛化性能

### 4.6 梯度累积

- 通过 `GradientAccumulationModel` 包装器实现
- 累积多个小 batch 的梯度后统一更新
- 用于在显存有限时模拟大 batch 训练
- 默认步数为 1（即不累积）

### 4.7 训练回调函数

| 回调 | 功能 |
|------|------|
| ModelCheckpoint | 保存验证集准确率最高的模型到 `saved_models/{model_name}_best.keras` |
| EarlyStopping | 验证损失 5 轮无改善停止，自动恢复最佳权重 |
| TensorBoard | 记录训练过程到 `logs/{model_name}/`，可通过 TensorBoard 可视化 |
| CSVLogger | 将每轮指标保存到 `outputs/{model_name}/training_log.csv` |
| TimeHistory | 记录每轮训练耗时 |
| LearningRateLogger | 记录并打印每轮学习率 |
| EMACallback | 指数移动平均（可选） |

### 4.8 微调（Fine-tuning）

`fine_tune_model()` 函数支持在初始训练后解冻基模型的最后 N 层进行端到端微调：

- 默认解冻 20 层
- 使用更小的学习率（1e-5）
- 2 个 epoch 的 warmup + cosine decay
- 配合标签平滑

### 4.9 训练流程代码架构

```
trainer.py
├── FocalLoss(Loss)                    # Focal Loss 实现
├── WarmupCosineDecay(LRSchedule)      # Warmup+Cosine 调度器
├── OneCycleLR(LRSchedule)             # OneCycle 调度器
├── SGDRSchedule(LRSchedule)           # SGDR 调度器
├── EMACallback(Callback)              # EMA 回调
├── GradientAccumulationModel(Model)   # 梯度累积包装器
├── TimeHistory(Callback)              # 时间记录回调
├── LearningRateLogger(Callback)       # 学习率日志回调
├── AdvancedTrainer                    # 高级训练器（集成所有优化）
├── train_model()                      # 主训练函数
├── fine_tune_model()                  # 微调函数
├── cross_validate_model()             # K折交叉验证
├── compile_model_with_focal_loss()    # 带 Focal Loss 的编译
├── save_training_results()            # 保存结果为 JSON
├── cleanup_memory()                   # 内存清理
└── get_gpu_memory_info()              # GPU 信息获取
```

---

## 5. 实验设计

### 5.1 实验一：多模型对比实验

**目的**：横向对比 5 种预训练模型在垃圾分类任务上的综合表现。

**对比维度**：
- 分类性能：Accuracy, Precision, Recall, F1-Score
- 计算效率：推理时间（ms/image）、FPS
- 存储效率：模型参数量、模型文件大小（MB）
- 训练效率：训练总耗时

**对比模型**：MobileNetV2, MobileNetV2_CBAM, VGG16, DenseNet121, EfficientNetB0

**执行方式**：
```python
run_all_experiments(data_dir, epochs=30)
```

### 5.2 实验二：消融实验（Ablation Study）

**目的**：验证 CBAM 注意力机制和 Focal Loss 各自的独立贡献。

**实验配置**：

| 配置名称 | 基础模型 | CBAM | Focal Loss | 说明 |
|----------|----------|------|------------|------|
| Baseline | MobileNetV2 | ✗ | ✗ | 原始迁移学习基线 |
| + CBAM | MobileNetV2_CBAM | ✓ | ✗ | 仅添加注意力机制 |
| + Focal Loss | MobileNetV2 | ✗ | ✓ | 仅使用 Focal Loss |
| + CBAM + Focal Loss (Proposed) | MobileNetV2_CBAM | ✓ | ✓ | 完整方案 |

**预期结论**：
- Baseline ~92%
- +CBAM 提升约 1-2%
- +Focal Loss 提升约 0.5-1%
- +Both 协同提升约 2-3%

**执行方式**：
```python
run_ablation_study(data_dir, epochs=30)
```

### 5.3 实验三：统计显著性检验

**目的**：证明改进方案的优势具有统计学显著性。

**方法**：
- **配对 t 检验（Paired t-test）**：适用于正态分布数据
- **Wilcoxon 符号秩检验**：非参数替代方案
- 显著性水平 α=0.05
- 需要 K 折交叉验证（K=5）的多次结果

### 5.4 实验四：模型可解释性分析

**Grad-CAM 可视化**：
- 使用 Grad-CAM（Selvaraju et al., ICCV 2017）生成类激活热力图
- 自动定位模型最后一个卷积层
- 计算目标类别对该层输出的梯度
- 全局平均池化梯度 → 通道权重 → 加权求和 → ReLU → 归一化
- 使用 OpenCV 叠加热力图到原图
- 展示模型决策依据，增强可信度

### 5.5 K 折交叉验证（可选）

- 5 折分层交叉验证
- 使用 `sklearn.model_selection.StratifiedKFold`
- 报告各指标的均值 ± 标准差

---

## 6. 评估指标体系

### 6.1 分类性能指标

| 指标 | 公式/定义 | 说明 |
|------|-----------|------|
| **Accuracy** | 正确预测 / 总样本 | 整体准确率 |
| **Precision (weighted)** | TP / (TP+FP) 加权平均 | 精确率，衡量误报 |
| **Recall (weighted)** | TP / (TP+FN) 加权平均 | 召回率，衡量漏报 |
| **F1-Score (weighted)** | 2×P×R / (P+R) 加权平均 | 精确率和召回率的调和平均 |

### 6.2 计算性能指标

| 指标 | 说明 |
|------|------|
| **单张推理时间 (ms)** | 单张图片的平均推理耗时 |
| **FPS** | 每秒可处理的图片数 |
| **批量推理效率** | 批量处理时每张图片的平均耗时 |
| **模型参数量** | 总参数数量（可训练 + 不可训练） |
| **模型大小 (MB)** | 以 float32 估算的模型文件大小 |
| **训练耗时 (min)** | 完整训练所需时间 |

### 6.3 每类别分析

- 使用 `sklearn.metrics.classification_report` 生成每个类别的 Precision/Recall/F1/Support
- `precision_recall_fscore_support(average=None)` 提供逐类分析
- 可识别模型在哪些垃圾类别上表现较差

---

## 7. 可视化输出

### 7.1 训练过程可视化

- **训练/验证准确率曲线**：展示模型收敛过程和过拟合情况
- **训练/验证损失曲线**：监控损失变化趋势

### 7.2 评估结果可视化

- **混淆矩阵（原始计数 + 归一化百分比）**：展示各类别间的错分情况
- **每类别 Precision/Recall/F1 柱状图**：直观对比各类别性能差异
- **预测样例图**：随机展示测试集上的预测结果，正确为绿色、错误为红色

### 7.3 模型对比可视化

- **准确率对比柱状图**
- **模型大小对比柱状图**
- **推理时间对比柱状图**
- **F1-Score 对比柱状图**

### 7.4 可解释性可视化

- **Grad-CAM 热力图**：叠加到原图上，展示模型关注区域
- **注意力权重可视化**：通道注意力柱状图、空间注意力热力图

### 7.5 论文导出

- `generate_latex_table()`：自动生成 LaTeX 格式的对比结果表格
- 所有图表以 150 DPI 的 PNG 格式保存，适合论文插图

---

## 8. 模型集成模块

### 8.1 集成策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **概率平均** | 所有模型输出概率取平均 | 简单且有效的基线集成 |
| **加权平均** | 按模型性能分配权重后加权平均 | 当模型间性能差异明显时 |
| **投票法** | 选择被最多模型预测的类别 | 分类任务 |
| **堆叠法** | 训练元模型学习如何组合基模型输出 | 复杂集成，性能通常最佳 |
| **Snapshot Ensemble** | 从单次训练的多个快照构建集成 | 节省计算资源 |
| **TTA（测试时增强）** | 对测试图像多次增强后平均预测 | 提高单模型鲁棒性 |

### 8.2 权重优化

使用 `scipy.optimize.minimize` 搜索最优集成权重，最大化验证集准确率。

---

## 9. 代码文件架构与依赖关系

### 9.1 文件结构与功能

```
项目根目录/
│
├── config.py                  # [210行] 全局配置中心
│   └── 所有超参数、路径、模型配置、消融实验配置
│
├── data_loader.py             # [426行] 数据处理
│   ├── 数据集下载引导
│   ├── 数据划分（分层随机）
│   ├── ImageDataGenerator 数据生成器
│   ├── tf.data 高效管道（备选方案）
│   └── 类别权重计算
│
├── models.py                  # [1369行] 模型定义
│   ├── CBAM / SE-Net 注意力机制
│   ├── 5 种预训练模型构建函数
│   ├── 注意力增强模型
│   ├── 模型编译与信息获取
│   └── 注意力权重可视化器
│
├── trainer.py                 # [1612行] 训练模块
│   ├── Focal Loss 损失函数
│   ├── 3 种学习率调度器
│   ├── Mixup / CutMix 增强
│   ├── EMA 指数移动平均
│   ├── 梯度累积
│   ├── 高级训练器 AdvancedTrainer
│   ├── 微调 fine_tune_model()
│   ├── K 折交叉验证
│   └── 结果保存与内存管理
│
├── evaluation.py              # [882行] 评估与可视化
│   ├── 模型评估（Acc/P/R/F1）
│   ├── 混淆矩阵绘制
│   ├── 训练曲线绘制
│   ├── 推理速度测试
│   ├── 模型对比表格与图表
│   ├── 每类别指标可视化
│   ├── Grad-CAM 可视化
│   ├── 统计显著性检验
│   └── LaTeX 表格生成
│
├── ensemble.py                # [1721行] 模型集成
│   ├── ModelEnsemble 类
│   ├── SnapshotEnsemble
│   ├── TestTimeAugmentation
│   ├── WeightOptimizer
│   └── CyclicLearningRate
│
├── main.py                    # [592行] 主实验脚本
│   ├── GPU 配置
│   ├── run_single_experiment()    单模型实验
│   ├── run_all_experiments()      全模型对比实验
│   ├── run_ablation_study()       消融实验
│   └── run_quick_test()           快速测试
│
├── download_data.py           # [445行] 数据集下载脚本
├── quick_test.py              # [236行] 环境验证脚本
├── requirements.txt           # Python 依赖
│
├── garbage_classification_complete.ipynb  # Colab 完整 Notebook
├── COLAB_GUIDE.md             # Colab 使用指南
├── OPTIMIZATION_SUMMARY.md    # 优化总结
└── README.md                  # 项目说明
```

### 9.2 模块依赖关系

```
config.py（无依赖）
    ↓
data_loader.py ← config
models.py ← config
    ↓
trainer.py ← config, models
    ↓
evaluation.py ← config
    ↓
ensemble.py ← config
    ↓
main.py ← config, data_loader, models, trainer, evaluation
```

### 9.3 技术栈

**核心框架：**
- TensorFlow >= 2.10（含 Keras API）
- Python >= 3.8

**数据处理：**
- NumPy >= 1.21
- Pandas >= 1.3
- Scikit-learn >= 1.0
- Pillow >= 8.0

**可视化：**
- Matplotlib >= 3.4
- Seaborn >= 0.11

**其他：**
- OpenCV >= 4.5（Grad-CAM 热力图叠加）
- SciPy >= 1.7（统计检验、权重优化）
- tqdm >= 4.62（进度条）
- Kaggle API >= 1.5.12（数据下载）

---

## 10. 输出目录结构

```
outputs/
├── MobileNetV2/
│   ├── MobileNetV2_confusion_matrix.png     # 混淆矩阵
│   ├── MobileNetV2_training_curves.png      # 训练曲线
│   ├── MobileNetV2_per_class_metrics.png    # 每类别指标
│   ├── MobileNetV2_predictions.png          # 预测样例
│   ├── MobileNetV2_gradcam.png              # Grad-CAM 热力图（如有）
│   ├── training_log.csv                      # 训练日志
│   └── results.json                          # 结果 JSON
├── MobileNetV2_CBAM/
│   └── ...（同上）
├── VGG16/
│   └── ...
├── DenseNet121/
│   └── ...
├── EfficientNetB0/
│   └── ...
├── ablation_Baseline/
│   └── ...（消融实验结果）
├── ablation_+_CBAM/
│   └── ...
├── model_comparison.csv                      # 模型对比表
└── model_comparison.png                      # 对比图表

saved_models/
├── MobileNetV2_best.keras                    # 最佳检查点
├── MobileNetV2_final.keras                   # 最终模型
└── ...

logs/
├── MobileNetV2/                              # TensorBoard 日志
└── ...
```

---

## 11. 关键超参数汇总

| 分类 | 参数 | 值 | 说明 |
|------|------|------|------|
| **数据** | IMG_SIZE | 224 | 输入图像尺寸 |
| | BATCH_SIZE | 32 | 批次大小 |
| | TRAIN/VAL/TEST | 0.8/0.1/0.1 | 数据划分比例 |
| | RANDOM_SEED | 42 | 随机种子 |
| **模型** | FREEZE_BASE | True | 冻结基模型 |
| | DROPOUT_RATE | 0.5 | Dropout 比率 |
| **训练** | EPOCHS | 30 | 最大训练轮数 |
| | LEARNING_RATE | 0.001 | 初始学习率 |
| | EARLY_STOPPING_PATIENCE | 5 | 早停耐心值 |
| | LABEL_SMOOTHING | 0.1 | 标签平滑系数 |
| **学习率调度** | LR_SCHEDULE_TYPE | warmup_cosine | 默认调度器 |
| | WARMUP_EPOCHS | 5 | 预热轮数 |
| | MIN_LR | 1e-7 | 最小学习率 |
| **高级增强** | MIXUP_ALPHA | 0.2 | Mixup 参数 |
| | CUTMIX_ALPHA | 1.0 | CutMix 参数 |
| | USE_MIXUP / USE_CUTMIX | True / True | 启用状态 |
| **EMA** | EMA_DECAY | 0.999 | 指数移动平均衰减率 |
| **Focal Loss** | gamma | 2.0 | 聚焦参数 |
| | alpha | 0.25 | 类别平衡参数 |
| **CBAM** | reduction_ratio | 16 | 通道压缩比 |
| | spatial_kernel_size | 7 | 空间注意力卷积核大小 |
| **交叉验证** | N_FOLDS | 5 | K 折数 |

---

## 12. 技术创新点总结（论文核心卖点）

### 创新点 1：轻量级网络 + 注意力机制

在 MobileNetV2 轻量级网络（仅 3.4M 参数）上集成 CBAM 注意力机制，以最小的参数增量（~0.1M）显著提升分类性能。创新点在于将通道注意力（学习"关注什么特征"）和空间注意力（学习"关注哪个区域"）的双重机制引入垃圾分类任务，使模型能更精准地聚焦于垃圾的关键视觉特征（如纹理、形状、颜色）。

### 创新点 2：Focal Loss 解决类别不平衡

垃圾分类数据集存在类别不平衡问题（例如"纸张"类远多于"电池"类）。Focal Loss 通过自适应调节各样本的损失权重——降低易分类样本的贡献、增大难分类样本的贡献——有效缓解不平衡带来的性能下降。

### 创新点 3：Grad-CAM 可解释性分析

通过梯度加权类激活映射可视化模型的决策过程，展示模型在分类时关注的图像区域。这不仅增强了模型的可信度和透明度，还可用于分析误分类的原因。

### 创新点 4：完整的实验方法论

- **消融实验**：逐步添加 CBAM 和 Focal Loss，量化每个组件的独立贡献
- **多模型对比**：5 种不同架构的全面横向对比
- **统计显著性检验**：配对 t 检验和 Wilcoxon 检验验证改进的显著性
- **多维度评估**：准确率、效率、大小三个维度的综合评价

---

## 13. 论文写作建议框架

### 建议的论文结构

```
1. Introduction（引言）
   - 垃圾分类的背景与挑战
   - 深度学习在图像分类中的应用
   - 现有方法的不足
   - 本文贡献（3-4 点）

2. Related Work（相关工作）
   - 2.1 基于深度学习的图像分类
   - 2.2 迁移学习
   - 2.3 轻量级网络（MobileNet系列）
   - 2.4 注意力机制（CBAM, SE-Net）
   - 2.5 垃圾分类相关研究

3. Methodology（方法）
   - 3.1 整体框架图
   - 3.2 MobileNetV2 基础网络
   - 3.3 CBAM 注意力机制集成
   - 3.4 Focal Loss 损失函数
   - 3.5 训练策略（数据增强、学习率调度、EMA 等）

4. Experiments（实验）
   - 4.1 数据集介绍
   - 4.2 实验设置（超参数、硬件环境）
   - 4.3 多模型对比实验
   - 4.4 消融实验
   - 4.5 Grad-CAM 可视化分析
   - 4.6 统计显著性检验（如有）
   - 4.7 按类别的详细分析
   - 4.8 效率分析（推理速度、模型大小）

5. Discussion（讨论）
   - MobileNetV2_CBAM 为何有效
   - 各类别分类难点分析
   - 模型效率与准确率的权衡
   - 局限性

6. Conclusion（结论）
   - 主要发现
   - 贡献总结
   - 未来工作
```

### 需要准备的关键图表

1. **方法框架图**（Figure 1）：展示整体方案的网络结构
2. **CBAM 模块结构图**（Figure 2）：通道注意力 + 空间注意力
3. **模型对比表**（Table 1）：5 模型的 Acc/P/R/F1/Size/Time
4. **消融实验表**（Table 2）：4 组配置的结果
5. **混淆矩阵**（Figure 3）：最佳模型的分类详情
6. **训练曲线**（Figure 4）：展示收敛过程
7. **Grad-CAM 热力图**（Figure 5）：各类别的注意力可视化
8. **每类别指标图**（Figure 6）：P/R/F1 柱状图
9. **效率对比图**（Figure 7）：Accuracy vs Size, Accuracy vs Speed 散点图

### 关键参考文献

- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
- Smith, "Super-Convergence: Very Fast Training of Neural Networks", 2018

---

## 14. 实验运行指南（快速参考）

```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 验证环境
python quick_test.py

# 3. 下载数据集
python download_data.py

# 4. 运行完整对比实验（5模型×30轮）
python main.py

# 5. 仅运行消融实验
python -c "from main import run_ablation_study; run_ablation_study('./data/garbage_classification')"

# 6. 快速测试（3轮，仅MobileNetV2）
python -c "from main import run_quick_test; run_quick_test()"

# 7. 查看 TensorBoard 日志
tensorboard --logdir=logs/
```

---

## 15. 预期性能指标

基于文献和经验值的**预期范围**（实际结果待实验验证）：

| 模型 | 预期 Accuracy | 预期 F1 | 模型大小(MB) | 推理(ms) |
|------|--------------|---------|-------------|---------|
| MobileNetV2 | ~92-95% | ~0.92-0.95 | ~14 | ~10 |
| MobileNetV2_CBAM | ~94-96% | ~0.94-0.96 | ~14 | ~11 |
| VGG16 | ~91-94% | ~0.91-0.94 | ~528 | ~50 |
| DenseNet121 | ~93-95% | ~0.93-0.95 | ~30 | ~15 |
| EfficientNetB0 | ~92-94% | ~0.92-0.94 | ~20 | ~12 |

消融实验预期：
- Baseline (MobileNetV2): ~92%
- + CBAM: 提升 1-2%
- + Focal Loss: 提升 0.5-1%
- + CBAM + Focal Loss (Proposed): 提升 2-3%

---

## 16. 给 Claude 的特别说明

当你在实验结果出来后分析本项目时，请注意以下要点：

1. **核心研究问题**：本文重点论证的是 MobileNetV2 + CBAM + Focal Loss 的组合方案在垃圾分类中的有效性，同时强调其轻量级（适合嵌入式部署）的优势。

2. **论文逻辑链**：问题（垃圾分类需自动化）→ 方案（迁移学习 + 注意力 + 损失优化）→ 验证（对比实验 + 消融实验 + 可解释性分析）→ 结论（所提方案在准确率和效率之间取得最佳平衡）。

3. **需要关注的实验结果**：
   - MobileNetV2_CBAM 相比 MobileNetV2 的提升幅度
   - 消融实验各组件的独立贡献
   - 哪些垃圾类别容易混淆（通过混淆矩阵分析）
   - Grad-CAM 是否揭示了合理的关注区域
   - MobileNetV2_CBAM 是否在准确率/效率权衡中表现最优

4. **潜在的写作重点**：
   - 如果 CBAM 提升显著：强调注意力机制对垃圾特征提取的帮助
   - 如果 Focal Loss 提升显著：强调类别不平衡是该任务的关键挑战
   - 如果某些类别错分严重：分析原因并讨论改进方向
   - 如果模型大小/速度优势明显：强调嵌入式部署的实用价值

5. **论文写作时的注意事项**：
   - 确保所有实验数据一致，图表中的数据与正文描述吻合
   - 消融实验表格需清晰展示每个组件的增量贡献
   - 统计检验的 p 值要明确标注显著性
   - 引用格式需符合目标会议/期刊要求
   - 方法描述要足够详细以支持可复现性

---

*文档生成时间：2026-02-11*
*项目总代码量：约 8,200+ 行 Python 代码*
*核心模块：6 个 Python 文件 + 1 个 Jupyter Notebook*
