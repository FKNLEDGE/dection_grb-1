# EI 论文写作指南：Introduction + Methods

> **写给谁**：第一次写英文学术论文的同学。本指南把每一段该写什么、怎么写、写多长都说清楚了，你只需要按照模板填空，然后润色就行。
>
> **什么时候写**：Introduction 和 Methods 不需要等实验结果，现在就能写。建议先写 Methods（因为你最清楚自己做了什么），再写 Introduction。

---

## 0. 论文整体结构一览

EI 会议论文一般 6-8 页，结构如下：

| 章节 | 英文标题 | 大概篇幅 | 什么时候写 |
|------|----------|----------|-----------|
| 1 | Introduction | 1-1.5 页 | **现在写** |
| 2 | Related Work | 0.5-1 页 | 现在写（可选，也可并入 Intro） |
| 3 | Proposed Method | 1.5-2 页 | **现在写** |
| 4 | Experiments | 2-2.5 页 | 等结果出来后写 |
| 5 | Conclusion | 0.5 页 | 最后写 |
| - | References | 0.5-1 页 | 边写边加 |

**写作顺序建议**：Methods → Introduction → Experiments → Conclusion → Abstract

---

## 1. Introduction 写作指南

Introduction 一共 4 段，像一个"漏斗"：从大背景逐渐聚焦到你的具体工作。

---

### 1.1 第一段：研究背景（为什么这个问题重要？）

**写作公式**：大背景 → 具体问题 → 为什么难

**你要回答的问题**：
- 垃圾分类为什么重要？（环保、政策）
- 人工分类有什么问题？（效率低、成本高、不一致）
- 为什么需要自动化？（深度学习的机会）

**英文范文**（你可以直接改写）：

> With the rapid urbanization and increasing consumption, municipal solid waste management has become a critical environmental challenge worldwide [1]. Effective waste classification is essential for recycling, reducing landfill usage, and promoting a sustainable circular economy. However, manual waste sorting is labor-intensive, time-consuming, and prone to human error, making it impractical for large-scale deployment. In recent years, deep learning-based image classification has achieved remarkable success in various visual recognition tasks [2], providing a promising solution for automatic waste classification. Nevertheless, deploying large-scale deep neural networks on resource-constrained edge devices, such as smart waste bins, remains a significant challenge due to the high computational cost and memory requirements.

**逐句解释**：
1. 大背景：城市化 → 垃圾管理是全球挑战
2. 为什么重要：分类对回收和环保至关重要
3. 现有问题：人工分类效率低、易出错
4. 机会：深度学习在图像分类上很成功
5. 但是：大模型难以部署到边缘设备（引出你的研究动机）

**小贴士**：
- 第一句要"大"，让读者觉得这个话题很重要
- 最后一句要"转折"，引出还没解决的问题
- 加 1-2 个引用（[1] [2]），显得有据可依

---

### 1.2 第二段：现有方法及其不足（别人做了什么？还差什么？）

**写作公式**：现有方法 A → 现有方法 B → 它们的共同不足

**你要回答的问题**：
- 迁移学习在垃圾分类上的应用
- 现有方法的局限（没有注意力机制、忽略类别不平衡、缺乏可解释性）

**英文范文**：

> Transfer learning has been widely adopted for waste classification tasks, where pre-trained convolutional neural networks (CNNs) such as VGG [3], ResNet [4], and DenseNet [5] are fine-tuned on waste image datasets. While these models achieve competitive accuracy, they suffer from several limitations. First, heavyweight architectures like VGG16 (138M parameters) are impractical for real-time deployment on embedded devices. Lightweight alternatives like MobileNetV2 [6] offer significantly reduced computational cost through depthwise separable convolutions, but their classification accuracy on fine-grained waste categories often lags behind larger models. Second, most existing approaches use standard cross-entropy loss, which treats all samples equally and fails to address the inherent class imbalance commonly found in waste datasets. Third, few studies provide interpretability analysis to explain the model's decision-making process, which is crucial for building trust in automated classification systems.

**逐句解释**：
1. 迁移学习是主流方法，列举几个常见模型
2. 但是有局限
3. 局限一：大模型太重，部署不了
4. 轻量级模型精度又不够
5. 局限二：标准交叉熵不处理类别不平衡
6. 局限三：缺少可解释性分析

**小贴士**：
- 不要说别人的方法"差"，说"有改进空间"
- 每个局限正好对应你后面的一个创新点
- 引用具体的论文名字，显得你读过文献

---

### 1.3 第三段：本文的方法和贡献（你做了什么？）

**写作公式**：In this paper, we propose ... → 具体方法 → 贡献点列表

**这是 Introduction 最重要的一段！**

**英文范文**：

> To address these limitations, this paper proposes an enhanced waste classification framework based on MobileNetV2 with integrated attention mechanism and optimized training strategy. Specifically, we incorporate the Convolutional Block Attention Module (CBAM) [7] into the MobileNetV2 backbone to enhance the model's ability to focus on discriminative features of waste objects, while maintaining the lightweight advantage suitable for edge deployment. Furthermore, we adopt Focal Loss [8] to mitigate the negative impact of class imbalance by dynamically down-weighting well-classified samples. We also employ Gradient-weighted Class Activation Mapping (Grad-CAM) [9] to provide visual explanations of the model's predictions. The main contributions of this work are summarized as follows:
>
> - We propose an attention-enhanced lightweight classification model (MobileNetV2-CBAM) that integrates channel and spatial attention mechanisms to improve feature representation for waste classification.
> - We introduce Focal Loss combined with advanced training techniques including label smoothing, Mixup/CutMix data augmentation, and exponential moving average (EMA) to boost classification performance.
> - We conduct comprehensive experiments including a 9-configuration ablation study to validate the effectiveness of each proposed component, along with Grad-CAM analysis for model interpretability.
> - We perform extensive comparisons with five CNN architectures (MobileNetV2, VGG16, DenseNet121, EfficientNetB0) and demonstrate that the proposed method achieves superior accuracy with minimal computational overhead.

**逐句解释**：
1. "To address these limitations" — 承接上一段的不足
2. 核心方法：MobileNetV2 + CBAM
3. 为什么用 CBAM：关注关键特征 + 保持轻量
4. 为什么用 Focal Loss：解决类别不平衡
5. 为什么用 Grad-CAM：可解释性
6. 贡献点 1：模型创新（CBAM + MobileNetV2）
7. 贡献点 2：训练策略创新（Focal Loss + 高级技术）
8. 贡献点 3：实验设计（9 组消融 + Grad-CAM）
9. 贡献点 4：全面对比实验

**小贴士**：
- 贡献点用 bullet list，一般 3-4 个
- 每个贡献点一句话，不要太长
- "We propose / We introduce / We conduct" 开头

---

### 1.4 第四段（可选）：论文组织结构

这段很短，有的会议论文会省略。写了显得更正式。

**英文范文**：

> The remainder of this paper is organized as follows. Section 2 describes the proposed method in detail, including the network architecture, attention mechanism, and loss function. Section 3 presents the experimental setup, results, and analysis. Section 4 concludes the paper and discusses future work.

---

## 2. Methods（Proposed Method）写作指南

Methods 是论文的技术核心，要把你的方法讲清楚，让读者能复现。

---

### 2.1 Overall Architecture（整体架构）

**你要做什么**：
- 画一张整体架构图（必须有！）
- 用文字描述整体流程

**英文范文**：

> Figure 1 illustrates the overall architecture of the proposed waste classification framework. The system takes a waste image of size 224 × 224 × 3 as input and outputs a probability distribution over 12 waste categories. The framework consists of three main components: (1) a MobileNetV2 backbone pre-trained on ImageNet for feature extraction, (2) a CBAM attention module that refines the extracted features through channel and spatial attention, and (3) a classification head that maps the refined features to class predictions.
>
> The MobileNetV2 [6] backbone employs inverted residual blocks with depthwise separable convolutions, achieving an effective balance between computational efficiency and feature representation capability. The base model contains approximately 3.4 million parameters and is initially frozen during training to leverage the pre-trained ImageNet features. The CBAM module is inserted after the encoder output to selectively emphasize informative features. The classification head consists of a global average pooling layer, followed by a dropout layer (rate = 0.5), a dense layer with 256 units and ReLU activation, a second dropout layer (rate = 0.25), and a final softmax layer producing 12-class probabilities.

**这里需要一张图**：

```
Input Image (224×224×3)
    ↓
MobileNetV2 Backbone (frozen, ImageNet pre-trained)
    ↓
CBAM Attention Module
  ├── Channel Attention (r=16)
  └── Spatial Attention (7×7)
    ↓
Global Average Pooling
    ↓
Dropout (0.5) → Dense (256, ReLU) → Dropout (0.25)
    ↓
Dense (12, Softmax) → Output
```

> **给学生的建议**：用 PowerPoint 或 draw.io 画这张图，导出为高清 PNG。这是论文最重要的图之一。

---

### 2.2 CBAM Attention Mechanism（核心创新，要写详细）

这是你论文的**核心技术贡献**，必须写得最详细。

**英文范文**：

> The Convolutional Block Attention Module (CBAM) [7] is a lightweight attention mechanism that sequentially applies channel attention and spatial attention to refine feature representations. Given an intermediate feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$, CBAM produces a refined feature map $\mathbf{F''} \in \mathbb{R}^{H \times W \times C}$ through two sequential steps.

#### 2.2.1 Channel Attention

> **Channel Attention** aims to identify "what" is meaningful among the feature channels. The channel attention map $\mathbf{M}_c \in \mathbb{R}^{1 \times 1 \times C}$ is computed as:
>
> $$\mathbf{M}_c(\mathbf{F}) = \sigma(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F})))$$
>
> where $\sigma$ denotes the sigmoid function, and MLP is a shared two-layer perceptron with a reduction ratio $r = 16$:
>
> $$\text{MLP}(\mathbf{x}) = \mathbf{W}_1(\text{ReLU}(\mathbf{W}_0(\mathbf{x})))$$
>
> where $\mathbf{W}_0 \in \mathbb{R}^{C/r \times C}$ and $\mathbf{W}_1 \in \mathbb{R}^{C \times C/r}$. The global average pooling and global max pooling aggregate spatial information to produce two channel descriptors, which are fed through the shared MLP and combined via element-wise summation. The refined feature map is obtained by:
>
> $$\mathbf{F'} = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$$

**逐句解释**：
1. 通道注意力的目标：找出哪些通道（特征）更重要
2. 公式：对特征图做全局平均池化和最大池化
3. 通过共享的两层 MLP（压缩比 r=16）
4. 两路输出相加后 sigmoid → 得到通道权重
5. 用权重乘回原始特征图

#### 2.2.2 Spatial Attention

> **Spatial Attention** focuses on "where" the informative regions are located. The spatial attention map $\mathbf{M}_s \in \mathbb{R}^{H \times W \times 1}$ is computed as:
>
> $$\mathbf{M}_s(\mathbf{F'}) = \sigma(f^{7 \times 7}([\text{AvgPool}_c(\mathbf{F'}); \text{MaxPool}_c(\mathbf{F'})]))$$
>
> where $f^{7 \times 7}$ denotes a $7 \times 7$ convolution layer, $[\cdot;\cdot]$ represents channel-wise concatenation, and $\text{AvgPool}_c$ and $\text{MaxPool}_c$ compute channel-wise average and maximum, respectively. The final output of CBAM is:
>
> $$\mathbf{F''} = \mathbf{M}_s(\mathbf{F'}) \otimes \mathbf{F'}$$

**逐句解释**：
1. 空间注意力的目标：找出特征图上哪些位置更重要
2. 对通道维度做平均池化和最大池化 → 得到两个 H×W×1 的图
3. 拼接后通过 7×7 卷积 + sigmoid → 得到空间权重
4. 用权重乘回特征图

> The sequential application of channel and spatial attention allows the model to selectively enhance discriminative features along both dimensions. Compared to SE-Net [10] which only performs channel attention, CBAM provides a more comprehensive attention mechanism with negligible computational overhead (approximately 0.1M additional parameters).

---

### 2.3 Focal Loss（损失函数）

**英文范文**：

> In waste classification datasets, certain categories (e.g., battery, biological waste) are often underrepresented compared to common categories (e.g., plastic, paper), leading to class imbalance. Standard cross-entropy loss treats all samples equally, causing the model to be biased toward majority classes. To address this issue, we adopt Focal Loss [8], which dynamically adjusts the contribution of each sample based on its classification difficulty:
>
> $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$
>
> where $p_t$ is the predicted probability of the ground-truth class, $\gamma = 2.0$ is the focusing parameter that reduces the relative loss for well-classified examples (when $p_t$ is large, $(1 - p_t)^{\gamma} \to 0$), and $\alpha = 0.25$ is the class-balancing factor. Intuitively, Focal Loss assigns higher weights to hard-to-classify samples while suppressing the overwhelming contribution of easy negatives, thereby improving the model's performance on minority and challenging categories.

**写作要点**：
- 先说"为什么需要"（类别不平衡问题）
- 再给公式
- 然后解释每个符号的含义和取值
- 最后用直觉语言解释公式的效果

---

### 2.4 Training Strategy（训练策略）

**英文范文**：

> **Data Augmentation.** To prevent overfitting and improve generalization, we apply both conventional and advanced data augmentation techniques. Conventional augmentations include random rotation (±20°), width and height shifting (±20%), horizontal flipping, zoom (±20%), and brightness adjustment (range [0.8, 1.2]). Additionally, we employ Mixup [11] ($\alpha = 0.2$) and CutMix [12] ($\alpha = 1.0$), which are randomly selected with equal probability during training.
>
> **Learning Rate Schedule.** We adopt a warmup-cosine annealing strategy. During the first 5 epochs, the learning rate linearly increases from $10^{-7}$ to the initial learning rate of $10^{-3}$ (warmup phase). Subsequently, the learning rate follows a cosine decay:
>
> $$\text{lr}_t = \text{lr}_{min} + \frac{1}{2}(\text{lr}_{init} - \text{lr}_{min})(1 + \cos(\frac{\pi \cdot t}{T}))$$
>
> where $\text{lr}_{min} = 10^{-7}$ and $T$ is the total number of decay steps.
>
> **Additional Techniques.** We incorporate several additional training techniques: (1) Label smoothing ($\epsilon = 0.1$) [13] to prevent over-confident predictions; (2) Exponential Moving Average (EMA) with a decay rate of 0.999 to stabilize training and improve generalization; (3) Mixed precision training (FP16 forward pass, FP32 weight updates) for computational acceleration.
>
> **Optimization.** All models are trained using the Adam optimizer with an initial learning rate of $10^{-3}$, a batch size of 32, and a maximum of 30 epochs. Early stopping is applied with a patience of 5 epochs based on validation loss to prevent overfitting.

---

## 3. Experimental Setup（实验设置，可提前写）

这部分介绍实验的设置细节，不需要实验结果。

---

### 3.1 Dataset

**英文范文**：

> We evaluate our method on the Garbage Classification dataset from Kaggle, which contains images of 12 waste categories: battery, biological, brown-glass, cardboard, clothes, glass, metal, paper, plastic, shoes, trash, and white-glass. The dataset is split into training, validation, and test sets with a ratio of 8:1:1 using stratified random sampling to maintain class distribution. All images are resized to 224 × 224 pixels and normalized to the range [0, 1]. Class weights are computed inversely proportional to class frequency to further address class imbalance during training.

### 3.2 Implementation Details

**英文范文**：

> All experiments are implemented in Python using TensorFlow 2.x and trained on an NVIDIA GPU. (具体型号跑完实验后填). The MobileNetV2 backbone is initialized with ImageNet pre-trained weights, and its layers are frozen during training. The CBAM module uses a reduction ratio of $r = 16$ for channel attention and a $7 \times 7$ kernel for spatial attention. For Focal Loss, we set $\gamma = 2.0$ and $\alpha = 0.25$. The random seed is fixed at 42 for reproducibility.

### 3.3 Evaluation Metrics

**英文范文**：

> We evaluate the classification performance using four standard metrics: Accuracy, weighted Precision, weighted Recall, and weighted F1-Score. Additionally, we report model parameters (M), model size (MB), and single-image inference time (ms) to assess computational efficiency. All metrics are computed on the held-out test set.

### 3.4 Compared Methods and Ablation Design

**英文范文**：

> **Model Comparison.** We compare the proposed MobileNetV2-CBAM with four baseline models: the original MobileNetV2 [6], VGG16 [3], DenseNet121 [5], and EfficientNetB0 [14]. All models follow the same transfer learning framework and training protocol for fair comparison.
>
> **Ablation Study.** To validate the contribution of each proposed component, we conduct an extensive ablation study with 9 configurations, progressively adding components to the baseline: (1) Baseline (plain MobileNetV2), (2) +CBAM, (3) +SE-Net (for attention comparison), (4) +Focal Loss, (5) +Label Smoothing, (6) +Mixup/CutMix, (7) +EMA, (8) +CBAM+Focal Loss, and (9) Full Proposed (all components combined).

---

## 4. 参考文献模板

以下是你**必须引用**的论文，按 IEEE 格式排列。复制粘贴到你的 References 章节即可。

```
[1]  中国/你所在地区的垃圾分类政策文献（自己找一篇相关的政策或综述论文）

[2]  A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep
     convolutional neural networks," in Advances in Neural Information Processing Systems
     (NeurIPS), 2012, pp. 1097-1105.

[3]  K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale
     image recognition," in International Conference on Learning Representations (ICLR), 2015.

[4]  K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition,"
     in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.

[5]  G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected
     convolutional networks," in IEEE Conference on Computer Vision and Pattern Recognition
     (CVPR), 2017, pp. 4700-4708.

[6]  M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted
     residuals and linear bottlenecks," in IEEE Conference on Computer Vision and Pattern
     Recognition (CVPR), 2018, pp. 4510-4520.

[7]  S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention
     module," in European Conference on Computer Vision (ECCV), 2018, pp. 3-19.

[8]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object
     detection," in IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980-2988.

[9]  R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM:
     Visual explanations from deep networks via gradient-based localization," in IEEE
     International Conference on Computer Vision (ICCV), 2017, pp. 618-626.

[10] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in IEEE Conference on
     Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132-7141.

[11] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk
     minimization," in International Conference on Learning Representations (ICLR), 2018.

[12] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization
     strategy to train strong classifiers with localizable features," in IEEE International
     Conference on Computer Vision (ICCV), 2019, pp. 6023-6032.

[13] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the inception
     architecture for computer vision," in IEEE Conference on Computer Vision and Pattern
     Recognition (CVPR), 2016, pp. 2818-2826.

[14] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural
     networks," in International Conference on Machine Learning (ICML), 2019, pp. 6105-6114.

[15] S. J. Pan and Q. Yang, "A survey on transfer learning," IEEE Transactions on Knowledge
     and Data Engineering, vol. 22, no. 10, pp. 1345-1359, 2010.
```

> **注意**：你还需要自己搜索 2-3 篇垃圾分类相关的论文加到引用里（特别是 Introduction 第一段和第二段）。去 Google Scholar 搜 "waste classification deep learning" 或 "garbage classification CNN"。

---

## 5. 常见错误和写作技巧

### 避免的错误

| 错误 | 正确做法 |
|------|---------|
| 中文直译式英语："In our country, waste problem is very serious" | 用学术表达："Municipal solid waste management has become a critical challenge" |
| 过度自夸："Our method is the best" | 客观表达："The proposed method achieves competitive/superior performance" |
| 不引用就用术语：直接说 "CBAM" | 第一次出现要写全称并引用："Convolutional Block Attention Module (CBAM) [7]" |
| 公式没有解释 | 每个公式后面都要解释符号含义 |
| 段落只有一两句话 | 每段至少 4-6 句话 |
| 用 "I" 或 "my" | 学术论文用 "we" 和 "our"（即使只有一个作者） |
| 拼写不一致："colour" 和 "color" 混用 | 全文统一用美式或英式拼写 |

### 实用写作技巧

1. **第一次提到缩写**：写全称 + (缩写) + [引用]，如 "Convolutional Block Attention Module (CBAM) [7]"。之后可以直接用 "CBAM"。

2. **引用位置**：放在句尾标点前面。
   - 正确："... has been widely studied [1, 2]."
   - 错误："... has been widely studied. [1, 2]"

3. **连接词**：
   - 并列："Furthermore, / Additionally, / Moreover,"
   - 转折："However, / Nevertheless, / In contrast,"
   - 因果："Therefore, / Consequently, / As a result,"
   - 举例："Specifically, / In particular, / For instance,"

4. **描述你的方法**用现在时："We propose... / The model consists of..."

5. **描述别人的工作**用过去时："Woo et al. [7] proposed CBAM..."

6. **数字规范**：
   - 小于 10 的数字用英文："three layers"
   - 大于等于 10 的数字用阿拉伯数字："12 classes"
   - 带单位的用数字："0.5 dropout rate"

7. **图表引用**：
   - "as shown in Fig. 1" 或 "Figure 1 illustrates..."
   - "as listed in Table 1" 或 "Table 1 summarizes..."

---

## 6. 你现在可以做的事

1. **画架构图**：用 PPT 或 draw.io 画 Figure 1（整体架构）和 Figure 2（CBAM 结构）
2. **按模板写 Methods**：最熟悉的部分先写，直接填入你的参数值
3. **按模板写 Introduction**：按照 4 段公式填写
4. **搜索 2-3 篇垃圾分类相关论文**：加到引用中
5. **确定投稿会议**：了解该会议的论文模板格式（IEEE, ACM, Springer 等）
6. **下载论文模板**：按会议要求的 LaTeX 或 Word 模板排版

> **下一步**：等实验结果出来后，用 `python generate_paper_results.py` 生成全部图表和 LaTeX 表格，然后写 Experiments 章节。到时候可以让 AI 帮你根据实验数据写 Results 和 Analysis。
