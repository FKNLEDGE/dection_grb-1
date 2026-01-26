# 智能垃圾分类系统

# Intelligent Waste Classification System Based on Transfer Learning and MobileNetV2

## 📖 项目简介

本项目实现了一个基于深度学习迁移学习的智能垃圾分类系统，对比分析了MobileNetV2、ResNet50、VGG16三种预训练模型在垃圾图像分类任务上的性能表现。

### 研究亮点

* ✅ 使用ImageNet预训练权重进行迁移学习，加速模型收敛

* ✅ MobileNetV2轻量级架构，适合嵌入式部署

* ✅ 完整的对比实验框架（准确率、模型大小、推理速度）

* ✅ 丰富的可视化分析（混淆矩阵、训练曲线、类别指标）

## 🗂️ 项目结构

```
garbage_classification/
├── config.py              # 配置文件（超参数、路径、高级训练设置）
├── data_loader.py         # 数据加载与预处理
├── models.py              # 模型定义（MobileNetV2/ResNet50/VGG16 + 注意力机制）
├── trainer.py             # 高级训练模块（混合精度、Mixup/CutMix、EMA等）
├── evaluation.py          # 评估与可视化
├── ensemble.py            # 模型集成模块（投票、堆叠、快照集成等）
├── main.py                # 主实验脚本
├── garbage_classification.ipynb  # Jupyter Notebook 版本
├── notebook_version.py    # Jupyter/Colab Python 脚本版本
├── download_data.py       # 数据集自动下载脚本
├── requirements.txt       # 依赖包
├── README.md              # 说明文档
│
├── data/                  # 数据集目录（需下载）
│   └── garbage_classification/
│       ├── battery/       # 电池
│       ├── biological/    # 生物垃圾
│       ├── cardboard/     # 纸板
│       ├── clothes/       # 衣物
│       ├── glass/         # 玻璃
│       ├── metal/         # 金属
│       ├── paper/         # 纸张
│       ├── plastic/       # 塑料
│       ├── shoes/         # 鞋子
│       ├── trash/         # 其他垃圾
│       ├── white-glass/   # 白色玻璃
│       └── brown-glass/   # 棕色玻璃
│
├── outputs/               # 实验输出
│   ├── MobileNetV2/
│   │   ├── confusion_matrix.png
│   │   ├── training_curves.png
│   │   └── results.json
│   ├── ResNet50/
│   ├── VGG16/
│   └── model_comparison.csv
│
├── saved_models/          # 保存的模型
│   ├── MobileNetV2_best.keras
│   ├── ResNet50_best.keras
│   └── VGG16_best.keras
│
└── logs/                  # TensorBoard日志
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

#### 方法1：自动下载脚本（推荐）

```bash
# 运行自动下载脚本
python download_data.py
```

脚本会引导你完成 Kaggle API 配置和数据下载。

#### 方法2：使用 Kaggle API 手动下载

1. **获取 Kaggle API Token**：

   * 登录 [Kaggle](https://www.kaggle.com)

   * 点击右上角头像 → "Settings"

   * 找到 "API" 部分，点击 "Create New Token"

   * 复制显示的 Token（格式如：`KGAT_xxxxxxxxxxxx`）

2. **配置 API Token**（二选一）：

**方式A：使用环境变量（新版，推荐）**

```bash
# Linux/Mac
export KAGGLE_API_TOKEN=你的Token

# Windows (PowerShell)
$env:KAGGLE_API_TOKEN = "你的Token"

# Python 中设置
import os
os.environ['KAGGLE_API_TOKEN'] = '你的Token'
```

**方式B：使用配置文件（旧版）**

```bash
# Linux/Mac
mkdir -p ~/.kaggle
echo '{"username":"你的用户名","key":"你的Token"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle -Force
echo '{"username":"你的用户名","key":"你的Token"}' > $env:USERPROFILE\.kaggle\kaggle.json
```

3. **下载并解压数据集**：

```bash
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip -d data/
mv data/garbage_classification data/garbage_classification_temp
mv data/garbage_classification_temp/* data/garbage_classification/ 2>/dev/null || mv data/Garbage\ classification data/garbage_classification
```

#### 方法3：手动下载

1. 访问 https://www.kaggle.com/datasets/mostafaabla/garbage-classification

2. 点击 "Download" 按钮下载 zip 文件

3. 解压到项目的 `data/garbage_classification/` 目录

4. 确保目录结构正确（12个类别子文件夹直接在 `garbage_classification` 下）

### 3. 运行实验

```bash
# 运行完整对比实验
python main.py
```

### 4. 查看结果

* 训练曲线和混淆矩阵保存在 `outputs/` 目录

* 模型对比表格 `outputs/model_comparison.csv`

* TensorBoard日志 `logs/` 目录

```bash
# 使用TensorBoard查看训练过程
tensorboard --logdir=logs/
```

## 📓 Jupyter 环境使用

本项目提供专门的 Jupyter Notebook 版本，方便在交互式环境中运行。

### 在本地 Jupyter 中运行

```bash
# 安装 Jupyter
pip install jupyter

# 启动 Jupyter Notebook
jupyter notebook garbage_classification.ipynb
```

### 在 Google Colab 中运行

1. 打开 [Google Colab](https://colab.research.google.com)

2. 选择 "文件" → "上传笔记本" → 上传 `garbage_classification.ipynb` 文件

3. 按照 Notebook 中的步骤配置 Kaggle API 并下载数据

4. 运行代码单元格

**Colab 中配置 Kaggle API**：

```python
from google.colab import files
files.upload()  # 上传 kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mostafaabla/garbage-classification
!unzip -q garbage-classification.zip -d ./data
```

### 快速测试（无需下载数据集）

如果只想测试模型构建功能，可以运行 `notebook_version.py` 中的模型测试部分：

```python
# 测试模型构建（不需要数据集）
python -c "from notebook_version import test_model_building; test_model_building()"
```

## 📊 实验结果（预期）

| Model       | Accuracy | F1-Score | Size(MB) | Inference(ms) |
| ----------- | -------- | -------- | -------- | ------------- |
| MobileNetV2 | ~95%     | ~0.95    | ~14      | ~10           |
| ResNet50    | ~96%     | ~0.96    | ~98      | ~25           |
| VGG16       | ~94%     | ~0.94    | ~528     | ~50           |

**结论**: MobileNetV2在保持高准确率的同时，模型体积最小、推理速度最快，最适合嵌入式部署。

## 🔧 自定义配置

修改 `config.py` 文件可以调整：

```python
# 训练参数
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# 数据增强
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    ...
}

# 模型对比
MODELS_TO_COMPARE = ['MobileNetV2', 'ResNet50', 'VGG16']
```

```
```

