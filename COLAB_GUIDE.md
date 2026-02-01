# Google Colab 使用指南

## 🚀 快速开始

### 步骤1：打开Colab

访问 [Google Colab](https://colab.research.google.com)

### 步骤2：上传Notebook

1. 点击 "文件" → "上传笔记本"
2. 选择 `garbage_classification_complete.ipynb`
3. 或者直接拖拽文件到Colab窗口

### 步骤3：配置GPU

1. 点击 "运行时" → "更改运行时类型"
2. 硬件加速器选择 "GPU" (推荐T4或更好)
3. 点击 "保存"

### 步骤4：获取Kaggle API Token

1. 登录 [Kaggle](https://www.kaggle.com)
2. 点击右上角头像 → Settings
3. 滚动到 "API" 部分
4. 点击 "Create New Token"
5. 下载 `kaggle.json` 文件

### 步骤5：运行Notebook

按照Notebook中的单元格顺序执行：

#### 5.1 检查环境
```python
# 第一个单元格会检查GPU和TensorFlow版本
!nvidia-smi
```

#### 5.2 安装依赖
```python
# 自动安装所有必需的包
!pip install -q tensorflow numpy pandas scikit-learn matplotlib seaborn Pillow opencv-python tqdm kaggle
```

#### 5.3 上传Kaggle Token
```python
# 上传kaggle.json文件
from google.colab import files
uploaded = files.upload()
```

#### 5.4 下载数据集
```python
# 自动下载并解压数据集
!kaggle datasets download -d mostafaabla/garbage-classification
!unzip -q garbage-classification.zip -d ./data
```

#### 5.5 运行训练

**快速测试（3个epoch，约5-10分钟）：**
```python
from main import run_quick_test
results, models, histories = run_quick_test(
    data_dir='./data/garbage_classification',
    epochs=3
)
```

**完整训练（30个epoch，约1-3小时）：**
```python
from main import run_all_experiments
results, models, histories = run_all_experiments(
    data_dir='./data/garbage_classification',
    epochs=30
)
```

**单模型训练（推荐）：**
```python
from main import run_all_experiments
results, models, histories = run_all_experiments(
    data_dir='./data/garbage_classification',
    epochs=20,
    models_to_compare=['MobileNetV2_CBAM']  # 只训练带CBAM的MobileNetV2
)
```

### 步骤6：查看结果

Notebook会自动显示：
- 📊 模型对比表格
- 📈 训练曲线图
- 🎯 混淆矩阵
- 📉 性能分析图

### 步骤7：下载结果

```python
# 打包所有结果
!zip -r experiment_results.zip ./outputs ./saved_models ./logs

# 下载到本地
from google.colab import files
files.download('experiment_results.zip')
```

---

## 💡 使用技巧

### 1. 节省运行时间

如果GPU运行时间有限，建议：
- 使用快速测试模式（3个epoch）
- 只训练1-2个模型
- 减少epoch数量（10-15个epoch通常已经足够）

### 2. 避免断连

Colab可能会因为长时间无操作而断开连接：
- 定期查看训练进度
- 使用浏览器插件防止断连（如 Colab Auto Clicker）
- 训练重要模型时，保存检查点

### 3. 查看训练进度

```python
# 启动TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/
```

### 4. 内存管理

如果遇到内存不足错误：
```python
# 清理GPU内存
import gc
import tensorflow as tf

gc.collect()
tf.keras.backend.clear_session()
```

---

## ⚠️ 常见问题

### Q1: Kaggle API配置失败
**解决方法：**
```python
# 手动配置
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Q2: 数据集下载失败
**解决方法：**
1. 检查Kaggle token是否正确
2. 确保已接受数据集的使用条款（访问数据集页面点击接受）
3. 尝试重新下载token

### Q3: GPU运行时间用完
**解决方法：**
- Colab免费版每天有GPU使用限制
- 等待一段时间后重试
- 考虑升级到Colab Pro

### Q4: 训练中断
**解决方法：**
- 模型会自动保存检查点
- 可以从最新检查点恢复训练
- 查看 `saved_models/` 目录中的模型文件

### Q5: 可视化图表不显示
**解决方法：**
```python
# 重新导入matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```

---

## 📊 推荐配置

### 快速体验（约10分钟）
- 模型：MobileNetV2
- Epoch：3
- 预期准确率：~80-85%

### 标准实验（约30-60分钟）
- 模型：MobileNetV2, MobileNetV2_CBAM
- Epoch：20
- 预期准确率：~92-95%

### 完整对比（约2-3小时）
- 模型：全部5个模型
- Epoch：30
- 预期准确率：~95-97%

### 消融实验（约1-2小时）
- 配置：4种配置（Baseline, +CBAM, +Focal Loss, +Both）
- Epoch：20
- 用于论文写作

---

## 🎓 学习资源

### TensorFlow教程
- [官方文档](https://www.tensorflow.org/tutorials)
- [迁移学习指南](https://www.tensorflow.org/tutorials/images/transfer_learning)

### 论文参考
- MobileNetV2: https://arxiv.org/abs/1801.04381
- CBAM: https://arxiv.org/abs/1807.06521
- Focal Loss: https://arxiv.org/abs/1708.02002

### Colab技巧
- [Colab官方教程](https://colab.research.google.com/notebooks/welcome.ipynb)
- [GPU使用指南](https://colab.research.google.com/notebooks/gpu.ipynb)

---

## 📧 支持

如有问题：
1. 查看项目README.md
2. 检查本指南的常见问题部分
3. 在GitHub仓库提交Issue

---

**祝实验顺利！🎉**
