"""
配置文件 - 智能垃圾分类系统
Configuration for Intelligent Waste Classification System

本文件集中管理所有配置参数，其他模块应显式导入所需配置。
"""

import os
from typing import List, Dict, Any

# ==================== 路径配置 ====================
# 数据集路径 (下载后修改为实际路径)
DATA_DIR: str = "./data/garbage_classification"
OUTPUT_DIR: str = "./outputs"
MODEL_DIR: str = "./saved_models"
LOG_DIR: str = "./logs"


def ensure_dir(path: str) -> str:
    """
    确保目录存在，如不存在则创建

    Args:
        path: 目录路径

    Returns:
        str: 创建后的目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path


# 创建必要目录
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    ensure_dir(dir_path)


# ==================== 数据集配置 ====================
# Kaggle Garbage Classification Dataset - 12类
CLASS_NAMES: List[str] = [
    'battery',      # 电池
    'biological',   # 生物垃圾
    'cardboard',    # 纸板
    'clothes',      # 衣物
    'glass',        # 玻璃
    'metal',        # 金属
    'paper',        # 纸张
    'plastic',      # 塑料
    'shoes',        # 鞋子
    'trash',        # 其他垃圾
    'white-glass',  # 白色玻璃
    'brown-glass'   # 棕色玻璃
]
NUM_CLASSES: int = len(CLASS_NAMES)


# ==================== 模型配置 ====================
# MobileNetV2/VGG16/ResNet50 标准输入尺寸
IMG_SIZE: int = 224  # 推荐值: 224 (标准) 或 299 (Inception系列)
IMG_SHAPE: tuple = (IMG_SIZE, IMG_SIZE, 3)

# 迁移学习配置
FREEZE_BASE: bool = True  # 是否冻结基模型 (True: 特征提取, False: 端到端微调)
DROPOUT_RATE: float = 0.5  # Dropout比率 (推荐: 0.3-0.5)


# ==================== 训练配置 ====================
BATCH_SIZE: int = 32           # 批次大小 (推荐: 16-64, 取决于GPU内存)
EPOCHS: int = 30               # 训练轮数 (推荐: 20-50, 配合早停)
LEARNING_RATE: float = 0.001   # 学习率 (迁移学习推荐: 0.0001-0.001)
EARLY_STOPPING_PATIENCE: int = 5   # 早停耐心值 (验证损失无改善的轮数)
REDUCE_LR_PATIENCE: int = 3        # 学习率衰减耐心值
REDUCE_LR_FACTOR: float = 0.5      # 学习率衰减因子
MIN_LR: float = 1e-7               # 最小学习率

# 数据划分比例 (总和应为1.0)
TRAIN_SPLIT: float = 0.8   # 训练集比例
VAL_SPLIT: float = 0.1     # 验证集比例
TEST_SPLIT: float = 0.1    # 测试集比例

# 最小样本数阈值 (低于此值会发出警告)
MIN_SAMPLES_PER_CLASS: int = 10


# ==================== 数据增强配置 ====================
AUGMENTATION_CONFIG: Dict[str, Any] = {
    'rotation_range': 20,           # 旋转角度范围 (度)
    'width_shift_range': 0.2,       # 水平平移范围 (比例)
    'height_shift_range': 0.2,      # 垂直平移范围 (比例)
    'horizontal_flip': True,        # 是否水平翻转
    'zoom_range': 0.2,              # 缩放范围
    'brightness_range': [0.8, 1.2], # 亮度变化范围
    'fill_mode': 'nearest'          # 填充模式
}


# ==================== 性能优化配置 ====================
USE_MIXED_PRECISION: bool = True   # 是否使用混合精度训练 (可加速30-50%)
USE_XLA: bool = False              # 是否启用XLA编译 (实验性)
PREFETCH_BUFFER: int = -1          # 预取缓冲区大小 (-1表示自动)
SHUFFLE_BUFFER: int = 1000         # 数据打乱缓冲区大小


# ==================== 高级训练配置 ====================
# 学习率调度器类型: 'warmup_cosine', 'one_cycle', 'sgdr', 'reduce_on_plateau'
LR_SCHEDULE_TYPE: str = 'warmup_cosine'

# Warmup + Cosine Annealing 配置
WARMUP_EPOCHS: int = 5             # Warmup轮数
WARMUP_LR_INIT: float = 1e-7       # Warmup起始学习率

# OneCycleLR 配置
ONE_CYCLE_MAX_LR: float = 0.01     # OneCycle最大学习率
ONE_CYCLE_PCT_START: float = 0.3   # 上升阶段占比

# SGDR (带热重启的余弦退火) 配置
SGDR_T_0: int = 10                 # 首次重启周期
SGDR_T_MULT: int = 2               # 周期倍增因子
SGDR_LR_MIN: float = 1e-7          # 最小学习率

# 梯度累积配置
GRADIENT_ACCUMULATION_STEPS: int = 1  # 梯度累积步数 (1=不累积)
EFFECTIVE_BATCH_SIZE: int = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# 标签平滑配置
LABEL_SMOOTHING: float = 0.1       # 标签平滑系数 (0=不平滑, 推荐0.1)

# Mixup / CutMix 配置
USE_MIXUP: bool = True             # 是否使用Mixup
MIXUP_ALPHA: float = 0.2           # Mixup Beta分布参数
USE_CUTMIX: bool = True            # 是否使用CutMix
CUTMIX_ALPHA: float = 1.0          # CutMix Beta分布参数
MIXUP_CUTMIX_PROB: float = 0.5     # 使用Mixup vs CutMix的概率

# EMA (指数移动平均) 配置
USE_EMA: bool = True               # 是否使用EMA
EMA_DECAY: float = 0.999           # EMA衰减率 (推荐0.999-0.9999)


# ==================== 实验对比模型 ====================
# 更新：添加了DenseNet121和EfficientNetB0，以及CBAM增强模型
MODELS_TO_COMPARE: List[str] = [
    'MobileNetV2',
    'MobileNetV2_CBAM',
    'VGG16',
    'DenseNet121',
    'EfficientNetB0'
]

# 各模型的分类头配置
MODEL_HEAD_CONFIG: Dict[str, Dict[str, Any]] = {
    'MobileNetV2': {'dense_units': [128], 'dropout_rates': [0.5, 0.25]},
    'ResNet50': {'dense_units': [256], 'dropout_rates': [0.5, 0.25]},
    'VGG16': {'dense_units': [512, 256], 'dropout_rates': [0.5, 0.25, 0.25]},
    'DenseNet121': {'dense_units': [256], 'dropout_rates': [0.5, 0.25]},
    'EfficientNetB0': {'dense_units': [128], 'dropout_rates': [0.5, 0.25]},
}

# ==================== 消融实验配置 ====================
# 消融实验：验证CBAM和Focal Loss各自的贡献
# 格式: (模型名称, 是否使用Focal Loss, 配置描述)
ABLATION_MODELS: List[tuple] = [
    ('MobileNetV2', False, 'Baseline'),
    ('MobileNetV2_CBAM', False, '+ CBAM'),
    ('MobileNetV2', True, '+ Focal Loss'),
    ('MobileNetV2_CBAM', True, '+ CBAM + Focal Loss (Proposed)'),
]

# ==================== 交叉验证配置 ====================
N_FOLDS: int = 5  # K折交叉验证的折数


# ==================== 随机种子 ====================
RANDOM_SEED: int = 42


# ==================== 导出配置 ====================
# 显式列出所有可导出的配置项，避免使用 import *
__all__ = [
    # 路径
    'DATA_DIR', 'OUTPUT_DIR', 'MODEL_DIR', 'LOG_DIR',
    # 工具函数
    'ensure_dir',
    # 数据集
    'CLASS_NAMES', 'NUM_CLASSES',
    # 模型
    'IMG_SIZE', 'IMG_SHAPE', 'FREEZE_BASE', 'DROPOUT_RATE',
    # 训练
    'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE',
    'EARLY_STOPPING_PATIENCE', 'REDUCE_LR_PATIENCE', 'REDUCE_LR_FACTOR', 'MIN_LR',
    'TRAIN_SPLIT', 'VAL_SPLIT', 'TEST_SPLIT', 'MIN_SAMPLES_PER_CLASS',
    # 数据增强
    'AUGMENTATION_CONFIG',
    # 性能优化
    'USE_MIXED_PRECISION', 'USE_XLA', 'PREFETCH_BUFFER', 'SHUFFLE_BUFFER',
    # 高级训练
    'LR_SCHEDULE_TYPE', 'WARMUP_EPOCHS', 'WARMUP_LR_INIT',
    'ONE_CYCLE_MAX_LR', 'ONE_CYCLE_PCT_START',
    'SGDR_T_0', 'SGDR_T_MULT', 'SGDR_LR_MIN',
    'GRADIENT_ACCUMULATION_STEPS', 'EFFECTIVE_BATCH_SIZE',
    'LABEL_SMOOTHING',
    'USE_MIXUP', 'MIXUP_ALPHA', 'USE_CUTMIX', 'CUTMIX_ALPHA', 'MIXUP_CUTMIX_PROB',
    'USE_EMA', 'EMA_DECAY',
    # 实验
    'MODELS_TO_COMPARE', 'MODEL_HEAD_CONFIG',
    # 消融实验
    'ABLATION_MODELS', 'N_FOLDS',
    # 随机种子
    'RANDOM_SEED',
]
