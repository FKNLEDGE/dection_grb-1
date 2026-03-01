#!/usr/bin/env python3
"""
============================================================================
论文结果生成器 - 智能垃圾分类系统
Paper Results Generator - Intelligent Waste Classification System
============================================================================

功能说明（一键运行，生成论文所需的全部图表和数据）：
    1. 加载已训练好的模型（从 saved_models/ 目录）
    2. 在测试集上全面评估每个模型
    3. 生成 EI 会议标准的高质量图表（300 DPI, 英文标签）
    4. 生成 LaTeX 格式的对比表格代码
    5. 生成消融实验结果表
    6. 生成 Grad-CAM 可解释性可视化
    7. 输出详细的纯文本分析报告

使用方法：
    python generate_paper_results.py

    可选参数：
    python generate_paper_results.py --data_dir ./data/garbage_classification
    python generate_paper_results.py --models_dir ./saved_models
    python generate_paper_results.py --no-gradcam      # 跳过Grad-CAM（更快）

输出目录：
    ./paper_results/
    ├── figures/          ← 所有论文用图（PNG, 300 DPI）
    ├── tables/           ← LaTeX 表格代码（.tex 文件）
    ├── data/             ← 原始数据（CSV 格式）
    └── report.txt        ← 完整文字版结果报告
============================================================================
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

# ============================================================================
# 全局样式配置 —— EI 会议标准
# ============================================================================
# 使用学术论文常用字体和配色
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# 学术论文配色方案（色盲友好）
COLORS = {
    'blue':    '#2166ac',
    'red':     '#b2182b',
    'green':   '#1b7837',
    'orange':  '#e08214',
    'purple':  '#7b3294',
    'cyan':    '#17becf',
    'gray':    '#636363',
}
MODEL_COLORS = [
    '#2166ac', '#b2182b', '#1b7837', '#e08214', '#7b3294',
    '#17becf', '#636363'
]

# 模型显示名称映射（论文中使用的正式名称）
MODEL_DISPLAY_NAMES = {
    'MobileNetV2':      'MobileNetV2',
    'MobileNetV2_CBAM': 'MobileNetV2-CBAM (Ours)',
    'VGG16':            'VGG16',
    'DenseNet121':      'DenseNet121',
    'EfficientNetB0':   'EfficientNet-B0',
    'ResNet50':         'ResNet50',
}


# ============================================================================
# 工具函数
# ============================================================================

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def format_params(n):
    """格式化参数量：3400000 -> '3.4M'"""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def get_display_name(model_name):
    """获取模型的论文显示名称"""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


class ReportWriter:
    """报告写入器：同时输出到控制台和文件"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []

    def write(self, text=""):
        print(text)
        self.lines.append(text)

    def section(self, title):
        sep = "=" * 70
        self.write(f"\n{sep}")
        self.write(f"  {title}")
        self.write(sep)

    def subsection(self, title):
        self.write(f"\n--- {title} ---")

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"\n[OK] 报告已保存: {self.filepath}")


# ============================================================================
# 数据准备
# ============================================================================

def setup_gpu():
    """配置 GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"检测到 {len(gpus)} 个 GPU")
    else:
        logger.info("未检测到 GPU，将使用 CPU")


def create_test_generator(data_dir, batch_size=32, img_size=224):
    """
    创建测试集数据生成器

    Args:
        data_dir: 数据根目录（包含 split/test 或直接包含类别子目录）
        batch_size: 批大小
        img_size: 图像尺寸

    Returns:
        test_generator: 测试数据生成器
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # 自动查找测试集目录
    possible_test_dirs = [
        os.path.join(data_dir, 'split', 'test'),
        os.path.join(data_dir, 'test'),
    ]
    test_dir = None
    for d in possible_test_dirs:
        if os.path.exists(d):
            test_dir = d
            break

    if test_dir is None:
        # 没有预划分的测试集，从原始数据创建
        logger.info("未找到预划分测试集，将自动划分数据...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data_loader import create_data_split
        split_dir = os.path.join(data_dir, 'split')
        create_data_split(data_dir, split_dir)
        test_dir = os.path.join(split_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    logger.info(f"测试集: {test_generator.samples} 张图片, "
                f"{test_generator.num_classes} 个类别")
    return test_generator


def find_saved_models(models_dir):
    """
    查找已保存的模型文件

    Returns:
        dict: {model_name: model_path}
    """
    model_files = {}
    if not os.path.exists(models_dir):
        logger.warning(f"模型目录不存在: {models_dir}")
        return model_files

    for f in os.listdir(models_dir):
        if f.endswith(('.keras', '.h5')):
            # 从文件名提取模型名
            # 例: MobileNetV2_best.keras -> MobileNetV2
            #     MobileNetV2_CBAM_final.keras -> MobileNetV2_CBAM
            name = f.replace('_best.keras', '').replace('_final.keras', '')
            name = name.replace('_best.h5', '').replace('_final.h5', '')
            name = name.replace('.keras', '').replace('.h5', '')
            model_files[name] = os.path.join(models_dir, f)

    return model_files


def load_model_safe(model_path, model_name):
    """
    安全加载模型，处理自定义层

    Args:
        model_path: 模型文件路径
        model_name: 模型名称

    Returns:
        model 或 None
    """
    # 注册自定义对象（CBAM、Focal Loss 等）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from models import ChannelAttention, SpatialAttention, CBAM, SEBlock
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'CBAM': CBAM,
            'SEBlock': SEBlock,
        }
    except ImportError:
        custom_objects = {}

    try:
        from trainer import FocalLoss
        custom_objects['FocalLoss'] = FocalLoss
    except ImportError:
        pass

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        logger.info(f"[OK] 已加载模型: {model_name} <- {model_path}")
        return model
    except Exception as e:
        logger.error(f"[FAIL] 加载模型失败 {model_name}: {e}")
        return None


# ============================================================================
# 模型评估
# ============================================================================

def evaluate_single_model(model, test_generator, model_name):
    """
    全面评估单个模型

    Returns:
        dict: 包含所有评估结果
    """
    logger.info(f"评估模型: {model_name}")

    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes[:len(y_pred)]
    class_names = list(test_generator.class_indices.keys())

    # 总体指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # macro 指标（不加权，更体现各类平衡性）
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 每类指标
    per_class_p, per_class_r, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # 模型参数信息
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 * 1024)

    # 推理速度
    test_generator.reset()
    batch = next(test_generator)
    sample_images = batch[0][:min(32, len(batch[0]))]

    # 预热
    _ = model.predict(sample_images[:1], verbose=0)

    # 单张推理
    single_times = []
    for img in sample_images:
        t0 = time.time()
        _ = model.predict(np.expand_dims(img, 0), verbose=0)
        single_times.append(time.time() - t0)

    inference_ms = np.mean(single_times) * 1000
    inference_std = np.std(single_times) * 1000
    fps = 1.0 / np.mean(single_times) if np.mean(single_times) > 0 else 0

    results = {
        'model_name': model_name,
        'display_name': get_display_name(model_name),
        # 总体指标（weighted）
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        # 总体指标（macro）
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        # 每类指标
        'per_class_precision': per_class_p,
        'per_class_recall': per_class_r,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        # 预测结果（用于绘图）
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'class_names': class_names,
        # 模型信息
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb,
        # 推理速度
        'inference_time_ms': inference_ms,
        'inference_time_std_ms': inference_std,
        'fps': fps,
    }

    logger.info(f"  Accuracy={acc:.4f}  F1={f1:.4f}  "
                f"Size={model_size_mb:.1f}MB  Inference={inference_ms:.1f}ms")
    return results


# ============================================================================
# 图表生成 —— EI 会议标准
# ============================================================================

# ---- Figure 1: 各模型混淆矩阵（归一化）----
def fig_confusion_matrices(all_results, output_dir):
    """为每个模型生成独立的归一化混淆矩阵，同时生成一张综合对比图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))
    n = len(all_results)

    for res in all_results:
        name = res['model_name']
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_names = res['class_names']

        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 8}
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {get_display_name(name)}')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        path = os.path.join(fig_dir, f'confusion_matrix_{name}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"  [Figure] 混淆矩阵 -> {path}")

    # 综合对比图（如果模型数 <= 6，放在同一张图里）
    if n <= 6:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, res in enumerate(all_results):
            cm = confusion_matrix(res['y_true'], res['y_pred'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(
                cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=res['class_names'],
                yticklabels=res['class_names'],
                ax=axes[idx], cbar=False,
                linewidths=0.3, linecolor='white',
                annot_kws={'size': 6}
            )
            axes[idx].set_title(get_display_name(res['model_name']), fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=9)
            axes[idx].set_ylabel('True', fontsize=9)
            axes[idx].tick_params(axis='x', rotation=45, labelsize=7)
            axes[idx].tick_params(axis='y', rotation=0, labelsize=7)

        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Normalized Confusion Matrices of All Models', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(fig_dir, 'confusion_matrices_all.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"  [Figure] 综合混淆矩阵 -> {path}")


# ---- Figure 2: 模型性能对比柱状图 ----
def fig_model_comparison_bars(all_results, output_dir):
    """准确率、F1、模型大小、推理时间四维度柱状图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))
    names = [get_display_name(r['model_name']) for r in all_results]
    n = len(names)
    x = np.arange(n)
    colors = MODEL_COLORS[:n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Accuracy
    vals = [r['accuracy'] for r in all_results]
    bars = axes[0, 0].bar(x, vals, color=colors, width=0.6, edgecolor='white')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('(a) Test Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    y_min = max(0, min(vals) - 0.05)
    axes[0, 0].set_ylim([y_min, min(1.0, max(vals) + 0.03)])
    for i, v in enumerate(vals):
        axes[0, 0].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=8,
                        fontweight='bold')

    # (b) F1-Score
    vals = [r['f1_score'] for r in all_results]
    axes[0, 1].bar(x, vals, color=colors, width=0.6, edgecolor='white')
    axes[0, 1].set_ylabel('Weighted F1-Score')
    axes[0, 1].set_title('(b) F1-Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    y_min = max(0, min(vals) - 0.05)
    axes[0, 1].set_ylim([y_min, min(1.0, max(vals) + 0.03)])
    for i, v in enumerate(vals):
        axes[0, 1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=8,
                        fontweight='bold')

    # (c) Model Size (MB)
    vals = [r['model_size_mb'] for r in all_results]
    axes[1, 0].bar(x, vals, color=colors, width=0.6, edgecolor='white')
    axes[1, 0].set_ylabel('Model Size (MB)')
    axes[1, 0].set_title('(c) Model Size')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    for i, v in enumerate(vals):
        axes[1, 0].text(i, v + max(vals) * 0.02, f'{v:.1f}', ha='center',
                        fontsize=8, fontweight='bold')

    # (d) Inference Time (ms)
    vals = [r['inference_time_ms'] for r in all_results]
    axes[1, 1].bar(x, vals, color=colors, width=0.6, edgecolor='white')
    axes[1, 1].set_ylabel('Inference Time (ms)')
    axes[1, 1].set_title('(d) Inference Time per Image')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + max(vals) * 0.02, f'{v:.1f}', ha='center',
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(fig_dir, 'model_comparison_bars.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] 模型对比柱状图 -> {path}")


# ---- Figure 3: Accuracy vs Efficiency 散点图 ----
def fig_accuracy_efficiency_scatter(all_results, output_dir):
    """横轴: 模型大小, 纵轴: 准确率, 气泡大小: 推理速度"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for idx, r in enumerate(all_results):
        size = max(r['inference_time_ms'] * 8, 60)
        ax.scatter(
            r['model_size_mb'], r['accuracy'],
            s=size, color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            alpha=0.8, edgecolors='black', linewidth=0.8,
            zorder=3, label=get_display_name(r['model_name'])
        )
        ax.annotate(
            get_display_name(r['model_name']),
            (r['model_size_mb'], r['accuracy']),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, alpha=0.9
        )

    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs. Model Size\n(bubble size = inference time)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'accuracy_vs_efficiency.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] 准确率-效率散点图 -> {path}")


# ---- Figure 4: 每类别 F1 对比雷达图 ----
def fig_per_class_radar(all_results, output_dir):
    """为所有模型绘制每类别 F1 雷达图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))

    class_names = all_results[0]['class_names']
    N = len(class_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, r in enumerate(all_results):
        values = list(r['per_class_f1']) + [r['per_class_f1'][0]]
        ax.plot(angles, values, 'o-', linewidth=1.5, markersize=4,
                color=MODEL_COLORS[idx % len(MODEL_COLORS)],
                label=get_display_name(r['model_name']))
        ax.fill(angles, values, alpha=0.05,
                color=MODEL_COLORS[idx % len(MODEL_COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.set_title('Per-class F1-Score Comparison', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'per_class_f1_radar.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] 每类别F1雷达图 -> {path}")


# ---- Figure 5: 每类别 P/R/F1 柱状图（最佳模型）----
def fig_per_class_bar(all_results, output_dir):
    """为最佳模型绘制每类别 Precision/Recall/F1 分组柱状图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))

    # 找到准确率最高的模型
    best = max(all_results, key=lambda r: r['accuracy'])
    class_names = best['class_names']
    N = len(class_names)
    x = np.arange(N)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5.5))

    ax.bar(x - width, best['per_class_precision'], width,
           label='Precision', color=COLORS['blue'], edgecolor='white')
    ax.bar(x, best['per_class_recall'], width,
           label='Recall', color=COLORS['green'], edgecolor='white')
    ax.bar(x + width, best['per_class_f1'], width,
           label='F1-Score', color=COLORS['red'], edgecolor='white')

    ax.set_ylabel('Score')
    ax.set_title(f'Per-class Metrics - {get_display_name(best["model_name"])}')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(fig_dir, f'per_class_metrics_{best["model_name"]}.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] 每类别指标柱状图 -> {path}")


# ---- Figure 6: 每类别 F1 热力图矩阵（所有模型 × 所有类别）----
def fig_f1_heatmap(all_results, output_dir):
    """模型 × 类别 的 F1 热力图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))
    class_names = all_results[0]['class_names']

    data = []
    model_names = []
    for r in all_results:
        data.append(r['per_class_f1'])
        model_names.append(get_display_name(r['model_name']))

    df = pd.DataFrame(data, index=model_names, columns=class_names)

    fig, ax = plt.subplots(figsize=(12, max(3, len(all_results) * 0.8 + 1.5)))
    sns.heatmap(
        df, annot=True, fmt='.3f', cmap='YlGnBu',
        linewidths=0.5, linecolor='white', ax=ax,
        cbar_kws={'shrink': 0.8, 'label': 'F1-Score'},
        annot_kws={'size': 9}
    )
    ax.set_title('F1-Score Heatmap: Models × Categories')
    ax.set_ylabel('Model')
    ax.set_xlabel('Category')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'f1_heatmap_all.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] F1热力图矩阵 -> {path}")


# ---- Figure 7: Grad-CAM 可视化 ----
def fig_gradcam(models_dict, test_generator, output_dir, num_samples=8):
    """为每个模型生成 Grad-CAM 热力图"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))

    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python 未安装，跳过 Grad-CAM 可视化")
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from evaluation import GradCAM
    except ImportError:
        logger.warning("无法导入 GradCAM 类，跳过")
        return

    class_names = list(test_generator.class_indices.keys())

    for model_name, model in models_dict.items():
        logger.info(f"  生成 Grad-CAM: {model_name}")
        try:
            gradcam = GradCAM(model)
        except Exception as e:
            logger.warning(f"  Grad-CAM 初始化失败 ({model_name}): {e}")
            continue

        test_generator.reset()
        batch = next(test_generator)
        images = batch[0][:num_samples]
        labels = batch[1][:num_samples]
        predictions = model.predict(images, verbose=0)

        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            img = images[i]
            true_idx = np.argmax(labels[i])
            pred_idx = np.argmax(predictions[i])

            try:
                heatmap = gradcam.compute_heatmap(np.expand_dims(img, 0), pred_idx)
                overlay = gradcam.overlay_heatmap(img, heatmap)
            except Exception:
                overlay = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img

            row = i // cols
            col = i % cols

            # 原图
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(img)
            ax_orig.set_title(f'True: {class_names[true_idx]}', fontsize=8)
            ax_orig.axis('off')

            # Grad-CAM
            ax_cam = axes[row, col * 2 + 1]
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if len(overlay.shape) == 3 else overlay
            ax_cam.imshow(overlay_rgb)
            conf = predictions[i][pred_idx]
            color = 'green' if true_idx == pred_idx else 'red'
            ax_cam.set_title(f'Pred: {class_names[pred_idx]} ({conf:.2f})',
                           fontsize=8, color=color)
            ax_cam.axis('off')

        # 隐藏多余格子
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')

        plt.suptitle(f'Grad-CAM Visualization - {get_display_name(model_name)}',
                     fontsize=13)
        plt.tight_layout()
        path = os.path.join(fig_dir, f'gradcam_{model_name}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"  [Figure] Grad-CAM -> {path}")


# ---- Figure 8: 预测样例展示 ----
def fig_prediction_samples(models_dict, test_generator, output_dir, num_images=12):
    """展示最佳模型的预测样例"""
    fig_dir = ensure_dir(os.path.join(output_dir, 'figures'))
    class_names = list(test_generator.class_indices.keys())

    # 选择第一个模型来做展示
    model_name = list(models_dict.keys())[0]
    model = models_dict[model_name]

    test_generator.reset()
    batch = next(test_generator)
    images = batch[0][:num_images]
    labels = batch[1][:num_images]
    predictions = model.predict(images, verbose=0)

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i in range(num_images):
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(predictions[i])
        conf = predictions[i][pred_idx]
        correct = true_idx == pred_idx

        axes_flat[i].imshow(images[i])
        axes_flat[i].set_title(
            f'True: {class_names[true_idx]}\n'
            f'Pred: {class_names[pred_idx]} ({conf:.2f})',
            fontsize=8, color='green' if correct else 'red'
        )
        axes_flat[i].axis('off')

    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.suptitle(f'Prediction Samples - {get_display_name(model_name)}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, f'prediction_samples_{model_name}.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  [Figure] 预测样例 -> {path}")


# ============================================================================
# LaTeX 表格生成
# ============================================================================

def gen_latex_comparison_table(all_results, output_dir):
    """Table 1: 模型性能对比总表"""
    table_dir = ensure_dir(os.path.join(output_dir, 'tables'))

    # 找到最佳值用于加粗
    best_acc = max(r['accuracy'] for r in all_results)
    best_f1 = max(r['f1_score'] for r in all_results)
    min_size = min(r['model_size_mb'] for r in all_results)
    min_time = min(r['inference_time_ms'] for r in all_results)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Performance Comparison of Different Models on the Garbage Classification Dataset}')
    lines.append(r'\label{tab:model_comparison}')
    lines.append(r'\begin{tabular}{l c c c c c c c}')
    lines.append(r'\toprule')
    lines.append(r'Model & Params & Size & Accuracy & Precision & Recall & F1-Score & Time \\')
    lines.append(r' & & (MB) & (\%) & (\%) & (\%) & (\%) & (ms) \\')
    lines.append(r'\midrule')

    for r in all_results:
        name = get_display_name(r['model_name']).replace('_', r'\_')
        params = format_params(r['total_params'])
        size = f"{r['model_size_mb']:.1f}"
        acc = f"{r['accuracy']*100:.2f}"
        prec = f"{r['precision']*100:.2f}"
        rec = f"{r['recall']*100:.2f}"
        f1_val = f"{r['f1_score']*100:.2f}"
        t = f"{r['inference_time_ms']:.1f}"

        # 最佳值加粗
        if r['accuracy'] == best_acc:
            acc = r'\textbf{' + acc + '}'
        if r['f1_score'] == best_f1:
            f1_val = r'\textbf{' + f1_val + '}'
        if r['model_size_mb'] == min_size:
            size = r'\textbf{' + size + '}'
        if r['inference_time_ms'] == min_time:
            t = r'\textbf{' + t + '}'

        lines.append(f'{name} & {params} & {size} & {acc} & {prec} & {rec} & {f1_val} & {t} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    path = os.path.join(table_dir, 'table_model_comparison.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    logger.info(f"  [Table] 模型对比表 -> {path}")
    return tex


def gen_latex_per_class_table(all_results, output_dir):
    """Table 2: 最佳模型的每类别详细指标"""
    table_dir = ensure_dir(os.path.join(output_dir, 'tables'))

    best = max(all_results, key=lambda r: r['accuracy'])
    class_names = best['class_names']

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Per-class Performance of ' +
                 get_display_name(best['model_name']).replace('_', r'\_') + '}')
    lines.append(r'\label{tab:per_class}')
    lines.append(r'\begin{tabular}{l c c c c}')
    lines.append(r'\toprule')
    lines.append(r'Category & Precision (\%) & Recall (\%) & F1-Score (\%) & Support \\')
    lines.append(r'\midrule')

    for i, cn in enumerate(class_names):
        p = f"{best['per_class_precision'][i]*100:.2f}"
        r_val = f"{best['per_class_recall'][i]*100:.2f}"
        f1_val = f"{best['per_class_f1'][i]*100:.2f}"
        sup = int(best['per_class_support'][i])
        lines.append(f'{cn} & {p} & {r_val} & {f1_val} & {sup} \\\\')

    lines.append(r'\midrule')
    lines.append(f'Weighted Avg & {best["precision"]*100:.2f} & '
                 f'{best["recall"]*100:.2f} & {best["f1_score"]*100:.2f} & '
                 f'{int(sum(best["per_class_support"]))} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    path = os.path.join(table_dir, 'table_per_class.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    logger.info(f"  [Table] 每类别指标表 -> {path}")
    return tex


def gen_latex_ablation_table(all_results, output_dir):
    """
    Table 3: 消融实验表
    如果有 MobileNetV2 和 MobileNetV2_CBAM，自动构建消融对比
    """
    table_dir = ensure_dir(os.path.join(output_dir, 'tables'))

    # 查找消融实验相关模型
    baseline = None
    cbam_model = None
    for r in all_results:
        if r['model_name'] == 'MobileNetV2':
            baseline = r
        elif r['model_name'] == 'MobileNetV2_CBAM':
            cbam_model = r

    if baseline is None:
        logger.warning("未找到 MobileNetV2 基线模型，跳过消融实验表")
        return ""

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation Study Results}')
    lines.append(r'\label{tab:ablation}')
    lines.append(r'\begin{tabular}{l c c c c c}')
    lines.append(r'\toprule')
    lines.append(r'Configuration & CBAM & Accuracy (\%) & Precision (\%) & Recall (\%) & F1-Score (\%) \\')
    lines.append(r'\midrule')

    # Baseline
    lines.append(
        f'Baseline (MobileNetV2) & \\xmark & '
        f'{baseline["accuracy"]*100:.2f} & {baseline["precision"]*100:.2f} & '
        f'{baseline["recall"]*100:.2f} & {baseline["f1_score"]*100:.2f} \\\\'
    )

    # + CBAM
    if cbam_model:
        acc_diff = (cbam_model["accuracy"] - baseline["accuracy"]) * 100
        lines.append(
            f'+ CBAM (Ours) & \\cmark & '
            f'\\textbf{{{cbam_model["accuracy"]*100:.2f}}} & '
            f'\\textbf{{{cbam_model["precision"]*100:.2f}}} & '
            f'\\textbf{{{cbam_model["recall"]*100:.2f}}} & '
            f'\\textbf{{{cbam_model["f1_score"]*100:.2f}}} \\\\'
        )
        lines.append(r'\midrule')
        lines.append(f'Improvement & & +{acc_diff:.2f} & '
                     f'+{(cbam_model["precision"]-baseline["precision"])*100:.2f} & '
                     f'+{(cbam_model["recall"]-baseline["recall"])*100:.2f} & '
                     f'+{(cbam_model["f1_score"]-baseline["f1_score"])*100:.2f} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    path = os.path.join(table_dir, 'table_ablation.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    logger.info(f"  [Table] 消融实验表 -> {path}")
    return tex


# ============================================================================
# CSV 数据导出
# ============================================================================

def export_csv_data(all_results, output_dir):
    """导出原始数据为 CSV，方便后续处理"""
    data_dir = ensure_dir(os.path.join(output_dir, 'data'))

    # 总表
    rows = []
    for r in all_results:
        rows.append({
            'Model': r['model_name'],
            'Display_Name': get_display_name(r['model_name']),
            'Accuracy': r['accuracy'],
            'Precision_weighted': r['precision'],
            'Recall_weighted': r['recall'],
            'F1_weighted': r['f1_score'],
            'Precision_macro': r['precision_macro'],
            'Recall_macro': r['recall_macro'],
            'F1_macro': r['f1_macro'],
            'Total_Params': r['total_params'],
            'Trainable_Params': r['trainable_params'],
            'Model_Size_MB': r['model_size_mb'],
            'Inference_Time_ms': r['inference_time_ms'],
            'Inference_Std_ms': r['inference_time_std_ms'],
            'FPS': r['fps'],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(data_dir, 'model_comparison.csv')
    df.to_csv(path, index=False)
    logger.info(f"  [CSV] 模型对比数据 -> {path}")

    # 每类别指标
    for r in all_results:
        class_names = r['class_names']
        rows_cls = []
        for i, cn in enumerate(class_names):
            rows_cls.append({
                'Category': cn,
                'Precision': r['per_class_precision'][i],
                'Recall': r['per_class_recall'][i],
                'F1_Score': r['per_class_f1'][i],
                'Support': int(r['per_class_support'][i]),
            })
        df_cls = pd.DataFrame(rows_cls)
        path = os.path.join(data_dir, f'per_class_{r["model_name"]}.csv')
        df_cls.to_csv(path, index=False)

    logger.info(f"  [CSV] 每类别数据已导出")


# ============================================================================
# 文字报告生成
# ============================================================================

def generate_text_report(all_results, output_dir):
    """生成详细的纯文本分析报告"""
    rpt = ReportWriter(os.path.join(output_dir, 'report.txt'))

    rpt.section("智能垃圾分类系统 - 实验结果报告")
    rpt.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.write(f"模型数量: {len(all_results)}")
    rpt.write(f"测试集大小: {len(all_results[0]['y_true'])} 张图片")
    rpt.write(f"类别数量: {len(all_results[0]['class_names'])}")
    rpt.write(f"类别列表: {', '.join(all_results[0]['class_names'])}")

    # ---- 1. 总体结果对比 ----
    rpt.section("1. 模型性能对比总表")
    rpt.write("")
    header = f"{'模型':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'大小(MB)':>10} {'推理(ms)':>10} {'FPS':>8}"
    rpt.write(header)
    rpt.write("-" * len(header))

    for r in all_results:
        line = (f"{r['model_name']:<25} "
                f"{r['accuracy']:>10.4f} "
                f"{r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} "
                f"{r['f1_score']:>10.4f} "
                f"{r['model_size_mb']:>10.1f} "
                f"{r['inference_time_ms']:>10.1f} "
                f"{r['fps']:>8.1f}")
        rpt.write(line)

    # ---- 2. 最佳模型分析 ----
    rpt.section("2. 最佳模型分析")
    best_acc = max(all_results, key=lambda x: x['accuracy'])
    best_f1 = max(all_results, key=lambda x: x['f1_score'])
    fastest = min(all_results, key=lambda x: x['inference_time_ms'])
    smallest = min(all_results, key=lambda x: x['model_size_mb'])

    rpt.write(f"\n准确率最高: {best_acc['model_name']}")
    rpt.write(f"  Accuracy = {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)")
    rpt.write(f"  F1-Score = {best_acc['f1_score']:.4f}")

    rpt.write(f"\nF1最高: {best_f1['model_name']}")
    rpt.write(f"  F1-Score = {best_f1['f1_score']:.4f}")

    rpt.write(f"\n推理最快: {fastest['model_name']}")
    rpt.write(f"  推理时间 = {fastest['inference_time_ms']:.1f} ms/image")
    rpt.write(f"  FPS = {fastest['fps']:.1f}")

    rpt.write(f"\n体积最小: {smallest['model_name']}")
    rpt.write(f"  模型大小 = {smallest['model_size_mb']:.1f} MB")
    rpt.write(f"  参数量 = {format_params(smallest['total_params'])}")

    # ---- 3. 消融实验分析 ----
    baseline = next((r for r in all_results if r['model_name'] == 'MobileNetV2'), None)
    cbam = next((r for r in all_results if r['model_name'] == 'MobileNetV2_CBAM'), None)

    if baseline and cbam:
        rpt.section("3. 消融实验分析 (CBAM 注意力机制效果)")

        rpt.write(f"\n{'指标':<20} {'MobileNetV2':>15} {'MobileNetV2_CBAM':>18} {'提升':>10}")
        rpt.write("-" * 65)

        for metric, key in [('Accuracy', 'accuracy'), ('Precision', 'precision'),
                            ('Recall', 'recall'), ('F1-Score', 'f1_score')]:
            b_val = baseline[key]
            c_val = cbam[key]
            diff = c_val - b_val
            symbol = "+" if diff >= 0 else ""
            rpt.write(f"{metric:<20} {b_val:>15.4f} {c_val:>18.4f} {symbol}{diff:>9.4f}")

        rpt.write(f"\n{'模型大小(MB)':<20} {baseline['model_size_mb']:>15.1f} "
                  f"{cbam['model_size_mb']:>18.1f} "
                  f"+{cbam['model_size_mb']-baseline['model_size_mb']:>8.1f}")
        rpt.write(f"{'推理时间(ms)':<20} {baseline['inference_time_ms']:>15.1f} "
                  f"{cbam['inference_time_ms']:>18.1f} "
                  f"+{cbam['inference_time_ms']-baseline['inference_time_ms']:>8.1f}")

        rpt.write("\n分析结论:")
        acc_diff = (cbam['accuracy'] - baseline['accuracy']) * 100
        if acc_diff > 0:
            rpt.write(f"  - CBAM 注意力机制使准确率提升了 {acc_diff:.2f} 个百分点")
            rpt.write(f"  - 模型参数增量仅 "
                      f"{format_params(cbam['total_params'] - baseline['total_params'])}，"
                      f"开销很小")
            rpt.write(f"  - 推理时间增加 "
                      f"{cbam['inference_time_ms'] - baseline['inference_time_ms']:.1f} ms，"
                      f"几乎可忽略")
        else:
            rpt.write(f"  - 在本次实验中 CBAM 未带来准确率提升 ({acc_diff:.2f}%)")
            rpt.write(f"  - 可能需要调整超参数或训练更多轮次")

    # ---- 4. 每类别详细分析 ----
    rpt.section("4. 每类别详细分析")

    for r in all_results:
        rpt.subsection(f"模型: {r['model_name']}")
        class_names = r['class_names']
        rpt.write(f"\n{'类别':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'样本数':>8}")
        rpt.write("-" * 55)

        for i, cn in enumerate(class_names):
            rpt.write(f"{cn:<15} "
                      f"{r['per_class_precision'][i]:>10.4f} "
                      f"{r['per_class_recall'][i]:>10.4f} "
                      f"{r['per_class_f1'][i]:>10.4f} "
                      f"{int(r['per_class_support'][i]):>8d}")

        rpt.write(f"\n{'加权平均':<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1_score']:>10.4f} {int(sum(r['per_class_support'])):>8d}")

        # 找出最弱类别
        worst_idx = np.argmin(r['per_class_f1'])
        best_idx = np.argmax(r['per_class_f1'])
        rpt.write(f"\n  最佳类别: {class_names[best_idx]} "
                  f"(F1={r['per_class_f1'][best_idx]:.4f})")
        rpt.write(f"  最弱类别: {class_names[worst_idx]} "
                  f"(F1={r['per_class_f1'][worst_idx]:.4f})")

    # ---- 5. 混淆矩阵分析（最佳模型的 Top 错分对）----
    rpt.section("5. 错分分析 (最常见的错误分类)")
    best = max(all_results, key=lambda x: x['accuracy'])
    cm = confusion_matrix(best['y_true'], best['y_pred'])
    class_names = best['class_names']

    # 将对角线置零，找到最大的错分值
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)

    rpt.write(f"\n模型: {best['model_name']}")
    rpt.write(f"{'True → Predicted':<35} {'Count':>8} {'占该类比例':>12}")
    rpt.write("-" * 57)

    # 找前 10 个最常见的错分
    flat_indices = np.argsort(cm_off_diag.ravel())[::-1]
    count = 0
    for flat_idx in flat_indices:
        if count >= 10:
            break
        true_class = flat_idx // len(class_names)
        pred_class = flat_idx % len(class_names)
        error_count = cm_off_diag[true_class, pred_class]
        if error_count == 0:
            break
        total = cm[true_class].sum()
        ratio = error_count / total if total > 0 else 0
        rpt.write(f"{class_names[true_class]:>15} → {class_names[pred_class]:<15} "
                  f"{error_count:>8d} {ratio:>11.1%}")
        count += 1

    # ---- 6. 参数效率分析 ----
    rpt.section("6. 参数效率分析")
    rpt.write(f"\n{'模型':<25} {'参数量':>12} {'可训练':>12} {'大小(MB)':>10} "
              f"{'Acc/MB':>10}")
    rpt.write("-" * 70)

    for r in all_results:
        acc_per_mb = r['accuracy'] / r['model_size_mb'] if r['model_size_mb'] > 0 else 0
        rpt.write(f"{r['model_name']:<25} "
                  f"{format_params(r['total_params']):>12} "
                  f"{format_params(r['trainable_params']):>12} "
                  f"{r['model_size_mb']:>10.1f} "
                  f"{acc_per_mb:>10.4f}")

    rpt.write("\n说明: Acc/MB = 准确率 / 模型大小，数值越大表示参数效率越高。")

    # ---- 7. 论文写作建议 ----
    rpt.section("7. 论文写作要点建议")
    rpt.write("""
基于以上实验结果，论文中应重点强调：

1. 【方法有效性】
   - 对比表（Table 1）展示了所提方案在多个指标上的表现
   - 使用消融实验（Table 3）量化了 CBAM 的贡献

2. 【关键图表】
   - Fig. model_comparison_bars.png  → 四维度对比柱状图
   - Fig. confusion_matrix_{best}.png → 最佳模型的混淆矩阵
   - Fig. per_class_f1_radar.png      → 各模型每类别 F1 雷达图
   - Fig. accuracy_vs_efficiency.png  → 准确率-效率权衡图
   - Fig. gradcam_{model}.png         → Grad-CAM 可解释性图
   - Fig. f1_heatmap_all.png          → 模型×类别 F1 热力图

3. 【讨论要点】
   - 分析哪些类别容易混淆（见错分分析表）
   - 分析轻量级模型（MobileNetV2）相比重量级（VGG16）的效率优势
   - 讨论 CBAM 对不同类别的提升差异
   - 讨论 Grad-CAM 展示的模型关注区域是否合理

4. 【LaTeX 表格】
   所有 .tex 文件已生成在 tables/ 目录，直接 \\input{} 即可使用
""")

    rpt.save()


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='论文结果生成器 - 一键生成 EI 会议标准的图表和数据'
    )
    parser.add_argument('--data_dir', type=str,
                        default='./data/garbage_classification',
                        help='数据集根目录')
    parser.add_argument('--models_dir', type=str,
                        default='./saved_models',
                        help='已保存模型的目录')
    parser.add_argument('--output_dir', type=str,
                        default='./paper_results',
                        help='论文结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='评估时的 batch size')
    parser.add_argument('--no-gradcam', action='store_true',
                        help='跳过 Grad-CAM 可视化（更快）')
    args = parser.parse_args()

    # ==============================
    # 0. 环境准备
    # ==============================
    print("\n" + "=" * 70)
    print("  论文结果生成器 (Paper Results Generator)")
    print("  智能垃圾分类系统 - Intelligent Waste Classification System")
    print("=" * 70)
    print(f"  数据目录:   {args.data_dir}")
    print(f"  模型目录:   {args.models_dir}")
    print(f"  输出目录:   {args.output_dir}")
    print(f"  Grad-CAM:   {'跳过' if args.no_gradcam else '生成'}")
    print("=" * 70 + "\n")

    setup_gpu()
    ensure_dir(args.output_dir)

    start_time = time.time()

    # ==============================
    # 1. 查找并加载模型
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 1/6: 加载已训练的模型")
    print("=" * 70)

    model_files = find_saved_models(args.models_dir)

    if not model_files:
        print(f"\n[错误] 在 {args.models_dir} 中未找到任何模型文件。")
        print("请确保已运行训练脚本，模型保存在 saved_models/ 目录中。")
        print("支持的格式: .keras, .h5")
        print(f"\n当前 {args.models_dir} 目录内容:")
        if os.path.exists(args.models_dir):
            for f in os.listdir(args.models_dir):
                print(f"  {f}")
        else:
            print("  (目录不存在)")
        sys.exit(1)

    print(f"\n找到 {len(model_files)} 个模型文件:")
    for name, path in model_files.items():
        print(f"  {name:<25} <- {path}")

    models_dict = OrderedDict()
    for name, path in model_files.items():
        model = load_model_safe(path, name)
        if model is not None:
            models_dict[name] = model

    if not models_dict:
        print("\n[错误] 没有成功加载任何模型，请检查模型文件。")
        sys.exit(1)

    print(f"\n成功加载 {len(models_dict)} 个模型")

    # ==============================
    # 2. 创建测试集
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 2/6: 准备测试数据集")
    print("=" * 70)

    test_generator = create_test_generator(args.data_dir, args.batch_size)

    # ==============================
    # 3. 评估所有模型
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 3/6: 评估所有模型")
    print("=" * 70)

    all_results = []
    for name, model in models_dict.items():
        results = evaluate_single_model(model, test_generator, name)
        all_results.append(results)

    # ==============================
    # 4. 生成图表
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 4/6: 生成 EI 标准论文图表 (300 DPI)")
    print("=" * 70)

    fig_confusion_matrices(all_results, args.output_dir)
    fig_model_comparison_bars(all_results, args.output_dir)
    fig_accuracy_efficiency_scatter(all_results, args.output_dir)
    fig_per_class_radar(all_results, args.output_dir)
    fig_per_class_bar(all_results, args.output_dir)
    fig_f1_heatmap(all_results, args.output_dir)
    fig_prediction_samples(models_dict, test_generator, args.output_dir)

    if not args.no_gradcam:
        fig_gradcam(models_dict, test_generator, args.output_dir)

    # ==============================
    # 5. 生成 LaTeX 表格
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 5/6: 生成 LaTeX 表格代码")
    print("=" * 70)

    tex_comparison = gen_latex_comparison_table(all_results, args.output_dir)
    tex_per_class = gen_latex_per_class_table(all_results, args.output_dir)
    tex_ablation = gen_latex_ablation_table(all_results, args.output_dir)

    # 输出一份完整的 LaTeX 代码到控制台
    print("\n" + "-" * 40)
    print("Table 1: 模型性能对比")
    print("-" * 40)
    print(tex_comparison)
    print("\n" + "-" * 40)
    print("Table 2: 每类别指标")
    print("-" * 40)
    print(tex_per_class)
    if tex_ablation:
        print("\n" + "-" * 40)
        print("Table 3: 消融实验")
        print("-" * 40)
        print(tex_ablation)

    # ==============================
    # 6. 导出数据并生成报告
    # ==============================
    print("\n" + "=" * 70)
    print("  步骤 6/6: 导出数据与生成报告")
    print("=" * 70)

    export_csv_data(all_results, args.output_dir)
    generate_text_report(all_results, args.output_dir)

    # ==============================
    # 完成
    # ==============================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  全部完成!")
    print("=" * 70)
    print(f"\n  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"\n  输出目录: {args.output_dir}/")
    print(f"  ├── figures/           ← 论文用图 (PNG, 300 DPI)")
    print(f"  │   ├── confusion_matrix_*.png     混淆矩阵")
    print(f"  │   ├── confusion_matrices_all.png  综合混淆矩阵")
    print(f"  │   ├── model_comparison_bars.png   四维度对比柱状图")
    print(f"  │   ├── accuracy_vs_efficiency.png  准确率-效率散点图")
    print(f"  │   ├── per_class_f1_radar.png      F1 雷达图")
    print(f"  │   ├── per_class_metrics_*.png     每类别指标")
    print(f"  │   ├── f1_heatmap_all.png          F1 热力图矩阵")
    print(f"  │   ├── gradcam_*.png               Grad-CAM 可视化")
    print(f"  │   └── prediction_samples_*.png    预测样例")
    print(f"  ├── tables/            ← LaTeX 表格 (.tex)")
    print(f"  │   ├── table_model_comparison.tex  模型对比表")
    print(f"  │   ├── table_per_class.tex         每类别指标表")
    print(f"  │   └── table_ablation.tex          消融实验表")
    print(f"  ├── data/              ← CSV 原始数据")
    print(f"  │   ├── model_comparison.csv        总对比数据")
    print(f"  │   └── per_class_*.csv             每模型每类别数据")
    print(f"  └── report.txt         ← 完整文字版结果报告")
    print()
    print("  下一步:")
    print("  1. 查看 report.txt 获取详细分析")
    print("  2. 将 figures/ 中的图片插入论文")
    print("  3. 将 tables/*.tex 用 \\input{} 导入 LaTeX")
    print("=" * 70)


if __name__ == '__main__':
    main()
