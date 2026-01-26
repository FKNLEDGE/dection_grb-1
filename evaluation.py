"""
评估与可视化模块
Evaluation and Visualization Module

包含:
1. 模型评估
2. 混淆矩阵绘制
3. 训练曲线绘制
4. 性能对比表格
5. 推理速度测试
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model

from config import OUTPUT_DIR, CLASS_NAMES, ensure_dir

# 配置日志
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(
    model: Model,
    test_generator: Any,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    全面评估模型性能
    Comprehensive model evaluation

    Args:
        model: 训练好的模型
        test_generator: 测试数据生成器
        model_name: 模型名称

    Returns:
        dict: 评估结果
    """
    print("\n" + "="*60)
    print(f"评估模型: {model_name or model.name}")
    print("="*60)

    # 获取预测结果
    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes

    # 获取类别名称
    class_names = list(test_generator.class_indices.keys())

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 打印结果
    print(f"\n准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")

    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'class_names': class_names
    }

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: str = OUTPUT_DIR,
    figsize: Tuple[int, int] = (12, 10)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    绘制混淆矩阵
    Plot confusion matrix

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        model_name: 模型名称
        output_dir: 输出目录
        figsize: 图像大小

    Returns:
        Tuple: (confusion_matrix, normalized_confusion_matrix)
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))

    # 绘制原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{model_name} - Confusion Matrix (Count)', fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 绘制归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()

    # 保存图片
    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    save_path = os.path.join(model_output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存: {save_path}")

    return cm, cm_normalized


def plot_training_curves(
    history: Any,
    model_name: str,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    绘制训练曲线（Accuracy和Loss）
    Plot training curves (Accuracy and Loss)

    Args:
        history: 训练历史
        model_name: 模型名称
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history['accuracy']) + 1)

    # 绘制准确率曲线
    axes[0].plot(epochs, history.history['accuracy'], 'b-',
                 label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-',
                 label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy Curves', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # 绘制损失曲线
    axes[1].plot(epochs, history.history['loss'], 'b-',
                 label='Training Loss', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'], 'r-',
                 label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss Curves', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    save_path = os.path.join(model_output_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存: {save_path}")


def measure_inference_time(
    model: Model,
    test_generator: Any,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    测量模型推理时间
    Measure model inference time

    Args:
        model: 模型
        test_generator: 测试数据生成器
        num_samples: 测试样本数

    Returns:
        dict: 推理时间统计
    """
    print(f"\n测量推理时间 (样本数: {num_samples})...")

    # 获取一些测试图像
    test_generator.reset()
    batch = next(test_generator)
    images = batch[0][:min(num_samples, len(batch[0]))]

    # 预热
    _ = model.predict(images[:1], verbose=0)

    # 测量单张图片推理时间
    single_times = []
    for img in images:
        start = time.time()
        _ = model.predict(np.expand_dims(img, 0), verbose=0)
        single_times.append(time.time() - start)

    # 测量批量推理时间
    start = time.time()
    _ = model.predict(images, verbose=0)
    batch_time = time.time() - start

    results = {
        'single_image_mean_ms': np.mean(single_times) * 1000,
        'single_image_std_ms': np.std(single_times) * 1000,
        'batch_total_ms': batch_time * 1000,
        'batch_per_image_ms': batch_time / len(images) * 1000,
        'fps': 1 / np.mean(single_times)
    }

    print(f"单张图片平均推理时间: {results['single_image_mean_ms']:.2f} "
          f"± {results['single_image_std_ms']:.2f} ms")
    print(f"批量推理每张图片: {results['batch_per_image_ms']:.2f} ms")
    print(f"FPS: {results['fps']:.1f}")

    return results


def compare_models(
    results_list: List[Dict[str, Any]],
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """
    生成模型对比表格和图表
    Generate model comparison table and charts

    Args:
        results_list: 各模型结果列表，每个元素包含:
            - model_name: 模型名称
            - accuracy: 准确率
            - precision: 精确率
            - recall: 召回率
            - f1_score: F1分数
            - model_size_mb: 模型大小(MB)
            - inference_time_ms: 推理时间(ms)
            - training_time_min: 训练时间(分钟)
        output_dir: 输出目录

    Returns:
        DataFrame: 对比结果
    """
    # 创建DataFrame
    df = pd.DataFrame(results_list)

    # 设置显示格式
    df_display = df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1_score']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')

    print("\n" + "="*80)
    print("模型性能对比表 (Model Performance Comparison)")
    print("="*80)
    print(df_display.to_string(index=False))
    print("="*80)

    # 保存为CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n对比表格已保存: {csv_path}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = df['model_name'].tolist()
    x = np.arange(len(model_names))
    width = 0.6

    # 1. 准确率对比
    axes[0, 0].bar(x, df['accuracy'], width, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].set_ylim([0.8, 1.0])
    for i, v in enumerate(df['accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

    # 2. 模型大小对比
    if 'model_size_mb' in df.columns:
        axes[0, 1].bar(x, df['model_size_mb'], width,
                       color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[0, 1].set_ylabel('Model Size (MB)')
        axes[0, 1].set_title('Model Size Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        for i, v in enumerate(df['model_size_mb']):
            axes[0, 1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

    # 3. 推理时间对比
    if 'inference_time_ms' in df.columns:
        axes[1, 0].bar(x, df['inference_time_ms'], width,
                       color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[1, 0].set_ylabel('Inference Time (ms)')
        axes[1, 0].set_title('Inference Time Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        for i, v in enumerate(df['inference_time_ms']):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    # 4. F1分数对比
    axes[1, 1].bar(x, df['f1_score'], width, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names)
    axes[1, 1].set_ylim([0.8, 1.0])
    for i, v in enumerate(df['f1_score']):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比图表已保存: {save_path}")

    return df


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    绘制每个类别的精确率、召回率、F1分数
    Plot per-class precision, recall, and F1 score

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        model_name: 模型名称
        output_dir: 输出目录
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    save_path = os.path.join(model_output_dir, f'{model_name}_per_class_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"类别指标图已保存: {save_path}")


def visualize_predictions(
    model: Model,
    test_generator: Any,
    model_name: str,
    num_images: int = 16,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    可视化部分预测结果
    Visualize some prediction results

    Args:
        model: 模型
        test_generator: 测试数据生成器
        model_name: 模型名称
        num_images: 可视化图像数量
        output_dir: 输出目录
    """
    test_generator.reset()
    batch = next(test_generator)
    images, labels = batch[0][:num_images], batch[1][:num_images]

    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    class_names = list(test_generator.class_indices.keys())

    # 绘制
    n_cols = 4
    n_rows = (num_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(images[i])
        true_label = class_names[true_classes[i]]
        pred_label = class_names[pred_classes[i]]
        confidence = predictions[i][pred_classes[i]]

        color = 'green' if true_classes[i] == pred_classes[i] else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
            color=color, fontsize=10
        )
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{model_name} - Prediction Samples', fontsize=14)
    plt.tight_layout()

    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    save_path = os.path.join(model_output_dir, f'{model_name}_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"预测可视化已保存: {save_path}")


def generate_latex_table(results_list: List[Dict[str, Any]]) -> str:
    """
    生成LaTeX格式的对比表格（用于论文）
    Generate LaTeX format comparison table for paper

    Args:
        results_list: 结果列表

    Returns:
        str: LaTeX表格代码
    """
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Different Models}
\\label{tab:comparison}
\\begin{tabular}{lcccccc}
\\toprule
Model & Accuracy & Precision & Recall & F1 & Size(MB) & Time(ms) \\\\
\\midrule
"""
    for r in results_list:
        model_size = r.get('model_size_mb', 0)
        inference_time = r.get('inference_time_ms', 0)
        latex += f"{r['model_name']} & {r['accuracy']:.4f} & {r['precision']:.4f} & "
        latex += f"{r['recall']:.4f} & {r['f1_score']:.4f} & "
        latex += f"{model_size:.1f} & {inference_time:.1f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    print("\nLaTeX表格代码:")
    print(latex)
    return latex
