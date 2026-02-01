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

# 尝试导入scipy用于统计检验
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy未安装，统计显著性检验功能将不可用")

# 尝试导入cv2用于Grad-CAM
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python未安装，Grad-CAM可视化功能将不可用")

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


# ==================== Grad-CAM 可视化 ====================

class GradCAM:
    """
    Grad-CAM: 梯度加权类激活映射
    Grad-CAM: Gradient-weighted Class Activation Mapping

    用于可视化卷积神经网络的决策过程，帮助理解模型关注的区域。

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017
        https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: Model, layer_name: Optional[str] = None):
        """
        初始化 Grad-CAM

        Args:
            model: 训练好的模型
            layer_name: 要可视化的卷积层名称。如果为None，自动选择最后一个卷积层
        """
        self.model = model
        self.layer_name = layer_name

        # 如果未指定层名，自动找最后一个卷积层
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()

        if self.layer_name is None:
            raise ValueError("无法找到卷积层。请手动指定layer_name")

        logger.info(f"Grad-CAM将使用层: {self.layer_name}")

        # 创建梯度模型
        try:
            conv_layer = model.get_layer(self.layer_name)
            self.grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output]
            )
        except Exception as e:
            raise ValueError(f"无法创建Grad-CAM模型: {e}")

    def _find_last_conv_layer(self) -> Optional[str]:
        """自动查找最后一个卷积层"""
        for layer in reversed(self.model.layers):
            # 检查是否是卷积层
            if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

            # 如果是嵌套模型，递归查找
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower() or isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name

        return None

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        计算热力图
        Compute heatmap

        Args:
            image: 输入图像 [1, H, W, C] 或 [H, W, C]
            class_idx: 目标类别索引。如果为None，使用预测类别
            eps: 防止除零的小常数

        Returns:
            heatmap: 热力图 [H, W]，值范围 [0, 1]
        """
        # 确保图像维度正确
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # 使用GradientTape记录梯度
        with tf.GradientTape() as tape:
            # 前向传播
            conv_output, predictions = self.grad_model(image)

            # 如果未指定类别，使用预测类别
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            # 获取目标类别的输出
            class_output = predictions[:, class_idx]

        # 计算梯度
        grads = tape.gradient(class_output, conv_output)

        # 全局平均池化，获取每个通道的权重
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 获取卷积输出
        conv_output = conv_output[0]

        # 加权求和
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU激活（只保留正值）
        heatmap = tf.maximum(heatmap, 0)

        # 归一化到 [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > eps:
            heatmap = heatmap / heatmap_max

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = None
    ) -> np.ndarray:
        """
        将热力图叠加到原图
        Overlay heatmap on original image

        Args:
            image: 原始图像 [H, W, C]，值范围 [0, 1] 或 [0, 255]
            heatmap: 热力图 [h, w]，值范围 [0, 1]
            alpha: 热力图透明度
            colormap: OpenCV colormap，默认为 COLORMAP_JET

        Returns:
            superimposed: 叠加后的图像 [H, W, C]
        """
        if not CV2_AVAILABLE:
            logger.warning("opencv-python未安装，无法叠加热力图")
            return image

        if colormap is None:
            colormap = cv2.COLORMAP_JET

        # 调整热力图大小以匹配原图
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # 转换为0-255范围
        heatmap = np.uint8(255 * heatmap)

        # 应用颜色映射
        heatmap = cv2.applyColorMap(heatmap, colormap)

        # 确保原图是0-255范围
        if image.max() <= 1.0:
            image = np.uint8(255 * image)

        # 叠加
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return superimposed


def visualize_gradcam(
    model: Model,
    test_generator: Any,
    class_names: List[str],
    model_name: str,
    num_samples: int = 12,
    output_dir: str = OUTPUT_DIR,
    layer_name: Optional[str] = None
) -> None:
    """
    批量生成Grad-CAM可视化
    Batch generate Grad-CAM visualizations

    Args:
        model: 训练好的模型
        test_generator: 测试数据生成器
        class_names: 类别名称列表
        model_name: 模型名称
        num_samples: 可视化样本数量
        output_dir: 输出目录
        layer_name: 目标卷积层名称（可选）
    """
    if not CV2_AVAILABLE:
        logger.warning("opencv-python未安装，跳过Grad-CAM可视化")
        return

    print(f"\n生成 {model_name} 的 Grad-CAM 可视化...")

    try:
        # 创建 Grad-CAM 对象
        gradcam = GradCAM(model, layer_name=layer_name)
    except Exception as e:
        logger.error(f"Grad-CAM初始化失败: {e}")
        return

    # 获取测试样本
    test_generator.reset()
    batch = next(test_generator)
    images, labels = batch[0][:num_samples], batch[1][:num_samples]

    # 获取预测
    predictions = model.predict(images, verbose=0)

    # 创建子图
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i in range(num_samples):
        img = images[i]
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(predictions[i])

        # 计算热力图
        heatmap = gradcam.compute_heatmap(np.expand_dims(img, 0), pred_idx)

        # 叠加热力图
        overlay = gradcam.overlay_heatmap(img, heatmap)

        # 显示
        axes[i].imshow(overlay)

        # 设置标题颜色（正确=绿色，错误=红色）
        color = 'green' if true_idx == pred_idx else 'red'
        conf = predictions[i][pred_idx]
        axes[i].set_title(
            f'True: {class_names[true_idx]}\n'
            f'Pred: {class_names[pred_idx]} ({conf:.2f})',
            color=color, fontsize=9
        )
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{model_name} - Grad-CAM Visualization', fontsize=14)
    plt.tight_layout()

    # 保存
    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    save_path = os.path.join(model_output_dir, f'{model_name}_gradcam.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Grad-CAM可视化已保存: {save_path}")


# ==================== 统计显著性检验 ====================

def statistical_significance_test(
    results1: List[float],
    results2: List[float],
    model1_name: str,
    model2_name: str,
    test_type: str = 'paired_t'
) -> Dict[str, float]:
    """
    统计显著性检验
    Statistical Significance Test

    比较两个模型的性能差异是否显著

    Args:
        results1: 模型1的K折结果列表
        results2: 模型2的K折结果列表
        model1_name: 模型1名称
        model2_name: 模型2名称
        test_type: 检验类型 ('paired_t', 'wilcoxon')

    Returns:
        dict: 包含统计量和p值
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy未安装，无法进行统计显著性检验")
        return {}

    results1 = np.array(results1)
    results2 = np.array(results2)

    if len(results1) != len(results2):
        raise ValueError(f"结果长度不匹配: {len(results1)} vs {len(results2)}")

    print(f"\n{'='*70}")
    print(f"统计显著性检验: {model1_name} vs {model2_name}")
    print(f"{'='*70}")

    if test_type == 'paired_t':
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(results1, results2)
        test_name = "配对t检验 (Paired t-test)"
        statistic_name = "t统计量"

    elif test_type == 'wilcoxon':
        # Wilcoxon符号秩检验（非参数）
        try:
            w_stat, p_value = stats.wilcoxon(results1, results2)
            t_stat = w_stat
            test_name = "Wilcoxon符号秩检验 (Wilcoxon signed-rank test)"
            statistic_name = "W统计量"
        except Exception as e:
            logger.warning(f"Wilcoxon检验失败: {e}，使用t检验")
            t_stat, p_value = stats.ttest_rel(results1, results2)
            test_name = "配对t检验 (Paired t-test)"
            statistic_name = "t统计量"

    else:
        raise ValueError(f"未知的检验类型: {test_type}")

    # 打印结果
    print(f"检验方法: {test_name}")
    print(f"{model1_name} 结果: {results1}")
    print(f"{model2_name} 结果: {results2}")
    print(f"\n{model1_name} 平均值: {np.mean(results1):.4f} ± {np.std(results1):.4f}")
    print(f"{model2_name} 平均值: {np.mean(results2):.4f} ± {np.std(results2):.4f}")
    print(f"\n{statistic_name}: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")

    # 判断显著性
    alpha = 0.05
    if p_value < alpha:
        print(f"\n结论: 差异显著 (p < {alpha})")
        if np.mean(results1) > np.mean(results2):
            print(f"      {model1_name} 显著优于 {model2_name}")
        else:
            print(f"      {model2_name} 显著优于 {model1_name}")
    else:
        print(f"\n结论: 差异不显著 (p >= {alpha})")

    print(f"{'='*70}")

    return {
        'statistic': float(t_stat),
        'p_value': float(p_value),
        'test_type': test_type,
        'mean_diff': float(np.mean(results1) - np.mean(results2)),
        'significant': p_value < alpha
    }
