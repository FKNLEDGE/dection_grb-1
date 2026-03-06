#!/usr/bin/env python3
"""
论文结果一键生成脚本
Paper Results Generator - One-click script for EI paper figures, tables and data

使用方法:
    python generate_paper_results.py --data_dir ./data/garbage_classification
    python generate_paper_results.py --data_dir ./data/garbage_classification --epochs 30
    python generate_paper_results.py --ablation_only    # 仅运行消融实验
    python generate_paper_results.py --comparison_only  # 仅运行模型对比

输出目录结构:
    outputs/paper/
    ├── tables/           # LaTeX 表格（可直接粘贴到论文）
    ├── figures/          # 高清图片（300 DPI，EI 标准）
    ├── data/             # 原始数据（CSV + JSON）
    └── PAPER_FIGURES_GUIDE.txt  # 论文章节→图表映射指南

本脚本会依次执行:
    1. 多模型对比实验（5个模型）
    2. 扩展消融实验（9组配置）
    3. Grad-CAM 可解释性分析
    4. 分类别详细指标分析
    5. 生成全部 LaTeX 表格
    6. 生成论文图表映射指南
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PaperResults')


# ============================================================
# 输出目录配置
# ============================================================
PAPER_OUTPUT_DIR = './outputs/paper'
TABLES_DIR = os.path.join(PAPER_OUTPUT_DIR, 'tables')
FIGURES_DIR = os.path.join(PAPER_OUTPUT_DIR, 'figures')
DATA_DIR_OUT = os.path.join(PAPER_OUTPUT_DIR, 'data')


def setup_output_dirs():
    """创建输出目录结构"""
    for d in [PAPER_OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, DATA_DIR_OUT]:
        os.makedirs(d, exist_ok=True)
    print_step("输出目录已创建", f"所有结果将保存到: {os.path.abspath(PAPER_OUTPUT_DIR)}")


# ============================================================
# 美化输出工具函数
# ============================================================
def print_banner():
    """打印启动横幅"""
    print("\n")
    print("=" * 70)
    print("  论文结果一键生成工具")
    print("  Paper Results Generator for EI Conference")
    print("=" * 70)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {os.path.abspath(PAPER_OUTPUT_DIR)}")
    print("=" * 70)
    print()


def print_section(title: str, description: str = ""):
    """打印大章节标题"""
    print("\n")
    print("*" * 70)
    print(f"  {title}")
    if description:
        print(f"  {description}")
    print("*" * 70)
    print()


def print_step(step_name: str, detail: str = ""):
    """打印步骤信息"""
    print(f"  [步骤] {step_name}")
    if detail:
        print(f"         {detail}")


def print_output(label: str, path: str):
    """打印输出文件路径"""
    print(f"  [输出] {label}")
    print(f"         -> {os.path.abspath(path)}")


def print_result(metric: str, value, fmt: str = ".4f"):
    """打印单个结果指标"""
    if isinstance(value, float):
        print(f"         {metric}: {value:{fmt}}")
    else:
        print(f"         {metric}: {value}")


def print_elapsed(start_time: float, task_name: str):
    """打印耗时"""
    elapsed = time.time() - start_time
    if elapsed < 60:
        print(f"  [耗时] {task_name}: {elapsed:.1f} 秒")
    else:
        print(f"  [耗时] {task_name}: {elapsed/60:.1f} 分钟")


# ============================================================
# 延迟导入（避免在--help时加载TensorFlow）
# ============================================================
def lazy_imports():
    """延迟导入TensorFlow和项目模块"""
    print_step("正在加载 TensorFlow 和项目模块...")
    print("         （首次加载可能需要 10-30 秒，请耐心等待）")
    print()

    global tf, np, plt, pd, sns
    global setup_gpu
    global DATA_DIR, OUTPUT_DIR, EPOCHS, MODELS_TO_COMPARE
    global ABLATION_CONFIGS_EXTENDED, CLASS_NAMES, ensure_dir
    global create_data_split, create_data_generators, get_class_weights, DatasetError
    global build_model, get_model_info, ModelBuildError
    global train_model, compile_model_with_focal_loss, cleanup_memory, save_training_results
    global evaluate_model, plot_confusion_matrix, plot_training_curves
    global measure_inference_time, compare_models, plot_ablation_results
    global plot_per_class_metrics, visualize_gradcam, generate_latex_table
    global statistical_significance_test
    global run_all_experiments, run_extended_ablation_study, check_dataset

    import tensorflow as tf
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 无头模式，服务器也能跑
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # 设置全局绘图参数（EI论文标准）
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.unicode_minus': False,
    })

    from main import (
        setup_gpu as _setup_gpu,
        run_all_experiments, run_extended_ablation_study, check_dataset
    )
    setup_gpu = _setup_gpu

    from config import (
        DATA_DIR, OUTPUT_DIR, EPOCHS, MODELS_TO_COMPARE,
        ABLATION_CONFIGS_EXTENDED, CLASS_NAMES, ensure_dir
    )
    from data_loader import (
        create_data_split, create_data_generators, get_class_weights, DatasetError
    )
    from models import build_model, get_model_info, ModelBuildError
    from trainer import (
        train_model, compile_model_with_focal_loss,
        cleanup_memory, save_training_results
    )
    from evaluation import (
        evaluate_model, plot_confusion_matrix, plot_training_curves,
        measure_inference_time, compare_models, plot_ablation_results,
        plot_per_class_metrics, visualize_gradcam, generate_latex_table,
        statistical_significance_test
    )

    print("  [完成] 所有模块加载成功!")
    print()


# ============================================================
# 第一部分：多模型对比实验
# ============================================================
def run_model_comparison(data_dir: str, epochs: int) -> List[Dict[str, Any]]:
    """
    运行多模型对比实验并生成论文所需的全部输出

    这个函数会：
    1. 训练 5 个不同架构的模型（MobileNetV2、MobileNetV2_CBAM、VGG16、DenseNet121、EfficientNetB0）
    2. 在测试集上评估每个模型的 Accuracy/Precision/Recall/F1
    3. 测量每个模型的推理速度和模型大小
    4. 生成对比表格（CSV + LaTeX）和对比图表
    5. 为每个模型生成混淆矩阵和训练曲线
    """
    print_section(
        "第一部分：多模型对比实验",
        "训练并评估 5 个预训练模型，生成性能对比表格和图表"
    )

    start_time = time.time()

    # --- 1. 运行实验 ---
    print_step("开始训练 5 个模型", "MobileNetV2, MobileNetV2_CBAM, VGG16, DenseNet121, EfficientNetB0")
    print("         每个模型训练完成后会自动评估并生成混淆矩阵、训练曲线")
    print()

    results, trained_models, histories = run_all_experiments(
        data_dir, epochs=epochs, cleanup_between_models=True
    )

    # --- 2. 生成对比图表 ---
    print()
    print_step("生成模型对比图表...")

    # 复制对比图到 paper 目录
    src_comparison = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    if os.path.exists(src_comparison):
        dst = os.path.join(FIGURES_DIR, 'fig1_model_comparison.png')
        shutil.copy2(src_comparison, dst)
        print_output("图1: 模型性能对比图", dst)

    # --- 3. 生成 LaTeX 对比表格 ---
    print_step("生成 LaTeX 模型对比表格...")
    latex_content = generate_model_comparison_latex(results)
    latex_path = os.path.join(TABLES_DIR, 'tab1_model_comparison.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print_output("表1: 模型对比表 (LaTeX)", latex_path)

    # --- 4. 保存 CSV 数据 ---
    csv_path = os.path.join(DATA_DIR_OUT, 'model_comparison.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print_output("模型对比原始数据 (CSV)", csv_path)

    # --- 5. 复制各模型的混淆矩阵和训练曲线 ---
    print_step("整理各模型的混淆矩阵和训练曲线...")
    for i, r in enumerate(results):
        name = r['model_name']
        # 混淆矩阵
        src_cm = os.path.join(OUTPUT_DIR, name, f'{name}_confusion_matrix.png')
        if os.path.exists(src_cm):
            dst_cm = os.path.join(FIGURES_DIR, f'fig{5+i}_confusion_matrix_{name}.png')
            shutil.copy2(src_cm, dst_cm)
            print_output(f"图{5+i}: {name} 混淆矩阵", dst_cm)
        # 训练曲线
        src_tc = os.path.join(OUTPUT_DIR, name, f'{name}_training_curves.png')
        if os.path.exists(src_tc):
            dst_tc = os.path.join(FIGURES_DIR, f'fig{10+i}_training_curves_{name}.png')
            shutil.copy2(src_tc, dst_tc)
            print_output(f"图{10+i}: {name} 训练曲线", dst_tc)

    # --- 6. 打印结果摘要 ---
    print()
    print_step("模型对比结果摘要:")
    print(f"         {'模型':<25} {'Accuracy':<12} {'F1-Score':<12} {'大小(MB)':<12}")
    print(f"         {'-'*60}")
    for r in results:
        size = r.get('model_size_mb', 0)
        print(f"         {r['model_name']:<25} {r['accuracy']:.4f}       "
              f"{r['f1_score']:.4f}       {size:.1f}")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n         最佳模型: {best['model_name']} (Accuracy={best['accuracy']:.4f})")

    print_elapsed(start_time, "模型对比实验")

    return results


def generate_model_comparison_latex(results: List[Dict[str, Any]]) -> str:
    """生成模型对比的 LaTeX 表格（booktabs 格式）"""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Performance Comparison of Different CNN Models}',
        r'\label{tab:model_comparison}',
        r'\begin{tabular}{lccccccc}',
        r'\toprule',
        r'Model & Accuracy & Precision & Recall & F1-Score & Params (M) & Size (MB) & Time (ms) \\',
        r'\midrule',
    ]

    for r in results:
        params = r.get('total_params', 0) / 1e6
        size = r.get('model_size_mb', 0)
        inf_time = r.get('inference_time_ms', 0)
        name = r['model_name'].replace('_', r'\_')
        lines.append(
            f"{name} & {r['accuracy']:.4f} & {r['precision']:.4f} & "
            f"{r['recall']:.4f} & {r['f1_score']:.4f} & "
            f"{params:.1f} & {size:.1f} & {inf_time:.1f} \\\\"
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# ============================================================
# 第二部分：扩展消融实验
# ============================================================
def run_ablation(data_dir: str, epochs: int) -> List[Dict[str, Any]]:
    """
    运行扩展消融实验（9组配置）并生成论文所需的全部输出

    消融实验验证每个组件的独立贡献：
    - Baseline → 无任何增强的 MobileNetV2
    - + CBAM → 添加通道+空间注意力
    - + SE-Net → 添加通道注意力（对比参照）
    - + Focal Loss → 改用 Focal Loss 损失函数
    - + Label Smoothing → 添加标签平滑正则化
    - + Mixup/CutMix → 添加高级数据增强
    - + EMA → 添加指数移动平均
    - + CBAM + Focal Loss → 双因素组合
    - Full Proposed → 全部组件组合
    """
    print_section(
        "第二部分：扩展消融实验（9组配置）",
        "逐个添加组件，验证每个组件的边际贡献"
    )

    start_time = time.time()

    print_step("消融实验配置一览:")
    for i, cfg in enumerate(ABLATION_CONFIGS_EXTENDED):
        flags = []
        if cfg['focal_loss']:
            flags.append('FL')
        if cfg.get('label_smoothing', 0) > 0:
            flags.append(f"LS={cfg['label_smoothing']}")
        if cfg.get('use_mixup'):
            flags.append('Mixup')
        if cfg.get('use_ema'):
            flags.append('EMA')
        flag_str = ', '.join(flags) if flags else '无增强'
        print(f"         [{i+1}/9] {cfg['name']:<25} 模型={cfg['model']:<20} 技术=[{flag_str}]")
    print()

    # 运行消融实验
    ablation_results = run_extended_ablation_study(data_dir, epochs=epochs)

    # --- 复制消融图表到 paper 目录 ---
    print()
    print_step("整理消融实验图表...")

    ablation_src = os.path.join(OUTPUT_DIR, 'ablation')
    fig_map = {
        'ablation_comparison.png': ('fig2_ablation_comparison.png', '图2: 消融对比柱状图'),
        'ablation_delta.png': ('fig3_ablation_delta.png', '图3: 增量贡献瀑布图'),
        'ablation_radar.png': ('fig4_ablation_radar.png', '图4: 消融雷达图'),
    }
    for src_name, (dst_name, label) in fig_map.items():
        src = os.path.join(ablation_src, src_name)
        if os.path.exists(src):
            dst = os.path.join(FIGURES_DIR, dst_name)
            shutil.copy2(src, dst)
            print_output(label, dst)

    # 复制 LaTeX 表格
    src_tex = os.path.join(ablation_src, 'ablation_table.tex')
    if os.path.exists(src_tex):
        dst_tex = os.path.join(TABLES_DIR, 'tab2_ablation_results.tex')
        shutil.copy2(src_tex, dst_tex)
        print_output("表2: 消融实验表 (LaTeX)", dst_tex)

    # 复制数据文件
    for fname in ['ablation_results.csv', 'ablation_results.json']:
        src = os.path.join(ablation_src, fname)
        if os.path.exists(src):
            dst = os.path.join(DATA_DIR_OUT, fname)
            shutil.copy2(src, dst)
            print_output(f"消融数据 ({fname})", dst)

    # --- 打印结果摘要 ---
    print()
    print_step("消融实验结果摘要:")
    if ablation_results:
        baseline_acc = ablation_results[0]['accuracy']
        print(f"         {'配置':<25} {'Accuracy':<12} {'ΔAcc':<12} {'F1-Score':<12}")
        print(f"         {'-'*60}")
        for r in ablation_results:
            delta = r['accuracy'] - baseline_acc
            print(f"         {r['configuration']:<25} {r['accuracy']:.4f}       "
                  f"{delta:+.4f}       {r['f1_score']:.4f}")

        # 找出最佳配置
        best = max(ablation_results, key=lambda x: x['accuracy'])
        print(f"\n         最佳配置: {best['configuration']} (Accuracy={best['accuracy']:.4f})")
        print(f"         相对Baseline提升: {best['accuracy'] - baseline_acc:+.4f} "
              f"({(best['accuracy'] - baseline_acc) / baseline_acc:+.2%})")

    print_elapsed(start_time, "消融实验")

    return ablation_results


# ============================================================
# 第三部分：Grad-CAM 可解释性分析
# ============================================================
def run_gradcam_analysis(data_dir: str, epochs: int):
    """
    生成 Grad-CAM 可解释性分析图

    对比 Baseline（MobileNetV2）和 CBAM 增强模型的 Grad-CAM 热力图，
    直观展示注意力机制如何改善模型对垃圾关键特征的关注。
    """
    print_section(
        "第三部分：Grad-CAM 可解释性分析",
        "对比 Baseline vs CBAM 模型的类激活热力图"
    )

    start_time = time.time()

    # 准备数据
    split_dir = os.path.join(data_dir, 'split')
    if not os.path.exists(os.path.join(split_dir, 'train')):
        print_step("划分数据集...")
        create_data_split(data_dir, split_dir)

    _, _, test_generator = create_data_generators(split_dir)

    # 训练 Baseline 和 CBAM 模型（如果尚未训练）
    models_to_visualize = [
        ('MobileNetV2', False, 'Baseline'),
        ('MobileNetV2_CBAM', False, 'CBAM'),
    ]

    for model_name, use_focal, label in models_to_visualize:
        print_step(f"生成 {label} 模型的 Grad-CAM 热力图...")

        try:
            model = build_model(model_name)
            model = compile_model_with_focal_loss(model, use_focal_loss=use_focal)

            # 查找已保存的最佳权重
            weights_path = os.path.join('./saved_models', f'{model_name}_best.keras')
            if os.path.exists(weights_path):
                print(f"         加载已保存的权重: {weights_path}")
                model.load_weights(weights_path)
            else:
                print(f"         未找到已保存权重，使用预训练权重直接生成Grad-CAM")

            # 获取类别名
            class_names = list(test_generator.class_indices.keys())

            # 生成 Grad-CAM
            test_generator.reset()
            gradcam_dir = os.path.join(FIGURES_DIR, f'gradcam_{label.lower()}')
            os.makedirs(gradcam_dir, exist_ok=True)

            try:
                visualize_gradcam(
                    model, test_generator, class_names,
                    f"paper_gradcam_{label}"
                )
                # 复制结果
                src_dir = os.path.join(OUTPUT_DIR, f"paper_gradcam_{label}")
                if os.path.exists(src_dir):
                    for fname in os.listdir(src_dir):
                        if fname.endswith('.png'):
                            shutil.copy2(
                                os.path.join(src_dir, fname),
                                os.path.join(gradcam_dir, fname)
                            )
                print_output(f"{label} Grad-CAM 热力图", gradcam_dir)
            except Exception as e:
                print(f"         警告: {label} Grad-CAM 生成失败: {e}")
                print(f"         （这不影响其他结果的生成，可以在训练完成后重新运行）")

            cleanup_memory(model)
            test_generator.reset()

        except Exception as e:
            print(f"         错误: {label} 模型处理失败: {e}")
            continue

    print_elapsed(start_time, "Grad-CAM 分析")


# ============================================================
# 第四部分：分类别详细指标
# ============================================================
def run_per_class_analysis(data_dir: str, epochs: int):
    """
    生成分类别详细指标分析

    对最佳模型（MobileNetV2_CBAM）生成每个垃圾类别的 Precision/Recall/F1 详细分析，
    帮助识别模型在哪些类别上表现较好/较差。
    """
    print_section(
        "第四部分：分类别详细指标分析",
        "分析模型在 12 个垃圾类别上的具体表现"
    )

    start_time = time.time()

    # 准备数据
    split_dir = os.path.join(data_dir, 'split')
    if not os.path.exists(os.path.join(split_dir, 'train')):
        create_data_split(data_dir, split_dir)

    _, _, test_generator = create_data_generators(split_dir)

    # 使用 CBAM 模型
    print_step("评估 MobileNetV2_CBAM 的分类别指标...")

    try:
        model = build_model('MobileNetV2_CBAM')
        model = compile_model_with_focal_loss(model, use_focal_loss=True)

        # 加载权重
        weights_path = os.path.join('./saved_models', 'MobileNetV2_CBAM_best.keras')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"         已加载权重: {weights_path}")

        # 评估
        eval_results = evaluate_model(model, test_generator, 'MobileNetV2_CBAM')

        # 生成分类别指标图
        plot_per_class_metrics(
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['class_names'],
            'paper_per_class',
            output_dir=FIGURES_DIR
        )

        # 复制图片
        src_fig = os.path.join(FIGURES_DIR, 'paper_per_class',
                               'paper_per_class_per_class_metrics.png')
        if os.path.exists(src_fig):
            dst_fig = os.path.join(FIGURES_DIR, 'fig16_per_class_metrics.png')
            shutil.copy2(src_fig, dst_fig)
            print_output("图16: 分类别指标图", dst_fig)

        # 生成分类别 LaTeX 表格
        print_step("生成分类别指标 LaTeX 表格...")
        latex_content = generate_per_class_latex(
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['class_names']
        )
        latex_path = os.path.join(TABLES_DIR, 'tab3_per_class_metrics.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print_output("表3: 分类别指标表 (LaTeX)", latex_path)

        cleanup_memory(model)

    except Exception as e:
        print(f"         错误: 分类别分析失败: {e}")
        logger.exception("分类别分析失败")

    print_elapsed(start_time, "分类别分析")


def generate_per_class_latex(y_true, y_pred, class_names) -> str:
    """生成分类别指标的 LaTeX 表格"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Per-class Classification Performance of the Proposed Method}',
        r'\label{tab:per_class}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Class & Precision & Recall & F1-Score & Support \\',
        r'\midrule',
    ]

    for i, name in enumerate(class_names):
        escaped_name = name.replace('_', r'\_').replace('-', r'-')
        lines.append(
            f"{escaped_name} & {precision[i]:.4f} & {recall[i]:.4f} & "
            f"{f1[i]:.4f} & {support[i]} \\\\"
        )

    # 加权平均
    p_avg = sum(precision * support) / sum(support)
    r_avg = sum(recall * support) / sum(support)
    f1_avg = sum(f1 * support) / sum(support)

    lines.extend([
        r'\midrule',
        f"Weighted Avg & {p_avg:.4f} & {r_avg:.4f} & {f1_avg:.4f} & {sum(support)} \\\\",
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# ============================================================
# 第五部分：生成论文图表映射指南
# ============================================================
def generate_paper_guide(
    comparison_results: Optional[List] = None,
    ablation_results: Optional[List] = None
):
    """
    生成论文章节→图表映射指南

    这个文件告诉你每个图表对应论文的哪个章节，方便写论文时快速查找。
    """
    print_section(
        "第五部分：生成论文图表映射指南",
        "汇总所有输出文件及其在论文中的对应位置"
    )

    guide_lines = [
        "=" * 70,
        "论文图表映射指南 (Paper Figures & Tables Guide)",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "说明: 以下列出了所有为论文生成的图表和数据文件，",
        "      以及它们在论文中建议放置的位置。",
        "",
        "-" * 70,
        "LaTeX 表格 (tables/ 目录)",
        "-" * 70,
        "",
        "tab1_model_comparison.tex",
        "  -> 论文 §4.1 多模型对比实验",
        "  -> 表1: 不同CNN模型的性能对比",
        "  -> 包含: Accuracy, Precision, Recall, F1, 参数量, 模型大小, 推理时间",
        "",
        "tab2_ablation_results.tex",
        "  -> 论文 §4.2 消融实验",
        "  -> 表2: 消融实验结果",
        "  -> 包含: 9组配置的 Accuracy, Precision, Recall, F1, ΔAccuracy",
        "",
        "tab3_per_class_metrics.tex",
        "  -> 论文 §4.3 分类别分析",
        "  -> 表3: 各垃圾类别的分类指标",
        "  -> 包含: 12个类别的 Precision, Recall, F1, Support",
        "",
        "-" * 70,
        "图片 (figures/ 目录)",
        "-" * 70,
        "",
        "fig1_model_comparison.png",
        "  -> 论文 §4.1 多模型对比实验",
        "  -> 图1: 模型性能对比柱状图（Accuracy/F1/Size/Speed四宫格）",
        "",
        "fig2_ablation_comparison.png",
        "  -> 论文 §4.2 消融实验",
        "  -> 图2: 消融实验分组柱状图（Accuracy + F1 并排对比）",
        "",
        "fig3_ablation_delta.png",
        "  -> 论文 §4.2 消融实验",
        "  -> 图3: 各组件增量贡献图（相对Baseline的ΔAccuracy百分比）",
        "",
        "fig4_ablation_radar.png",
        "  -> 论文 §4.2 消融实验",
        "  -> 图4: 多指标雷达图（Baseline vs 最佳单因素 vs Full Proposed）",
        "",
        "fig5-fig9_confusion_matrix_*.png",
        "  -> 论文 §4.3 结果分析",
        "  -> 图5-9: 各模型的混淆矩阵",
        "",
        "fig10-fig14_training_curves_*.png",
        "  -> 论文 §4.1 或附录",
        "  -> 图10-14: 各模型的训练/验证曲线（Accuracy + Loss）",
        "",
        "gradcam_baseline/ 和 gradcam_cbam/",
        "  -> 论文 §4.4 模型可解释性分析",
        "  -> 图15: Grad-CAM 热力图对比（Baseline vs CBAM）",
        "",
        "fig16_per_class_metrics.png",
        "  -> 论文 §4.3 分类别分析",
        "  -> 图16: 各垃圾类别的P/R/F1指标柱状图",
        "",
        "-" * 70,
        "原始数据 (data/ 目录)",
        "-" * 70,
        "",
        "model_comparison.csv    -> 模型对比实验原始数据",
        "ablation_results.csv    -> 消融实验原始数据",
        "ablation_results.json   -> 消融实验结果（JSON格式）",
        "",
        "-" * 70,
        "论文章节建议结构",
        "-" * 70,
        "",
        "§1 Introduction",
        "§2 Related Work",
        "§3 Proposed Method",
        "  §3.1 Overall Architecture (网络结构图)",
        "  §3.2 CBAM Attention Mechanism (CBAM结构图)",
        "  §3.3 Focal Loss (公式)",
        "  §3.4 Training Strategy (训练技术说明)",
        "§4 Experiments",
        "  §4.1 Model Comparison (表1 + 图1)",
        "  §4.2 Ablation Study (表2 + 图2 + 图3 + 图4)",
        "  §4.3 Per-class Analysis (表3 + 图16 + 混淆矩阵)",
        "  §4.4 Interpretability Analysis (Grad-CAM图)",
        "§5 Conclusion",
        "",
        "=" * 70,
    ]

    guide_path = os.path.join(PAPER_OUTPUT_DIR, 'PAPER_FIGURES_GUIDE.txt')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(guide_lines))

    print_output("论文图表映射指南", guide_path)

    # 同时打印到控制台
    print()
    for line in guide_lines:
        print(f"  {line}")


# ============================================================
# 最终汇总
# ============================================================
def print_final_summary(
    comparison_results: Optional[List] = None,
    ablation_results: Optional[List] = None,
    total_start_time: float = 0
):
    """打印最终汇总"""
    print("\n")
    print("=" * 70)
    print("  全部完成! 论文结果已生成")
    print("=" * 70)

    # 列出所有生成的文件
    print()
    print("  生成的文件清单:")
    print(f"  {'─'*60}")

    total_files = 0
    for root, dirs, files in os.walk(PAPER_OUTPUT_DIR):
        for f in sorted(files):
            rel_path = os.path.relpath(os.path.join(root, f), PAPER_OUTPUT_DIR)
            print(f"  {rel_path}")
            total_files += 1

    print(f"  {'─'*60}")
    print(f"  共 {total_files} 个文件")
    print()

    # 关键结果摘要
    if comparison_results:
        best = max(comparison_results, key=lambda x: x['accuracy'])
        print(f"  [模型对比] 最佳模型: {best['model_name']} (Acc={best['accuracy']:.4f})")

    if ablation_results:
        baseline_acc = ablation_results[0]['accuracy']
        best_abl = max(ablation_results, key=lambda x: x['accuracy'])
        improvement = best_abl['accuracy'] - baseline_acc
        print(f"  [消融实验] 最佳配置: {best_abl['configuration']} "
              f"(Acc={best_abl['accuracy']:.4f}, Δ={improvement:+.4f})")

    if total_start_time:
        total_time = time.time() - total_start_time
        print(f"\n  总耗时: {total_time/60:.1f} 分钟")

    print()
    print(f"  所有结果保存在: {os.path.abspath(PAPER_OUTPUT_DIR)}")
    print()
    print("  下一步:")
    print("  1. 查看 PAPER_FIGURES_GUIDE.txt 了解每个图表对应论文的哪个章节")
    print("  2. 将 tables/ 目录中的 .tex 文件直接粘贴到 LaTeX 论文中")
    print("  3. 将 figures/ 目录中的 .png 图片插入论文对应位置")
    print("  4. 如需调整某个实验，可以使用 --ablation_only 或 --comparison_only 单独重跑")
    print()
    print("=" * 70)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='论文结果一键生成工具 - Paper Results Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python generate_paper_results.py --data_dir ./data/garbage_classification
  python generate_paper_results.py --data_dir ./data/garbage_classification --epochs 30
  python generate_paper_results.py --ablation_only    # 仅运行消融实验
  python generate_paper_results.py --comparison_only  # 仅运行模型对比
  python generate_paper_results.py --gradcam_only     # 仅运行Grad-CAM分析

输出目录: outputs/paper/
  ├── tables/    LaTeX 表格
  ├── figures/   高清图片 (300 DPI)
  ├── data/      原始数据 (CSV/JSON)
  └── PAPER_FIGURES_GUIDE.txt  论文章节→图表映射指南
        """
    )
    parser.add_argument('--data_dir', type=str, default='./data/garbage_classification',
                        help='数据集目录路径 (默认: ./data/garbage_classification)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--ablation_only', action='store_true',
                        help='仅运行消融实验（跳过模型对比）')
    parser.add_argument('--comparison_only', action='store_true',
                        help='仅运行模型对比（跳过消融实验）')
    parser.add_argument('--gradcam_only', action='store_true',
                        help='仅运行Grad-CAM分析（需要已训练的模型）')
    parser.add_argument('--skip_gradcam', action='store_true',
                        help='跳过Grad-CAM分析（节省时间）')

    args = parser.parse_args()

    # 打印启动横幅
    print_banner()

    # 延迟导入
    lazy_imports()

    # 检查数据集
    print_step("检查数据集...")
    if not check_dataset(args.data_dir):
        print("\n  错误: 未找到数据集!")
        print(f"  请将数据集下载到: {os.path.abspath(args.data_dir)}")
        print("  下载地址: https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
        sys.exit(1)
    print(f"         数据集路径: {os.path.abspath(args.data_dir)}")
    print(f"         训练轮数: {args.epochs}")
    print()

    # 创建输出目录
    setup_output_dirs()

    total_start_time = time.time()
    comparison_results = None
    ablation_results = None

    try:
        # --- 模型对比实验 ---
        if not args.ablation_only and not args.gradcam_only:
            comparison_results = run_model_comparison(args.data_dir, args.epochs)

        # --- 消融实验 ---
        if not args.comparison_only and not args.gradcam_only:
            ablation_results = run_ablation(args.data_dir, args.epochs)

        # --- Grad-CAM ---
        if not args.skip_gradcam:
            run_gradcam_analysis(args.data_dir, args.epochs)

        # --- 分类别分析 ---
        if not args.gradcam_only:
            run_per_class_analysis(args.data_dir, args.epochs)

        # --- 生成指南 ---
        generate_paper_guide(comparison_results, ablation_results)

        # --- 最终汇总 ---
        print_final_summary(comparison_results, ablation_results, total_start_time)

    except KeyboardInterrupt:
        print("\n\n  用户中断! 已生成的结果仍然有效。")
        sys.exit(0)
    except Exception as e:
        logger.exception("论文结果生成出错")
        print(f"\n  错误: {e}")
        print("  已生成的部分结果仍然保存在 outputs/paper/ 目录中。")
        sys.exit(1)


if __name__ == '__main__':
    main()
