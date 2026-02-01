"""
主实验脚本 - 智能垃圾分类系统
Main Experiment Script - Intelligent Waste Classification System

论文题目: An Intelligent Waste Classification System Based on
         Transfer Learning and MobileNetV2

使用方法:
    1. 下载数据集到 ./data/garbage_classification/
    2. 运行: python main.py

实验内容:
    1. 数据预处理与增强
    2. 三种模型对比: MobileNetV2, ResNet50, VGG16
    3. 训练与评估
    4. 可视化分析

优化内容:
    - 移除全局变量修改
    - 显式导入替代星号导入
    - 支持混合精度训练
    - 训练间内存清理
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Any, Tuple, Optional

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import tensorflow as tf

# GPU配置
def setup_gpu() -> None:
    """配置GPU内存增长和混合精度"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 {len(gpus)} 个GPU")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")

    # 导入配置
    from config import USE_MIXED_PRECISION

    # 启用混合精度训练（可提速30-50%）
    if USE_MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("已启用混合精度训练 (mixed_float16)")
        except Exception as e:
            logger.warning(f"无法启用混合精度训练: {e}")


# 在导入其他模块前配置GPU
setup_gpu()

# 显式导入配置
from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_DIR,
    EPOCHS, MODELS_TO_COMPARE, ABLATION_MODELS,
    ensure_dir
)

# 导入各模块
from data_loader import (
    download_dataset_instructions,
    create_data_split,
    create_data_generators,
    get_class_weights,
    DatasetError
)
from models import (
    build_model,
    compile_model,
    print_model_summary,
    get_model_info,
    ModelBuildError
)
from trainer import (
    train_model,
    save_training_results,
    print_training_summary,
    cleanup_memory,
    compile_model_with_focal_loss
)
from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_curves,
    measure_inference_time,
    compare_models,
    plot_per_class_metrics,
    visualize_predictions,
    generate_latex_table,
    visualize_gradcam,
    statistical_significance_test
)


def check_dataset(data_dir: str) -> bool:
    """
    检查数据集是否存在

    Args:
        data_dir: 数据目录路径

    Returns:
        bool: 数据集是否有效
    """
    if not os.path.exists(data_dir):
        print(f"\n错误: 数据集目录不存在: {data_dir}")
        download_dataset_instructions()
        return False

    # 检查是否有子目录
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) == 0:
        print(f"\n错误: 数据集目录为空: {data_dir}")
        download_dataset_instructions()
        return False

    print(f"数据集检测成功，发现 {len(subdirs)} 个类别")
    return True


def run_single_experiment(
    model_name: str,
    train_generator: Any,
    val_generator: Any,
    test_generator: Any,
    epochs: int = EPOCHS,
    class_weights: Optional[Dict[int, float]] = None,
    cleanup_after: bool = False
) -> Tuple[Dict[str, Any], Any, Any]:
    """
    运行单个模型的完整实验
    Run complete experiment for a single model

    Args:
        model_name: 模型名称
        train_generator: 训练数据生成器
        val_generator: 验证数据生成器
        test_generator: 测试数据生成器
        epochs: 训练轮数
        class_weights: 类别权重
        cleanup_after: 训练后是否清理内存

    Returns:
        Tuple: (results, model, history)
    """
    print("\n" + "="*70)
    print(f"开始实验: {model_name}")
    print("="*70)

    try:
        # 1. 构建模型
        print(f"\n[1/5] 构建 {model_name} 模型...")
        model = build_model(model_name)
        model = compile_model(model)
        model_info = print_model_summary(model)

        # 2. 训练模型
        print(f"\n[2/5] 训练 {model_name} 模型...")
        history, training_time = train_model(
            model, train_generator, val_generator,
            model_name=model_name,
            epochs=epochs,
            class_weights=class_weights
        )
        print_training_summary(history)

        # 3. 评估模型
        print(f"\n[3/5] 评估 {model_name} 模型...")
        eval_results = evaluate_model(model, test_generator, model_name)

        # 4. 测量推理时间
        print(f"\n[4/5] 测量推理时间...")
        inference_results = measure_inference_time(model, test_generator)

        # 5. 可视化
        print(f"\n[5/5] 生成可视化图表...")

        # 混淆矩阵
        plot_confusion_matrix(
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['class_names'],
            model_name
        )

        # 训练曲线
        plot_training_curves(history, model_name)

        # 类别指标
        plot_per_class_metrics(
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['class_names'],
            model_name
        )

        # 预测样例
        visualize_predictions(model, test_generator, model_name)

        # 整合结果
        results = {
            'model_name': model_name,
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1_score': eval_results['f1_score'],
            'model_size_mb': model_info['model_size_mb'],
            'total_params': model_info['total_params'],
            'trainable_params': model_info['trainable_params'],
            'inference_time_ms': inference_results['single_image_mean_ms'],
            'fps': inference_results['fps'],
            'training_time_min': training_time / 60
        }

        # 保存结果
        save_training_results(model_name, history, training_time, model_info, results)

        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'{model_name}_final.keras')
        model.save(model_path)
        print(f"\n模型已保存: {model_path}")

        # 可选：清理内存
        if cleanup_after:
            cleanup_memory(model)
            model = None

        return results, model, history

    except ModelBuildError as e:
        logger.error(f"模型构建失败: {e}")
        raise
    except Exception as e:
        logger.error(f"实验失败: {e}")
        raise


def run_all_experiments(
    data_dir: str,
    epochs: int = EPOCHS,
    models_to_compare: Optional[List[str]] = None,
    cleanup_between_models: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    运行所有模型的对比实验
    Run comparison experiments for all models

    Args:
        data_dir: 数据目录
        epochs: 训练轮数
        models_to_compare: 要对比的模型列表
        cleanup_between_models: 是否在模型间清理内存

    Returns:
        Tuple: (all_results, models_trained, histories)
    """
    if models_to_compare is None:
        models_to_compare = MODELS_TO_COMPARE

    print("\n" + "="*70)
    print("智能垃圾分类系统 - 对比实验")
    print("Intelligent Waste Classification System - Comparison Experiments")
    print("="*70)

    # 1. 准备数据
    print("\n[Step 1] 准备数据...")

    # 检查是否已经划分
    split_dir = os.path.join(data_dir, 'split')
    if not os.path.exists(os.path.join(split_dir, 'train')):
        print("划分数据集...")
        try:
            create_data_split(data_dir, split_dir)
        except DatasetError as e:
            logger.error(f"数据集划分失败: {e}")
            raise
    else:
        print("使用已有的数据划分")

    # 创建数据生成器
    train_generator, val_generator, test_generator = create_data_generators(split_dir)

    # 计算类别权重
    class_weights = get_class_weights(train_generator)

    # 2. 运行各模型实验
    all_results: List[Dict[str, Any]] = []
    models_trained: Dict[str, Any] = {}
    histories: Dict[str, Any] = {}

    for i, model_name in enumerate(models_to_compare):
        is_last_model = (i == len(models_to_compare) - 1)

        results, model, history = run_single_experiment(
            model_name,
            train_generator,
            val_generator,
            test_generator,
            epochs=epochs,
            class_weights=class_weights,
            cleanup_after=cleanup_between_models and not is_last_model
        )
        all_results.append(results)

        # 只保留最后一个模型（节省内存）
        if is_last_model or not cleanup_between_models:
            models_trained[model_name] = model
        histories[model_name] = history

        # 重置生成器
        train_generator.reset()
        val_generator.reset()
        test_generator.reset()

    # 3. 生成对比报告
    print("\n" + "="*70)
    print("生成对比报告")
    print("="*70)

    compare_models(all_results)

    # 生成LaTeX表格
    generate_latex_table(all_results)

    # 4. 打印最终总结
    _print_final_summary(all_results)

    return all_results, models_trained, histories


def _print_final_summary(all_results: List[Dict[str, Any]]) -> None:
    """
    打印最终实验总结

    Args:
        all_results: 所有模型结果
    """
    print("\n" + "="*70)
    print("实验完成 - 最终总结")
    print("="*70)

    best_model = max(all_results, key=lambda x: x['accuracy'])
    print(f"\n最佳模型: {best_model['model_name']}")
    print(f"  - 准确率: {best_model['accuracy']:.4f}")
    print(f"  - F1分数: {best_model['f1_score']:.4f}")
    print(f"  - 模型大小: {best_model['model_size_mb']:.2f} MB")
    print(f"  - 推理时间: {best_model['inference_time_ms']:.2f} ms")

    fastest_model = min(all_results, key=lambda x: x['inference_time_ms'])
    print(f"\n最快模型: {fastest_model['model_name']}")
    print(f"  - 推理时间: {fastest_model['inference_time_ms']:.2f} ms")
    print(f"  - FPS: {fastest_model['fps']:.1f}")

    smallest_model = min(all_results, key=lambda x: x['model_size_mb'])
    print(f"\n最小模型: {smallest_model['model_name']}")
    print(f"  - 模型大小: {smallest_model['model_size_mb']:.2f} MB")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("="*70)


def run_ablation_study(
    data_dir: str,
    epochs: int = EPOCHS,
    ablation_configs: Optional[List[tuple]] = None
) -> List[Dict[str, Any]]:
    """
    消融实验：验证CBAM和Focal Loss各自的贡献
    Ablation Study: Verify the contribution of CBAM and Focal Loss

    Args:
        data_dir: 数据目录
        epochs: 训练轮数
        ablation_configs: 消融实验配置列表 [(model_name, use_focal_loss, config_name), ...]

    Returns:
        list: 消融实验结果列表
    """
    if ablation_configs is None:
        ablation_configs = ABLATION_MODELS

    print("\n" + "="*70)
    print("消融实验 (Ablation Study)")
    print("验证 CBAM 注意力机制和 Focal Loss 的贡献")
    print("="*70)

    # 准备数据
    split_dir = os.path.join(data_dir, 'split')
    if not os.path.exists(os.path.join(split_dir, 'train')):
        print("划分数据集...")
        try:
            create_data_split(data_dir, split_dir)
        except DatasetError as e:
            logger.error(f"数据集划分失败: {e}")
            raise

    # 创建数据生成器
    train_generator, val_generator, test_generator = create_data_generators(split_dir)
    class_weights = get_class_weights(train_generator)

    ablation_results = []

    for model_name, use_focal_loss, config_name in ablation_configs:
        print(f"\n{'='*70}")
        print(f"配置: {config_name}")
        print(f"  模型: {model_name}")
        print(f"  Focal Loss: {'是' if use_focal_loss else '否'}")
        print(f"{'='*70}")

        try:
            # 1. 构建模型
            model = build_model(model_name)

            # 2. 编译模型（使用或不使用Focal Loss）
            model = compile_model_with_focal_loss(
                model,
                use_focal_loss=use_focal_loss
            )

            # 3. 训练模型
            history, training_time = train_model(
                model, train_generator, val_generator,
                model_name=f"ablation_{config_name.replace(' ', '_')}",
                epochs=epochs,
                class_weights=class_weights
            )

            # 4. 评估模型
            eval_results = evaluate_model(model, test_generator, config_name)

            # 5. 可视化（包括Grad-CAM）
            plot_confusion_matrix(
                eval_results['y_true'],
                eval_results['y_pred'],
                eval_results['class_names'],
                f"ablation_{config_name.replace(' ', '_')}"
            )

            plot_training_curves(
                history,
                f"ablation_{config_name.replace(' ', '_')}"
            )

            # 6. Grad-CAM可视化（如果是CBAM模型）
            if 'CBAM' in model_name:
                try:
                    visualize_gradcam(
                        model,
                        test_generator,
                        eval_results['class_names'],
                        f"ablation_{config_name.replace(' ', '_')}"
                    )
                except Exception as e:
                    logger.warning(f"Grad-CAM可视化失败: {e}")

            # 记录结果
            ablation_results.append({
                'configuration': config_name,
                'model_name': model_name,
                'use_focal_loss': use_focal_loss,
                'accuracy': eval_results['accuracy'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1_score': eval_results['f1_score']
            })

            # 重置生成器
            train_generator.reset()
            val_generator.reset()
            test_generator.reset()

            # 清理内存
            cleanup_memory(model)

        except Exception as e:
            logger.error(f"配置 {config_name} 实验失败: {e}")
            continue

    # 打印消融实验结果汇总
    print("\n" + "="*70)
    print("消融实验结果汇总 (Ablation Study Results)")
    print("="*70)
    print(f"{'Configuration':<40} {'Accuracy':<12} {'F1-Score':<12}")
    print("-"*70)
    for r in ablation_results:
        print(f"{r['configuration']:<40} {r['accuracy']:.4f}       {r['f1_score']:.4f}")
    print("="*70)

    # 如果有scipy，进行统计显著性检验
    if len(ablation_results) >= 2:
        # 比较 Baseline vs Proposed
        baseline = next((r for r in ablation_results if 'Baseline' in r['configuration']), None)
        proposed = next((r for r in ablation_results if 'Proposed' in r['configuration']), None)

        if baseline and proposed:
            print("\n提示：完整的统计显著性检验需要K折交叉验证的多次结果。")
            print(f"当前单次实验结果：")
            print(f"  Baseline: {baseline['accuracy']:.4f}")
            print(f"  Proposed: {proposed['accuracy']:.4f}")
            print(f"  提升: {(proposed['accuracy'] - baseline['accuracy']):.4f}")

    return ablation_results


def run_quick_test(
    data_dir: str = DATA_DIR,
    epochs: int = 3,
    models: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    快速测试（使用少量epoch）
    Quick test with fewer epochs

    Args:
        data_dir: 数据目录
        epochs: 训练轮数（默认3）
        models: 要测试的模型列表

    Returns:
        Tuple: (results, models, histories)
    """
    print("\n" + "="*70)
    print(f"快速测试模式 ({epochs} epochs)")
    print("="*70)

    if models is None:
        models = ['MobileNetV2']  # 快速测试只用一个模型

    return run_all_experiments(
        data_dir,
        epochs=epochs,
        models_to_compare=models,
        cleanup_between_models=True
    )


def main() -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """
    主函数

    Returns:
        Optional[Tuple]: (results, models, histories) 或 None（如果失败）
    """
    print("\n" + "="*70)
    print("智能垃圾分类系统")
    print("An Intelligent Waste Classification System")
    print("Based on Transfer Learning and MobileNetV2")
    print("="*70)

    # 检查数据集
    if not check_dataset(DATA_DIR):
        print("\n请按照说明下载数据集后重新运行程序。")
        sys.exit(1)

    # 运行实验
    try:
        results, models, histories = run_all_experiments(DATA_DIR)
        print("\n所有实验成功完成!")
        return results, models, histories
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        sys.exit(0)
    except DatasetError as e:
        print(f"\n数据集错误: {e}")
        sys.exit(1)
    except ModelBuildError as e:
        print(f"\n模型构建错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("实验出错")
        print(f"\n实验出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    results = main()
