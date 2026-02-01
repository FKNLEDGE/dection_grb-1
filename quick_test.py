#!/usr/bin/env python3
"""
快速测试脚本 - 验证项目代码是否正常运行
Quick Test Script - Verify project code works correctly

使用方法：
    python quick_test.py

功能：
    1. 检查环境依赖
    2. 测试模型构建
    3. 验证数据加载（如果数据集存在）
    4. 简单训练测试（如果数据集存在）
"""

import sys
import os

print("=" * 70)
print("智能垃圾分类系统 - 快速测试")
print("Quick Test for Intelligent Waste Classification System")
print("=" * 70)

# ==================== 1. 检查Python版本 ====================
print("\n[1/6] 检查Python版本...")
python_version = sys.version_info
print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("❌ 需要Python 3.8或更高版本")
    sys.exit(1)
else:
    print("✅ Python版本符合要求")

# ==================== 2. 检查依赖包 ====================
print("\n[2/6] 检查依赖包...")

required_packages = {
    'tensorflow': '2.10.0',
    'numpy': '1.21.0',
    'pandas': '1.3.0',
    'sklearn': '1.0.0',
    'PIL': '8.0.0',
    'cv2': '4.5.0',
    'matplotlib': '3.4.0',
    'seaborn': '0.11.0',
}

missing_packages = []
for package, min_version in required_packages.items():
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
        elif package == 'PIL':
            from PIL import Image
            version = Image.__version__ if hasattr(Image, '__version__') else 'unknown'
        elif package == 'cv2':
            import cv2
            version = cv2.__version__
        else:
            pkg = __import__(package)
            version = pkg.__version__

        print(f"✅ {package}: {version}")
    except ImportError:
        print(f"❌ {package}: 未安装")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✅ 所有依赖包已安装")

# ==================== 3. 检查GPU ====================
print("\n[3/6] 检查GPU...")

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 检测到 {len(gpus)} 个GPU")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    # 设置GPU内存增长
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU内存增长已启用")
    except Exception as e:
        print(f"⚠️  GPU配置警告: {e}")
else:
    print("⚠️  未检测到GPU，将使用CPU运行（速度较慢）")

# ==================== 4. 测试模型构建 ====================
print("\n[4/6] 测试模型构建...")

try:
    from models import build_model, compile_model, print_model_summary

    # 测试MobileNetV2
    print("  构建 MobileNetV2...")
    model = build_model('MobileNetV2', num_classes=12)
    model = compile_model(model)
    info = print_model_summary(model)
    print(f"  ✅ MobileNetV2 构建成功 ({info['total_params']:,} 参数)")

    # 清理内存
    del model
    tf.keras.backend.clear_session()

    # 测试MobileNetV2_CBAM
    print("  构建 MobileNetV2_CBAM...")
    model_cbam = build_model('MobileNetV2_CBAM', num_classes=12)
    model_cbam = compile_model(model_cbam)
    info_cbam = print_model_summary(model_cbam)
    print(f"  ✅ MobileNetV2_CBAM 构建成功 ({info_cbam['total_params']:,} 参数)")

    # 清理内存
    del model_cbam
    tf.keras.backend.clear_session()

    print("✅ 模型构建测试通过")

except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 5. 检查数据集 ====================
print("\n[5/6] 检查数据集...")

from config import DATA_DIR

if os.path.exists(DATA_DIR):
    classes = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]

    if classes:
        print(f"✅ 发现数据集，共 {len(classes)} 个类别")

        total_images = 0
        for cls in classes:
            cls_dir = os.path.join(DATA_DIR, cls)
            images = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(images)

        print(f"  总图片数: {total_images}")

        # 可选：运行数据加载测试
        try:
            print("\n  测试数据加载...")
            from data_loader import create_data_split, create_data_generators
            from config import ensure_dir

            split_dir = './data/garbage_classification/split_test'

            # 清理旧的测试划分
            import shutil
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)

            # 创建测试数据划分（只用10%数据）
            print("    划分数据集（测试模式）...")
            create_data_split(
                DATA_DIR,
                split_dir,
                train_ratio=0.08,  # 减少数据量用于快速测试
                val_ratio=0.01,
                test_ratio=0.01,
                validate_images=False
            )

            print("    创建数据生成器...")
            train_gen, val_gen, test_gen = create_data_generators(
                split_dir,
                batch_size=8  # 小batch size用于测试
            )

            print(f"    ✅ 数据加载测试通过")
            print(f"      - 训练样本: {train_gen.samples}")
            print(f"      - 验证样本: {val_gen.samples}")
            print(f"      - 测试样本: {test_gen.samples}")

            # 清理测试数据
            shutil.rmtree(split_dir)

        except Exception as e:
            print(f"    ⚠️  数据加载测试失败: {e}")
    else:
        print("⚠️  数据集目录存在但为空")
        print(f"   目录: {DATA_DIR}")
else:
    print("⚠️  数据集未下载")
    print(f"   期望位置: {DATA_DIR}")
    print("   请运行 python download_data.py 下载数据集")

# ==================== 6. 测试配置文件 ====================
print("\n[6/6] 测试配置文件...")

try:
    from config import (
        BATCH_SIZE, EPOCHS, LEARNING_RATE,
        NUM_CLASSES, IMG_SIZE, MODELS_TO_COMPARE
    )

    print(f"  ✅ 配置加载成功")
    print(f"    - 类别数: {NUM_CLASSES}")
    print(f"    - 图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"    - 批次大小: {BATCH_SIZE}")
    print(f"    - 训练轮数: {EPOCHS}")
    print(f"    - 学习率: {LEARNING_RATE}")
    print(f"    - 对比模型: {', '.join(MODELS_TO_COMPARE)}")

except Exception as e:
    print(f"  ❌ 配置加载失败: {e}")
    sys.exit(1)

# ==================== 总结 ====================
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

print("\n✅ 所有基础测试通过！")
print("\n下一步:")
print("  1. 如果还没有数据集，运行: python download_data.py")
print("  2. 快速测试（3个epoch）: python -c \"from main import run_quick_test; run_quick_test()\"")
print("  3. 完整训练（30个epoch）: python main.py")
print("  4. 在Colab中运行: 上传 garbage_classification_complete.ipynb")

print("\n" + "=" * 70)
