"""
数据加载与预处理模块
Data Loading and Preprocessing Module

功能:
1. 数据集下载指导
2. 数据集划分（训练/验证/测试）
3. 数据生成器创建（带数据增强）
4. 类别权重计算
5. tf.data高效数据管道
"""

import os
import shutil
import logging
from collections import Counter
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from config import (
    BATCH_SIZE, IMG_SIZE, RANDOM_SEED,
    AUGMENTATION_CONFIG, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    MIN_SAMPLES_PER_CLASS, PREFETCH_BUFFER, SHUFFLE_BUFFER,
    ensure_dir
)

# 配置日志
logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """数据集相关错误"""
    pass


class DatasetValidationWarning(UserWarning):
    """数据集验证警告"""
    pass


def download_dataset_instructions() -> None:
    """
    打印数据集下载说明
    Print dataset download instructions
    """
    instructions = """
    ============================================================
    数据集下载说明 (Dataset Download Instructions)
    ============================================================

    方法1: 使用Kaggle API下载
    --------------------------
    1. 安装kaggle: pip install kaggle
    2. 配置API密钥: 将kaggle.json放到 ~/.kaggle/
    3. 运行命令:
       kaggle datasets download -d mostafaabla/garbage-classification
    4. 解压到 ./data/garbage_classification/

    方法2: 手动下载
    --------------------------
    1. 访问: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
    2. 点击 "Download" 下载数据集
    3. 解压到 ./data/garbage_classification/

    数据集结构应为:
    ./data/garbage_classification/
        ├── battery/
        ├── biological/
        ├── cardboard/
        ├── clothes/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        ├── shoes/
        ├── trash/
        ├── white-glass/
        └── brown-glass/
    ============================================================
    """
    print(instructions)


def validate_image(image_path: str) -> bool:
    """
    验证图像文件是否有效

    Args:
        image_path: 图像文件路径

    Returns:
        bool: 图像是否有效
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_data_split(
    data_dir: str,
    output_dir: str,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    validate_images: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    将数据集划分为训练集、验证集、测试集
    Split dataset into train, validation, and test sets

    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        validate_images: 是否验证图像有效性

    Returns:
        Dict: 各划分集的统计信息

    Raises:
        DatasetError: 数据集目录不存在或为空
    """
    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("数据划分比例之和必须为1.0")

    # 检查数据目录
    if not os.path.exists(data_dir):
        raise DatasetError(f"数据集目录不存在: {data_dir}")

    np.random.seed(RANDOM_SEED)

    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        ensure_dir(split_dir)

    statistics: Dict[str, Dict[str, int]] = {split: {} for split in splits}
    invalid_images: List[str] = []

    # 遍历每个类别
    class_dirs = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]

    if not class_dirs:
        raise DatasetError(f"数据集目录为空: {data_dir}")

    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)

        # 获取该类别所有图片
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if len(images) == 0:
            logger.warning(f"类别 {class_name} 没有有效图像，跳过")
            continue

        # 可选：验证图像有效性
        if validate_images:
            valid_images = []
            for img in images:
                img_path = os.path.join(class_dir, img)
                if validate_image(img_path):
                    valid_images.append(img)
                else:
                    invalid_images.append(img_path)
            images = valid_images

        # 随机打乱
        np.random.shuffle(images)

        # 计算划分点
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # 复制文件到对应目录
        for split, split_images in [('train', train_images),
                                     ('val', val_images),
                                     ('test', test_images)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            ensure_dir(split_class_dir)

            for img in split_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_class_dir, img)
                try:
                    shutil.copy2(src, dst)
                except IOError as e:
                    logger.warning(f"无法复制文件 {src}: {e}")
                    continue

            statistics[split][class_name] = len(split_images)

        print(f"类别 {class_name}: 训练集 {len(train_images)}, "
              f"验证集 {len(val_images)}, 测试集 {len(test_images)}")

    # 报告无效图像
    if invalid_images:
        logger.warning(f"发现 {len(invalid_images)} 个无效图像文件")

    # 验证划分结果
    _validate_data_split(output_dir, statistics)

    print("\n数据划分完成!")
    return statistics


def _validate_data_split(output_dir: str, statistics: Dict[str, Dict[str, int]]) -> None:
    """
    验证数据划分结果

    Args:
        output_dir: 输出目录
        statistics: 划分统计信息
    """
    warnings_list = []

    for split, class_counts in statistics.items():
        for class_name, count in class_counts.items():
            if count < MIN_SAMPLES_PER_CLASS:
                warnings_list.append(
                    f"警告: {split}/{class_name} 只有 {count} 个样本 "
                    f"(建议至少 {MIN_SAMPLES_PER_CLASS} 个)"
                )

        # 检查是否有空的划分
        if not class_counts:
            warnings_list.append(f"错误: {split} 集为空!")

    for warning in warnings_list:
        print(warning)
        logger.warning(warning)


def create_data_generators(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE
) -> Tuple[Any, Any, Any]:
    """
    创建数据生成器（带数据增强）
    Create data generators with augmentation

    Args:
        data_dir: 包含 train/val/test 子目录的数据目录
        batch_size: 批次大小
        img_size: 图像尺寸

    Returns:
        Tuple: (train_generator, val_generator, test_generator)

    Raises:
        DatasetError: 数据目录结构不正确
    """
    # 验证目录结构
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            raise DatasetError(f"缺少 {split} 目录: {split_path}")

    # 训练集数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        brightness_range=AUGMENTATION_CONFIG['brightness_range'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode']
    )

    # 验证集和测试集只做归一化，不做增强
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # 创建生成器
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # 打印数据集统计信息
    print("\n" + "="*50)
    print("数据集统计 (Dataset Statistics)")
    print("="*50)
    print(f"训练集样本数: {train_generator.samples}")
    print(f"验证集样本数: {val_generator.samples}")
    print(f"测试集样本数: {test_generator.samples}")
    print(f"类别数: {train_generator.num_classes}")
    print(f"类别映射: {train_generator.class_indices}")
    print("="*50 + "\n")

    return train_generator, val_generator, test_generator


def get_class_weights(train_generator: Any) -> Dict[int, float]:
    """
    计算类别权重（处理类别不平衡）
    Calculate class weights for imbalanced dataset

    Args:
        train_generator: 训练数据生成器

    Returns:
        Dict[int, float]: 类别ID到权重的映射
    """
    # 统计每个类别的样本数
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)

    # 计算权重：总样本数 / (类别数 * 该类样本数)
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (n_classes * count)

    print("类别权重:", class_weights)
    return class_weights


def create_tf_dataset(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    使用tf.data创建高效数据管道
    Create efficient data pipeline using tf.data

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        img_size: 图像尺寸

    Returns:
        Tuple: (train_ds, val_ds, test_ds, class_names)
    """
    AUTOTUNE = tf.data.AUTOTUNE if PREFETCH_BUFFER == -1 else PREFETCH_BUFFER

    def parse_image(file_path: tf.Tensor) -> tf.Tensor:
        """读取并预处理图像"""
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = img / 255.0
        return img

    def augment(image: tf.Tensor) -> tf.Tensor:
        """数据增强"""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image

    # 获取类别列表
    class_names = sorted(os.listdir(os.path.join(data_dir, 'train')))

    def get_label(file_path: tf.Tensor) -> tf.Tensor:
        """从文件路径提取标签"""
        parts = tf.strings.split(file_path, os.sep)
        class_name = parts[-2]
        one_hot = tf.cast(class_name == class_names, tf.float32)
        return one_hot

    def process_path(file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """处理单个文件路径"""
        label = get_label(file_path)
        img = parse_image(file_path)
        return img, label

    # 创建数据集
    train_ds = tf.data.Dataset.list_files(os.path.join(data_dir, 'train', '*', '*'))
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(os.path.join(data_dir, 'val', '*', '*'))
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.list_files(os.path.join(data_dir, 'test', '*', '*'))
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    download_dataset_instructions()
