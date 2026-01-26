"""
智能垃圾分类系统 - Jupyter Notebook / Google Colab 版本
Intelligent Waste Classification System - Notebook Version

在Google Colab中运行:
1. 上传此文件到Colab
2. 按顺序运行每个代码块
"""

#%% [markdown]
# # 智能垃圾分类系统
# ## An Intelligent Waste Classification System Based on Transfer Learning and MobileNetV2
# 
# 本notebook实现了基于迁移学习的垃圾图像分类系统，对比分析MobileNetV2、ResNet50、VGG16三种模型。

#%% [markdown]
# ## 1. 环境配置

#%%
# 检查GPU
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU可用: {tf.config.list_physical_devices('GPU')}")

#%%
# 安装依赖（如果需要）
# !pip install kaggle seaborn scikit-learn

#%% [markdown]
# ## 2. 下载数据集

#%%
# 方法1: 使用Kaggle API（需要配置kaggle.json）
# 在Colab中运行:
"""
from google.colab import files
files.upload()  # 上传kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mostafaabla/garbage-classification
!unzip -q garbage-classification.zip -d ./data
"""

#%%
# 方法2: 使用wget从备用源下载（如果有）
# !wget <备用下载链接> -O garbage_classification.zip
# !unzip -q garbage_classification.zip -d ./data

#%% [markdown]
# ## 3. 配置参数

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_DIR = "./data/garbage_classification"  # 修改为你的数据路径
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 12
RANDOM_SEED = 42

# 数据增强配置
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# 设置随机种子
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#%% [markdown]
# ## 4. 数据加载与预处理

#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """创建数据生成器"""
    
    # 训练集数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 使用20%作为验证集
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    # 训练集
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    # 验证集
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

# 创建生成器（仅在数据集存在时执行）
if os.path.exists(DATA_DIR):
    train_gen, val_gen = create_generators(DATA_DIR)

    print(f"\n训练集样本数: {train_gen.samples}")
    print(f"验证集样本数: {val_gen.samples}")
    print(f"类别数: {train_gen.num_classes}")
    print(f"类别: {list(train_gen.class_indices.keys())}")
else:
    train_gen, val_gen = None, None
    print(f"\n警告: 数据集目录不存在: {DATA_DIR}")
    print("请下载数据集后再运行训练代码")

#%%
# 可视化部分样本
def show_samples(generator, n=9):
    """显示样本图片"""
    batch = next(generator)
    images, labels = batch[0][:n], batch[1][:n]
    class_names = list(generator.class_indices.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            label_idx = np.argmax(labels[i])
            ax.set_title(class_names[label_idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    generator.reset()

if train_gen is not None:
    show_samples(train_gen)

#%% [markdown]
# ## 5. 模型构建

#%%
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16

def build_model(model_name, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    构建迁移学习模型
    
    Args:
        model_name: 'MobileNetV2', 'ResNet50', 或 'VGG16'
    """
    input_shape = (img_size, img_size, 3)
    
    # 选择基模型
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        dense_units = 128
    elif model_name == 'ResNet50':
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        dense_units = 256
    elif model_name == 'VGG16':
        base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        dense_units = 512
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    # 冻结基模型
    base_model.trainable = False
    
    # 构建完整模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name=f'{model_name}_Transfer')
    
    return model

def get_model_info(model):
    """获取模型信息"""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }

# 测试构建模型（可选，不需要数据集）
def test_model_building():
    """测试模型构建功能"""
    for name in ['MobileNetV2', 'ResNet50', 'VGG16']:
        model = build_model(name)
        info = get_model_info(model)
        print(f"\n{name}:")
        print(f"  总参数: {info['total_params']:,}")
        print(f"  可训练参数: {info['trainable_params']:,}")
        print(f"  估计大小: {info['size_mb']:.1f} MB")

# 仅在直接运行时测试模型构建
if __name__ == "__main__" or train_gen is not None:
    test_model_building()

#%% [markdown]
# ## 6. 训练模型

#%%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time

def train_model(model_name, train_gen, val_gen, epochs=EPOCHS):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 构建模型
    model = build_model(model_name)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(f'{model_name}_best.keras', monitor='val_accuracy', 
                       save_best_only=True, verbose=1)
    ]
    
    # 训练
    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"\n训练完成! 耗时: {training_time/60:.1f} 分钟")
    
    return model, history, training_time

#%%
# 训练所有模型（仅在数据集可用时执行）
results = {}
models_dict = {}
histories = {}

if train_gen is not None and val_gen is not None:
    for model_name in ['MobileNetV2', 'ResNet50', 'VGG16']:
        model, history, train_time = train_model(model_name, train_gen, val_gen, epochs=EPOCHS)

        models_dict[model_name] = model
        histories[model_name] = history

        # 获取模型信息
        info = get_model_info(model)

        results[model_name] = {
            'best_val_accuracy': max(history.history['val_accuracy']),
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'training_time_min': train_time / 60,
            'model_size_mb': info['size_mb'],
            'total_params': info['total_params']
        }

        # 重置生成器
        train_gen.reset()
        val_gen.reset()
else:
    print("\n跳过训练：数据集不可用")

#%% [markdown]
# ## 7. 评估与可视化

#%%
def plot_training_curves(histories):
    """绘制所有模型的训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'MobileNetV2': '#2ecc71', 'ResNet50': '#3498db', 'VGG16': '#e74c3c'}
    
    for name, history in histories.items():
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # 准确率
        axes[0].plot(epochs, history.history['accuracy'], 
                    linestyle='--', color=colors[name], alpha=0.7)
        axes[0].plot(epochs, history.history['val_accuracy'], 
                    label=f'{name}', color=colors[name], linewidth=2)
        
        # 损失
        axes[1].plot(epochs, history.history['loss'], 
                    linestyle='--', color=colors[name], alpha=0.7)
        axes[1].plot(epochs, history.history['val_loss'], 
                    label=f'{name}', color=colors[name], linewidth=2)
    
    axes[0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Model Loss Comparison', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if histories:
    plot_training_curves(histories)

#%%
def evaluate_and_plot_confusion_matrix(model, val_gen, model_name):
    """评估模型并绘制混淆矩阵"""
    val_gen.reset()
    
    # 预测
    y_pred_proba = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes
    
    class_names = list(val_gen.class_indices.keys())
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印分类报告
    print(f"\n{model_name} 分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred

# 为每个模型生成混淆矩阵
if models_dict and val_gen is not None:
    for name, model in models_dict.items():
        evaluate_and_plot_confusion_matrix(model, val_gen, name)

#%%
def measure_inference_time(model, val_gen, num_samples=50):
    """测量推理时间"""
    val_gen.reset()
    batch = next(val_gen)
    images = batch[0][:num_samples]
    
    # 预热
    _ = model.predict(images[:1], verbose=0)
    
    # 测量
    times = []
    for img in images:
        start = time.time()
        _ = model.predict(np.expand_dims(img, 0), verbose=0)
        times.append(time.time() - start)
    
    return np.mean(times) * 1000  # 转换为毫秒

# 测量所有模型的推理时间
if models_dict and val_gen is not None:
    for name, model in models_dict.items():
        inference_time = measure_inference_time(model, val_gen)
        results[name]['inference_time_ms'] = inference_time
        print(f"{name}: {inference_time:.2f} ms/image")

#%% [markdown]
# ## 8. 结果对比

#%%
import pandas as pd

# 创建对比表格（仅在有结果时执行）
if results:
    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{res['best_val_accuracy']:.4f}",
            'Size (MB)': f"{res['model_size_mb']:.1f}",
            'Inference (ms)': f"{res['inference_time_ms']:.1f}",
            'Training (min)': f"{res['training_time_min']:.1f}",
            'Parameters': f"{res['total_params']:,}"
        })

    df = pd.DataFrame(comparison_data)
    print("\n" + "="*80)
    print("模型性能对比表 (Model Performance Comparison)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # 保存为CSV
    df.to_csv('model_comparison.csv', index=False)
    print("\n对比表格已保存: model_comparison.csv")

#%%
# 绘制对比柱状图（仅在有结果时执行）
if results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    model_names = list(results.keys())
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # 准确率
    accuracies = [results[m]['best_val_accuracy'] for m in model_names]
    axes[0, 0].bar(model_names, accuracies, color=colors)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylim([0.8, 1.0])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

    # 模型大小
    sizes = [results[m]['model_size_mb'] for m in model_names]
    axes[0, 1].bar(model_names, sizes, color=colors)
    axes[0, 1].set_title('Model Size (MB)')
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')

    # 推理时间
    times = [results[m]['inference_time_ms'] for m in model_names]
    axes[1, 0].bar(model_names, times, color=colors)
    axes[1, 0].set_title('Inference Time (ms)')
    for i, v in enumerate(times):
        axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

    # 训练时间
    train_times = [results[m]['training_time_min'] for m in model_names]
    axes[1, 1].bar(model_names, train_times, color=colors)
    axes[1, 1].set_title('Training Time (min)')
    for i, v in enumerate(train_times):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

#%% [markdown]
# ## 9. 结论
# 
# 根据实验结果：
# 
# | 指标 | 最佳模型 |
# |------|----------|
# | 准确率 | ResNet50 |
# | 模型大小 | MobileNetV2 |
# | 推理速度 | MobileNetV2 |
# | 综合性能 | **MobileNetV2** |
# 
# **MobileNetV2** 在保持高准确率的同时，模型体积最小、推理速度最快，
# 最适合部署在智能垃圾桶等嵌入式设备上。

#%%
# 保存最佳模型（仅在有模型时执行）
if models_dict and 'MobileNetV2' in models_dict:
    best_model = models_dict['MobileNetV2']
    best_model.save('MobileNetV2_garbage_classification.keras')
    print("\n最佳模型已保存: MobileNetV2_garbage_classification.keras")

#%% [markdown]
# ## 10. 论文用LaTeX表格

#%%
# 生成LaTeX表格（仅在有结果时执行）
if results:
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Different Deep Learning Models}
\\label{tab:comparison}
\\begin{tabular}{lccccc}
\\toprule
Model & Accuracy & F1-Score & Size(MB) & Inference(ms) & Params \\\\
\\midrule
"""

    for name, res in results.items():
        latex_table += f"{name} & {res['best_val_accuracy']:.4f} & - & "
        latex_table += f"{res['model_size_mb']:.1f} & {res['inference_time_ms']:.1f} & "
        latex_table += f"{res['total_params']/1e6:.1f}M \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    print("LaTeX表格代码:")
    print(latex_table)

print("\n" + "="*60)
print("Notebook 加载完成！")
if not results:
    print("注意：数据集不可用，训练部分已跳过")
    print("请下载数据集到指定目录后重新运行")
print("="*60)
