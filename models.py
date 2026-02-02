"""
模型定义模块 - 迁移学习实现
Model Definition Module - Transfer Learning Implementation

包含三种预训练模型:
1. MobileNetV2 (轻量级 - 本文推荐)
2. ResNet50 (中型)
3. VGG16 (重型)

注意力机制模块:
1. CBAM (Convolutional Block Attention Module)
2. SE-Net (Squeeze-and-Excitation)

优化内容:
- 使用工厂模式减少代码重复
- 添加网络错误处理
- 支持配置驱动的分类头
- 支持注意力机制增强
"""

import logging
import os
from typing import Dict, List, Optional, Any, Type, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, DenseNet121, EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from config import (
    NUM_CLASSES, IMG_SHAPE, FREEZE_BASE, DROPOUT_RATE,
    LEARNING_RATE, MODEL_HEAD_CONFIG, USE_XLA, OUTPUT_DIR, ensure_dir
)

logger = logging.getLogger(__name__)


class ModelBuildError(Exception):
    """模型构建错误"""
    pass


# ============================================================================
# 注意力机制模块 (Attention Modules)
# ============================================================================

class ChannelAttention(layers.Layer):
    """
    通道注意力模块 (Channel Attention Module)

    通过对特征图的通道维度进行全局平均池化和全局最大池化，
    然后通过共享的MLP网络学习通道间的依赖关系。

    Args:
        reduction_ratio: 通道压缩比例，默认16
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)

        # 共享的MLP网络
        self.fc1 = layers.Dense(
            reduced_channels,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc1'
        )
        self.fc2 = layers.Dense(
            channels,
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc2'
        )
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs, return_attention_weights: bool = False):
        # 全局平均池化分支
        avg_pool = self.global_avg_pool(inputs)
        avg_out = self.fc2(self.fc1(avg_pool))

        # 全局最大池化分支
        max_pool = self.global_max_pool(inputs)
        max_out = self.fc2(self.fc1(max_pool))

        # 合并两个分支
        attention = tf.sigmoid(avg_out + max_out)

        # 应用注意力
        output = inputs * attention

        if return_attention_weights:
            return output, attention
        return output

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """
    空间注意力模块 (Spatial Attention Module)

    通过对特征图的空间维度进行通道级的平均池化和最大池化，
    然后通过卷积网络学习空间位置间的依赖关系。

    Args:
        kernel_size: 卷积核大小，默认7
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        # 使用7x7卷积核
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
            name=f'{self.name}_conv'
        )

    def call(self, inputs, return_attention_weights: bool = False):
        # 沿通道维度计算平均值和最大值
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # 拼接
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # 卷积生成注意力图
        attention = self.conv(concat)

        # 应用注意力
        output = inputs * attention

        if return_attention_weights:
            return output, attention
        return output

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    """
    卷积块注意力模块 (Convolutional Block Attention Module)

    CBAM = Channel Attention + Spatial Attention

    先应用通道注意力，再应用空间注意力，顺序处理能够更好地
    捕获"什么"和"在哪里"的信息。

    Args:
        reduction_ratio: 通道注意力的压缩比例，默认16
        spatial_kernel_size: 空间注意力的卷积核大小，默认7

    Example:
        >>> cbam = CBAM(reduction_ratio=16, spatial_kernel_size=7)
        >>> x = tf.random.normal([1, 32, 32, 64])
        >>> output = cbam(x)
        >>> print(output.shape)  # (1, 32, 32, 64)
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        **kwargs
    ):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size

        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio,
            name=f'{self.name}_channel_att' if self.name else 'channel_att'
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
            name=f'{self.name}_spatial_att' if self.name else 'spatial_att'
        )

    def call(
        self,
        inputs,
        return_attention_weights: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        前向传播

        Args:
            inputs: 输入特征图 [B, H, W, C]
            return_attention_weights: 是否返回注意力权重

        Returns:
            如果 return_attention_weights=False: 输出特征图
            如果 return_attention_weights=True: (输出特征图, 注意力权重字典)
        """
        # 通道注意力
        if return_attention_weights:
            x, channel_weights = self.channel_attention(
                inputs, return_attention_weights=True
            )
            # 空间注意力
            output, spatial_weights = self.spatial_attention(
                x, return_attention_weights=True
            )
            attention_weights = {
                'channel': channel_weights,
                'spatial': spatial_weights
            }
            return output, attention_weights
        else:
            x = self.channel_attention(inputs)
            output = self.spatial_attention(x)
            return output

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'spatial_kernel_size': self.spatial_kernel_size
        })
        return config


class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation 模块 (SE Block)

    通过全局信息嵌入和通道间依赖关系建模，自适应地重新校准通道特征响应。

    包含两个操作:
    1. Squeeze: 全局平均池化，将空间维度压缩为1x1
    2. Excitation: 通过两个全连接层学习通道间的依赖关系

    Args:
        reduction_ratio: 通道压缩比例，默认16

    Example:
        >>> se = SEBlock(reduction_ratio=16)
        >>> x = tf.random.normal([1, 32, 32, 64])
        >>> output = se(x)
        >>> print(output.shape)  # (1, 32, 32, 64)
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc1 = layers.Dense(
            reduced_channels,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc1'
        )
        self.fc2 = layers.Dense(
            channels,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc2'
        )
        super(SEBlock, self).build(input_shape)

    def call(
        self,
        inputs,
        return_attention_weights: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        前向传播

        Args:
            inputs: 输入特征图 [B, H, W, C]
            return_attention_weights: 是否返回注意力权重

        Returns:
            如果 return_attention_weights=False: 输出特征图
            如果 return_attention_weights=True: (输出特征图, 通道注意力权重)
        """
        # Squeeze: 全局平均池化
        squeezed = self.global_avg_pool(inputs)

        # Excitation: 学习通道权重
        attention = self.fc1(squeezed)
        attention = self.fc2(attention)

        # Scale: 重新校准
        output = inputs * attention

        if return_attention_weights:
            return output, attention
        return output

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


# ============================================================================
# 预训练模型映射
# ============================================================================

BASE_MODEL_CLASSES: Dict[str, Type] = {
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'VGG16': VGG16,
    'DenseNet121': DenseNet121,
    'EfficientNetB0': EfficientNetB0,
}


# ============================================================================
# 模型构建工具函数
# ============================================================================

def _download_weights_with_retry(
    model_class: Type,
    input_shape: tuple,
    max_retries: int = 3
) -> Any:
    """
    带重试机制的预训练权重下载

    Args:
        model_class: 模型类
        input_shape: 输入形状
        max_retries: 最大重试次数

    Returns:
        base_model: 加载了预训练权重的基模型

    Raises:
        ModelBuildError: 无法下载预训练权重
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            base_model = model_class(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            return base_model
        except Exception as e:
            last_error = e
            logger.warning(f"下载预训练权重失败 (尝试 {attempt + 1}/{max_retries}): {e}")

    # 所有重试都失败，尝试不加载权重
    logger.warning("无法下载预训练权重，使用随机初始化")
    try:
        base_model = model_class(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
        return base_model
    except Exception as e:
        raise ModelBuildError(f"无法构建模型: {e}") from last_error


def _build_classification_head(
    base_model: Model,
    num_classes: int,
    dense_units: List[int],
    dropout_rates: List[float],
    model_name: str
) -> Model:
    """
    构建分类头

    Args:
        base_model: 预训练基模型
        num_classes: 分类类别数
        dense_units: 全连接层单元数列表
        dropout_rates: Dropout比率列表
        model_name: 模型名称

    Returns:
        Model: 完整的模型
    """
    layers_list = [
        base_model,
        layers.GlobalAveragePooling2D(),
    ]

    # 添加全连接层和Dropout
    for i, units in enumerate(dense_units):
        if i < len(dropout_rates):
            layers_list.append(layers.Dropout(dropout_rates[i]))
        layers_list.append(layers.Dense(units, activation='relu'))

    # 最后一个Dropout
    if len(dropout_rates) > len(dense_units):
        layers_list.append(layers.Dropout(dropout_rates[-1]))

    # 输出层
    layers_list.append(layers.Dense(num_classes, activation='softmax'))

    model = models.Sequential(layers_list, name=f'{model_name}_Transfer')
    return model


def build_transfer_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    config: Optional[Dict[str, Any]] = None
) -> Model:
    """
    构建迁移学习模型的通用函数

    Args:
        model_name: 模型名称 ('MobileNetV2', 'ResNet50', 'VGG16')
        num_classes: 分类类别数
        input_shape: 输入图像尺寸
        freeze_base: 是否冻结基模型
        config: 分类头配置，包含 dense_units 和 dropout_rates

    Returns:
        model: Keras模型

    Raises:
        ModelBuildError: 模型构建失败
    """
    if model_name not in BASE_MODEL_CLASSES:
        raise ModelBuildError(
            f"未知模型: {model_name}. 可选: {list(BASE_MODEL_CLASSES.keys())}"
        )

    # 获取配置
    if config is None:
        config = MODEL_HEAD_CONFIG.get(model_name, {
            'dense_units': [256],
            'dropout_rates': [DROPOUT_RATE, DROPOUT_RATE / 2]
        })

    dense_units = config.get('dense_units', [256])
    dropout_rates = config.get('dropout_rates', [DROPOUT_RATE, DROPOUT_RATE / 2])

    # 加载预训练模型
    model_class = BASE_MODEL_CLASSES[model_name]
    base_model = _download_weights_with_retry(model_class, input_shape)

    # 冻结基模型
    base_model.trainable = not freeze_base

    # 构建完整模型
    model = _build_classification_head(
        base_model, num_classes, dense_units, dropout_rates, model_name
    )

    logger.info(f"成功构建模型: {model_name}")
    return model


# ============================================================================
# 带注意力机制的模型构建函数
# ============================================================================

def build_mobilenetv2_cbam(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE,
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
    attention_position: str = 'after_encoder'
) -> Model:
    """
    构建带CBAM注意力机制的MobileNetV2模型
    Build MobileNetV2 with CBAM (Convolutional Block Attention Module)

    CBAM通过通道注意力和空间注意力的顺序组合，能够让模型关注
    "什么"(通道)和"在哪里"(空间)的重要特征。

    Args:
        num_classes: 分类类别数
        input_shape: 输入图像尺寸
        freeze_base: 是否冻结基模型
        dropout_rate: Dropout比率
        reduction_ratio: CBAM通道压缩比例
        spatial_kernel_size: 空间注意力卷积核大小
        attention_position: 注意力模块位置
            - 'after_encoder': 在编码器输出后（默认）
            - 'before_gap': 在全局平均池化前
            - 'after_gap': 在全局平均池化后（仅通道注意力）
            - 'multi_scale': 多尺度注意力（在多个位置插入）

    Returns:
        Model: 带CBAM的MobileNetV2模型

    Example:
        >>> model = build_mobilenetv2_cbam(num_classes=12)
        >>> model.summary()
    """
    # 加载预训练的MobileNetV2
    base_model = _download_weights_with_retry(MobileNetV2, input_shape)
    base_model.trainable = not freeze_base

    # 构建模型
    inputs = layers.Input(shape=input_shape, name='input')

    # 获取MobileNetV2的输出
    x = base_model(inputs)

    if attention_position == 'after_encoder':
        # 在编码器输出后添加CBAM
        x = CBAM(
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
            name='cbam'
        )(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)

    elif attention_position == 'before_gap':
        # 与after_encoder相同
        x = CBAM(
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
            name='cbam'
        )(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)

    elif attention_position == 'after_gap':
        # 在GAP后只用通道注意力（无空间维度）
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        # 此时只能用SE-like的通道注意力
        x = layers.Reshape((1, 1, -1))(x)
        x = ChannelAttention(
            reduction_ratio=reduction_ratio,
            name='channel_attention'
        )(x)
        x = layers.Flatten()(x)

    elif attention_position == 'multi_scale':
        # 多尺度注意力 - 在多个位置添加CBAM
        # 获取MobileNetV2的中间层输出
        intermediate_model = Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer('block_6_expand_relu').output,  # 早期特征
                base_model.get_layer('block_13_expand_relu').output,  # 中期特征
                base_model.output  # 最终特征
            ]
        )
        intermediate_model.trainable = not freeze_base

        early_feat, mid_feat, final_feat = intermediate_model(inputs)

        # 对每个尺度应用CBAM
        early_att = CBAM(reduction_ratio=reduction_ratio, name='cbam_early')(early_feat)
        mid_att = CBAM(reduction_ratio=reduction_ratio, name='cbam_mid')(mid_feat)
        final_att = CBAM(reduction_ratio=reduction_ratio, name='cbam_final')(final_feat)

        # 对每个特征进行全局平均池化
        early_gap = layers.GlobalAveragePooling2D(name='gap_early')(early_att)
        mid_gap = layers.GlobalAveragePooling2D(name='gap_mid')(mid_att)
        final_gap = layers.GlobalAveragePooling2D(name='gap_final')(final_att)

        # 拼接多尺度特征
        x = layers.Concatenate(name='multi_scale_concat')([early_gap, mid_gap, final_gap])

    else:
        raise ValueError(f"未知的attention_position: {attention_position}")

    # 分类头
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='MobileNetV2_CBAM')

    logger.info(f"成功构建模型: MobileNetV2_CBAM (attention_position={attention_position})")
    return model


def build_mobilenetv2_se(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE,
    reduction_ratio: int = 16,
    attention_position: str = 'after_encoder'
) -> Model:
    """
    构建带SE-Net注意力机制的MobileNetV2模型
    Build MobileNetV2 with SE-Net (Squeeze-and-Excitation)

    SE-Net通过Squeeze和Excitation操作，自适应地重新校准通道特征响应，
    使模型能够学习到更有判别力的特征表示。

    Args:
        num_classes: 分类类别数
        input_shape: 输入图像尺寸
        freeze_base: 是否冻结基模型
        dropout_rate: Dropout比率
        reduction_ratio: SE模块通道压缩比例
        attention_position: 注意力模块位置
            - 'after_encoder': 在编码器输出后（默认）
            - 'before_gap': 在全局平均池化前
            - 'multi_scale': 多尺度注意力

    Returns:
        Model: 带SE-Net的MobileNetV2模型

    Example:
        >>> model = build_mobilenetv2_se(num_classes=12)
        >>> model.summary()
    """
    # 加载预训练的MobileNetV2
    base_model = _download_weights_with_retry(MobileNetV2, input_shape)
    base_model.trainable = not freeze_base

    # 构建模型
    inputs = layers.Input(shape=input_shape, name='input')

    # 获取MobileNetV2的输出
    x = base_model(inputs)

    if attention_position in ['after_encoder', 'before_gap']:
        # 在编码器输出后添加SE Block
        x = SEBlock(reduction_ratio=reduction_ratio, name='se_block')(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)

    elif attention_position == 'multi_scale':
        # 多尺度SE注意力
        intermediate_model = Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer('block_6_expand_relu').output,
                base_model.get_layer('block_13_expand_relu').output,
                base_model.output
            ]
        )
        intermediate_model.trainable = not freeze_base

        early_feat, mid_feat, final_feat = intermediate_model(inputs)

        # 对每个尺度应用SE Block
        early_se = SEBlock(reduction_ratio=reduction_ratio, name='se_early')(early_feat)
        mid_se = SEBlock(reduction_ratio=reduction_ratio, name='se_mid')(mid_feat)
        final_se = SEBlock(reduction_ratio=reduction_ratio, name='se_final')(final_feat)

        # 对每个特征进行全局平均池化
        early_gap = layers.GlobalAveragePooling2D(name='gap_early')(early_se)
        mid_gap = layers.GlobalAveragePooling2D(name='gap_mid')(mid_se)
        final_gap = layers.GlobalAveragePooling2D(name='gap_final')(final_se)

        # 拼接多尺度特征
        x = layers.Concatenate(name='multi_scale_concat')([early_gap, mid_gap, final_gap])

    else:
        raise ValueError(f"未知的attention_position: {attention_position}")

    # 分类头
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='MobileNetV2_SE')

    logger.info(f"成功构建模型: MobileNetV2_SE (attention_position={attention_position})")
    return model


def build_mobilenetv2_attention(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE,
    attention_type: str = 'cbam',
    reduction_ratio: int = 16,
    attention_position: str = 'after_encoder',
    **kwargs
) -> Model:
    """
    构建带注意力机制的MobileNetV2模型的通用接口
    Universal interface for building MobileNetV2 with attention mechanisms

    Args:
        num_classes: 分类类别数
        input_shape: 输入图像尺寸
        freeze_base: 是否冻结基模型
        dropout_rate: Dropout比率
        attention_type: 注意力类型 ('cbam', 'se', 'none')
        reduction_ratio: 通道压缩比例
        attention_position: 注意力模块位置
        **kwargs: 其他参数

    Returns:
        Model: 带注意力的MobileNetV2模型
    """
    if attention_type.lower() == 'cbam':
        return build_mobilenetv2_cbam(
            num_classes=num_classes,
            input_shape=input_shape,
            freeze_base=freeze_base,
            dropout_rate=dropout_rate,
            reduction_ratio=reduction_ratio,
            attention_position=attention_position,
            **kwargs
        )
    elif attention_type.lower() == 'se':
        return build_mobilenetv2_se(
            num_classes=num_classes,
            input_shape=input_shape,
            freeze_base=freeze_base,
            dropout_rate=dropout_rate,
            reduction_ratio=reduction_ratio,
            attention_position=attention_position
        )
    elif attention_type.lower() == 'none':
        return build_mobilenetv2(
            num_classes=num_classes,
            input_shape=input_shape,
            freeze_base=freeze_base,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"未知的attention_type: {attention_type}. 可选: 'cbam', 'se', 'none'")


# ============================================================================
# 注意力权重可视化
# ============================================================================

class AttentionVisualizer:
    """
    注意力权重可视化器
    Attention Weight Visualizer

    用于可视化CBAM和SE-Net的注意力权重，帮助理解模型关注的区域。

    Example:
        >>> visualizer = AttentionVisualizer(model)
        >>> visualizer.visualize_attention(image, save_path='attention.png')
    """

    def __init__(self, model: Model):
        """
        初始化可视化器

        Args:
            model: 带注意力机制的模型
        """
        self.model = model
        self.attention_layers = self._find_attention_layers()

    def _find_attention_layers(self) -> Dict[str, layers.Layer]:
        """查找模型中的注意力层"""
        attention_layers = {}
        for layer in self.model.layers:
            if isinstance(layer, (CBAM, SEBlock, ChannelAttention, SpatialAttention)):
                attention_layers[layer.name] = layer
            # 递归查找子模型中的注意力层
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, (CBAM, SEBlock, ChannelAttention, SpatialAttention)):
                        attention_layers[sublayer.name] = sublayer
        return attention_layers

    def get_attention_weights(
        self,
        image: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        获取给定图像的注意力权重

        Args:
            image: 输入图像 [H, W, C] 或 [1, H, W, C]

        Returns:
            Dict: 注意力权重字典
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        attention_weights = {}

        # 为每个注意力层创建子模型获取注意力权重
        for name, layer in self.attention_layers.items():
            try:
                # 获取该层的输入
                layer_input = self._get_layer_input(layer, image)
                if layer_input is not None:
                    if isinstance(layer, CBAM):
                        _, weights = layer(layer_input, return_attention_weights=True)
                        attention_weights[name] = {
                            'channel': weights['channel'].numpy(),
                            'spatial': weights['spatial'].numpy()
                        }
                    elif isinstance(layer, (SEBlock, ChannelAttention)):
                        _, weights = layer(layer_input, return_attention_weights=True)
                        attention_weights[name] = {'channel': weights.numpy()}
                    elif isinstance(layer, SpatialAttention):
                        _, weights = layer(layer_input, return_attention_weights=True)
                        attention_weights[name] = {'spatial': weights.numpy()}
            except Exception as e:
                logger.warning(f"无法获取 {name} 的注意力权重: {e}")

        return attention_weights

    def _get_layer_input(
        self,
        target_layer: layers.Layer,
        image: np.ndarray
    ) -> Optional[tf.Tensor]:
        """获取目标层的输入"""
        try:
            # 创建一个子模型来获取目标层之前的输出
            for i, layer in enumerate(self.model.layers):
                if layer == target_layer and i > 0:
                    intermediate_model = Model(
                        inputs=self.model.input,
                        outputs=self.model.layers[i-1].output
                    )
                    return intermediate_model(image, training=False)
            # 如果是第一层之后，直接用输入
            return tf.constant(image)
        except Exception:
            return None

    def visualize_attention(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        可视化注意力权重

        Args:
            image: 输入图像 [H, W, C]，值范围 [0, 1] 或 [0, 255]
            save_path: 保存路径（可选）
            figsize: 图像大小

        Returns:
            matplotlib Figure对象
        """
        if len(image.shape) == 4:
            image = image[0]

        # 归一化图像到 [0, 1]
        if image.max() > 1:
            image = image / 255.0

        attention_weights = self.get_attention_weights(image)

        if not attention_weights:
            logger.warning("未找到注意力权重")
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(image)
            ax.set_title("Original Image (No attention weights found)")
            ax.axis('off')
            return fig

        # 计算子图数量
        n_plots = 1  # 原图
        for name, weights in attention_weights.items():
            n_plots += len(weights)

        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        # 绘制原图
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 绘制注意力图
        plot_idx = 1
        for name, weights in attention_weights.items():
            for weight_type, weight in weights.items():
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]

                if weight_type == 'spatial':
                    # 空间注意力 - 热力图叠加
                    att_map = weight[0, :, :, 0]
                    att_map_resized = tf.image.resize(
                        att_map[..., np.newaxis],
                        image.shape[:2]
                    ).numpy()[:, :, 0]

                    ax.imshow(image)
                    ax.imshow(att_map_resized, cmap='jet', alpha=0.5)
                    ax.set_title(f"{name}\n{weight_type.capitalize()} Attention")

                elif weight_type == 'channel':
                    # 通道注意力 - 条形图
                    channel_weights = weight.flatten()
                    ax.bar(range(len(channel_weights)), channel_weights)
                    ax.set_xlabel("Channel")
                    ax.set_ylabel("Weight")
                    ax.set_title(f"{name}\n{weight_type.capitalize()} Attention")

                ax.axis('off') if weight_type == 'spatial' else None
                plot_idx += 1

        # 隐藏多余的子图
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path) or '.')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"注意力可视化已保存: {save_path}")

        return fig

    def visualize_channel_importance(
        self,
        images: np.ndarray,
        save_path: Optional[str] = None,
        top_k: int = 20
    ) -> plt.Figure:
        """
        可视化通道重要性（基于多个样本的平均）

        Args:
            images: 多个输入图像 [N, H, W, C]
            save_path: 保存路径
            top_k: 显示前K个最重要的通道

        Returns:
            matplotlib Figure对象
        """
        all_channel_weights = []

        for img in images:
            weights = self.get_attention_weights(img)
            for name, w in weights.items():
                if 'channel' in w:
                    all_channel_weights.append(w['channel'].flatten())

        if not all_channel_weights:
            logger.warning("未找到通道注意力权重")
            return None

        # 计算平均通道重要性
        avg_weights = np.mean(all_channel_weights, axis=0)
        top_indices = np.argsort(avg_weights)[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(range(top_k), avg_weights[top_indices])
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"Ch{i}" for i in top_indices], rotation=45)
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Average Attention Weight")
        ax.set_title(f"Top {top_k} Important Channels")

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path) or '.')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"通道重要性可视化已保存: {save_path}")

        return fig


def visualize_attention_maps(
    model: Model,
    image: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    便捷函数：可视化模型的注意力图

    Args:
        model: 带注意力机制的模型
        image: 输入图像
        save_path: 保存路径

    Returns:
        matplotlib Figure对象
    """
    visualizer = AttentionVisualizer(model)
    return visualizer.visualize_attention(image, save_path)


# ============================================================================
# 保持向后兼容的函数
# ============================================================================

def build_mobilenetv2(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    构建基于MobileNetV2的迁移学习模型
    Build MobileNetV2-based transfer learning model

    MobileNetV2特点:
    - 倒残差结构 (Inverted Residuals)
    - 深度可分离卷积 (Depthwise Separable Convolution)
    - 参数量小 (~3.4M), 适合嵌入式部署
    """
    config = {
        'dense_units': [128],
        'dropout_rates': [dropout_rate, dropout_rate / 2]
    }
    return build_transfer_model('MobileNetV2', num_classes, input_shape, freeze_base, config)


def build_resnet50(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    构建基于ResNet50的迁移学习模型
    Build ResNet50-based transfer learning model

    ResNet50特点:
    - 残差连接 (Skip Connections)
    - 50层深度网络
    - 参数量中等 (~25.6M)
    """
    config = {
        'dense_units': [256],
        'dropout_rates': [dropout_rate, dropout_rate / 2]
    }
    return build_transfer_model('ResNet50', num_classes, input_shape, freeze_base, config)


def build_vgg16(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    构建基于VGG16的迁移学习模型
    Build VGG16-based transfer learning model

    VGG16特点:
    - 简单的3x3卷积堆叠
    - 16层网络
    - 参数量大 (~138M), 计算量大
    """
    config = {
        'dense_units': [512, 256],
        'dropout_rates': [dropout_rate, dropout_rate / 2, dropout_rate / 2]
    }
    return build_transfer_model('VGG16', num_classes, input_shape, freeze_base, config)


def build_densenet121(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    构建基于DenseNet121的迁移学习模型
    Build DenseNet121-based transfer learning model

    DenseNet121特点:
    - 密集连接 (Dense Connections)
    - 特征复用，减少参数量
    - 参数量 (~8M), 性能优秀
    """
    config = {
        'dense_units': [256],
        'dropout_rates': [dropout_rate, dropout_rate / 2]
    }
    return build_transfer_model('DenseNet121', num_classes, input_shape, freeze_base, config)


def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    freeze_base: bool = FREEZE_BASE,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    构建基于EfficientNet-B0的迁移学习模型
    Build EfficientNet-B0-based transfer learning model

    EfficientNet-B0特点:
    - 复合缩放策略 (Compound Scaling)
    - 高效的网络架构搜索
    - 参数量小 (~5.3M), 性能卓越
    """
    config = {
        'dense_units': [128],
        'dropout_rates': [dropout_rate, dropout_rate / 2]
    }
    return build_transfer_model('EfficientNetB0', num_classes, input_shape, freeze_base, config)


def build_model(model_name: str, num_classes: int = NUM_CLASSES, **kwargs) -> Model:
    """
    根据模型名称构建模型
    Build model by name

    Args:
        model_name: 模型名称
            - 标准模型: 'MobileNetV2', 'ResNet50', 'VGG16', 'DenseNet121', 'EfficientNetB0'
            - 注意力模型: 'MobileNetV2_CBAM', 'MobileNetV2_SE'
        num_classes: 分类类别数
        **kwargs: 其他参数

    Returns:
        model: Keras模型
    """
    # 检查是否是带注意力的模型
    model_name_upper = model_name.upper().replace('-', '_').replace(' ', '_')

    if model_name_upper == 'MOBILENETV2_CBAM':
        return build_mobilenetv2_cbam(num_classes=num_classes, **kwargs)
    elif model_name_upper == 'MOBILENETV2_SE':
        return build_mobilenetv2_se(num_classes=num_classes, **kwargs)
    else:
        return build_transfer_model(model_name, num_classes, **kwargs)


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    """
    编译模型
    Compile model with optimizer, loss, and metrics

    Args:
        model: Keras模型
        learning_rate: 学习率

    Returns:
        model: 编译后的模型
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
        jit_compile=USE_XLA  # 可选XLA编译加速
    )
    return model


def get_model_info(model: Model) -> Dict[str, Any]:
    """
    获取模型信息
    Get model information

    Args:
        model: Keras模型

    Returns:
        dict: 模型信息字典
    """
    # 计算参数量
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    total_params = trainable_params + non_trainable_params

    # 估算模型大小 (假设float32, 4 bytes per param)
    model_size_mb = total_params * 4 / (1024 * 1024)

    info = {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb
    }

    return info


def print_model_summary(model: Model) -> Dict[str, Any]:
    """
    打印模型摘要信息
    Print model summary

    Args:
        model: Keras模型

    Returns:
        dict: 模型信息字典
    """
    info = get_model_info(model)

    print("\n" + "="*60)
    print(f"模型名称: {info['name']}")
    print("="*60)
    print(f"总参数量: {info['total_params']:,}")
    print(f"可训练参数: {info['trainable_params']:,}")
    print(f"不可训练参数: {info['non_trainable_params']:,}")
    print(f"估计模型大小: {info['model_size_mb']:.2f} MB")
    print("="*60 + "\n")

    return info


def unfreeze_base_model(model: Model, num_layers_to_unfreeze: int = 20) -> None:
    """
    解冻基模型的部分层用于微调
    Unfreeze some layers of base model for fine-tuning

    Args:
        model: Keras模型
        num_layers_to_unfreeze: 要解冻的层数
    """
    # 获取基模型 - 处理不同的模型结构
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        base_model = model.layers[0] if hasattr(model.layers[0], 'layers') else None

    if base_model is None:
        logger.warning("无法找到基模型进行解冻")
        return

    # 解冻最后几层
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    print(f"已解冻基模型的最后 {num_layers_to_unfreeze} 层用于微调")


def build_simple_cnn(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE
) -> Model:
    """
    构建简单的CNN基线模型（不使用迁移学习）
    Build simple CNN baseline model (without transfer learning)

    用于对比迁移学习的效果提升
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='Simple_CNN_Baseline')

    return model


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试模型构建...")
    print("="*60)

    # 测试标准模型
    for model_name in ['MobileNetV2', 'ResNet50', 'VGG16']:
        print(f"\n构建 {model_name}...")
        try:
            model = build_model(model_name)
            model = compile_model(model)
            print_model_summary(model)
        except ModelBuildError as e:
            print(f"构建失败: {e}")

    # 测试带注意力的模型
    print("\n" + "="*60)
    print("测试注意力机制模型...")
    print("="*60)

    # 测试 CBAM
    print("\n构建 MobileNetV2_CBAM...")
    try:
        model_cbam = build_mobilenetv2_cbam(num_classes=12)
        model_cbam = compile_model(model_cbam)
        print_model_summary(model_cbam)
    except Exception as e:
        print(f"构建失败: {e}")

    # 测试 SE-Net
    print("\n构建 MobileNetV2_SE...")
    try:
        model_se = build_mobilenetv2_se(num_classes=12)
        model_se = compile_model(model_se)
        print_model_summary(model_se)
    except Exception as e:
        print(f"构建失败: {e}")

    # 测试多尺度CBAM
    print("\n构建 MobileNetV2_CBAM (multi_scale)...")
    try:
        model_cbam_ms = build_mobilenetv2_cbam(
            num_classes=12,
            attention_position='multi_scale'
        )
        model_cbam_ms = compile_model(model_cbam_ms)
        print_model_summary(model_cbam_ms)
    except Exception as e:
        print(f"构建失败: {e}")

    print("\n所有模型测试完成!")
