"""
训练模块 - 高级模型训练与回调函数
Training Module - Advanced Model Training and Callbacks

功能:
1. 混合精度训练 (Mixed Precision Training)
2. 学习率调度优化 (Warmup+Cosine, OneCycleLR, SGDR)
3. 梯度累积 (Gradient Accumulation)
4. 标签平滑 (Label Smoothing)
5. Mixup / CutMix 数据增强
6. 指数移动平均 (EMA)
7. 训练回调函数配置
8. 模型训练流程
9. 微调支持
10. 训练结果保存
11. 内存管理
"""

import gc
import os
import time
import json
import logging
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
    Callback,
    LearningRateScheduler
)
from tensorflow.keras import Model

from config import (
    OUTPUT_DIR, MODEL_DIR, LOG_DIR, EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR,
    USE_MIXED_PRECISION, LABEL_SMOOTHING,
    LR_SCHEDULE_TYPE, WARMUP_EPOCHS, WARMUP_LR_INIT,
    ONE_CYCLE_MAX_LR, ONE_CYCLE_PCT_START,
    SGDR_T_0, SGDR_T_MULT, SGDR_LR_MIN,
    GRADIENT_ACCUMULATION_STEPS,
    USE_MIXUP, MIXUP_ALPHA, USE_CUTMIX, CUTMIX_ALPHA, MIXUP_CUTMIX_PROB,
    USE_EMA, EMA_DECAY,
    ensure_dir
)
from models import unfreeze_base_model

logger = logging.getLogger(__name__)


# ==================== 混合精度训练设置 ====================

def setup_mixed_precision(enable: bool = USE_MIXED_PRECISION) -> bool:
    """
    设置混合精度训练
    Setup mixed precision training

    Args:
        enable: 是否启用混合精度

    Returns:
        bool: 是否成功启用
    """
    if not enable:
        logger.info("混合精度训练已禁用")
        return False

    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"混合精度训练已启用: {policy.name}")
        logger.info(f"  计算dtype: {policy.compute_dtype}")
        logger.info(f"  变量dtype: {policy.variable_dtype}")
        return True
    except Exception as e:
        logger.warning(f"无法启用混合精度训练: {e}")
        return False


def get_mixed_precision_loss_scale_optimizer(
    optimizer: tf.keras.optimizers.Optimizer
) -> tf.keras.optimizers.Optimizer:
    """
    为混合精度训练包装优化器（仅在需要时）
    Wrap optimizer for mixed precision training (if needed)

    注意: TensorFlow 2.4+ 会自动处理损失缩放
    """
    return optimizer


# ==================== 学习率调度器 ====================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup + 余弦退火学习率调度
    Warmup + Cosine Annealing learning rate schedule

    学习率变化:
    1. Warmup阶段: 从warmup_lr线性增长到initial_lr
    2. 余弦退火阶段: 从initial_lr按余弦函数衰减到min_lr
    """

    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        decay_steps: int,
        warmup_lr: float = 1e-7,
        min_lr: float = 1e-7,
        name: str = "WarmupCosineDecay"
    ):
        """
        Args:
            initial_lr: 目标学习率（warmup结束后的学习率）
            warmup_steps: warmup步数
            decay_steps: 总衰减步数（不包括warmup）
            warmup_lr: warmup起始学习率
            min_lr: 最小学习率
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self._name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Warmup阶段: 线性增长
        warmup_lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * (step / warmup_steps)

        # 余弦退火阶段
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_step / decay_steps))
        cosine_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        # 根据当前步数选择学习率
        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "warmup_lr": self.warmup_lr,
            "min_lr": self.min_lr,
            "name": self._name
        }


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    OneCycleLR学习率调度
    OneCycle learning rate schedule (Smith, 2018)

    特点:
    1. 学习率先升后降，呈现"超级收敛"效果
    2. 可以使用更大的学习率，训练更快
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        name: str = "OneCycleLR"
    ):
        """
        Args:
            max_lr: 最大学习率
            total_steps: 总训练步数
            pct_start: 上升阶段占比
            div_factor: 初始学习率 = max_lr / div_factor
            final_div_factor: 最终学习率 = max_lr / final_div_factor
        """
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self._name = name

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step_up = tf.cast(self.step_up, tf.float32)
        step_down = tf.cast(self.step_down, tf.float32)

        # 上升阶段: 余弦插值从initial_lr到max_lr
        up_progress = step / step_up
        up_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
            0.5 * (1 - tf.cos(math.pi * up_progress))
        )

        # 下降阶段: 余弦插值从max_lr到final_lr
        down_progress = (step - step_up) / step_down
        down_lr = self.final_lr + (self.max_lr - self.final_lr) * (
            0.5 * (1 + tf.cos(math.pi * down_progress))
        )

        lr = tf.where(step < step_up, up_lr, down_lr)
        return lr

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "name": self._name
        }


class SGDRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    SGDR: 带热重启的随机梯度下降
    Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)

    特点:
    1. 周期性重启学习率，帮助跳出局部最优
    2. 每次重启后周期可以倍增
    """

    def __init__(
        self,
        initial_lr: float,
        t_0: int,
        t_mult: int = 2,
        lr_min: float = 1e-7,
        name: str = "SGDRSchedule"
    ):
        """
        Args:
            initial_lr: 初始/最大学习率
            t_0: 首次重启的周期（步数）
            t_mult: 周期倍增因子
            lr_min: 最小学习率
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_min = lr_min
        self._name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t_0 = tf.cast(self.t_0, tf.float32)
        t_mult = tf.cast(self.t_mult, tf.float32)

        if self.t_mult == 1:
            # 固定周期
            t_cur = tf.math.mod(step, t_0)
            t_i = t_0
        else:
            # 计算当前处于哪个周期
            # 使用几何级数公式
            n_restarts = tf.math.floor(
                tf.math.log(step / t_0 * (t_mult - 1) + 1) / tf.math.log(t_mult)
            )
            n_restarts = tf.maximum(n_restarts, 0.0)

            # 当前周期的起始步数
            if t_mult == 1:
                t_start = t_0 * n_restarts
            else:
                t_start = t_0 * (tf.pow(t_mult, n_restarts) - 1) / (t_mult - 1)

            # 当前周期的长度
            t_i = t_0 * tf.pow(t_mult, n_restarts)

            # 当前周期内的步数
            t_cur = step - t_start
            t_cur = tf.maximum(t_cur, 0.0)

        # 余弦退火
        lr = self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * (
            1 + tf.cos(math.pi * t_cur / t_i)
        )
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "t_0": self.t_0,
            "t_mult": self.t_mult,
            "lr_min": self.lr_min,
            "name": self._name
        }


def get_lr_schedule(
    schedule_type: str,
    steps_per_epoch: int,
    epochs: int,
    initial_lr: float = LEARNING_RATE
) -> Union[tf.keras.optimizers.schedules.LearningRateSchedule, float]:
    """
    获取学习率调度器
    Get learning rate scheduler

    Args:
        schedule_type: 调度器类型 ('warmup_cosine', 'one_cycle', 'sgdr', 'constant')
        steps_per_epoch: 每epoch步数
        epochs: 总轮数
        initial_lr: 初始学习率

    Returns:
        学习率调度器或常数学习率
    """
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    if schedule_type == 'warmup_cosine':
        logger.info(f"使用 Warmup + Cosine Annealing 学习率调度")
        return WarmupCosineDecay(
            initial_lr=initial_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            warmup_lr=WARMUP_LR_INIT,
            min_lr=MIN_LR
        )

    elif schedule_type == 'one_cycle':
        logger.info(f"使用 OneCycleLR 学习率调度")
        return OneCycleLR(
            max_lr=ONE_CYCLE_MAX_LR,
            total_steps=total_steps,
            pct_start=ONE_CYCLE_PCT_START
        )

    elif schedule_type == 'sgdr':
        logger.info(f"使用 SGDR (带热重启) 学习率调度")
        return SGDRSchedule(
            initial_lr=initial_lr,
            t_0=SGDR_T_0 * steps_per_epoch,
            t_mult=SGDR_T_MULT,
            lr_min=SGDR_LR_MIN
        )

    else:
        logger.info(f"使用常数学习率: {initial_lr}")
        return initial_lr


# ==================== Mixup / CutMix 数据增强 ====================

def mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Mixup数据增强
    Mixup data augmentation (Zhang et al., 2017)

    通过线性插值混合两个样本及其标签

    Args:
        images: 图像批次 [B, H, W, C]
        labels: 标签批次 [B, num_classes] (one-hot)
        alpha: Beta分布参数，越大混合程度越高

    Returns:
        Tuple: (混合后的图像, 混合后的标签)
    """
    batch_size = tf.shape(images)[0]

    # 从Beta分布采样混合系数
    lam = tf.numpy_function(
        lambda a: np.random.beta(a, a, size=1).astype(np.float32)[0],
        [alpha],
        tf.float32
    )
    lam = tf.maximum(lam, 1 - lam)  # 确保lam >= 0.5

    # 随机打乱索引
    indices = tf.random.shuffle(tf.range(batch_size))

    # 混合图像和标签
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def cutmix(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 1.0
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    CutMix数据增强
    CutMix data augmentation (Yun et al., 2019)

    将一个图像的矩形区域剪切并粘贴到另一个图像上

    Args:
        images: 图像批次 [B, H, W, C]
        labels: 标签批次 [B, num_classes] (one-hot)
        alpha: Beta分布参数

    Returns:
        Tuple: (CutMix后的图像, 混合后的标签)
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    # 从Beta分布采样
    lam = tf.numpy_function(
        lambda a: np.random.beta(a, a, size=1).astype(np.float32)[0],
        [alpha],
        tf.float32
    )

    # 计算剪切区域大小
    cut_ratio = tf.sqrt(1 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)

    # 随机选择剪切区域中心
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)

    # 计算边界框
    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

    # 随机打乱索引
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)

    # 创建掩码
    mask_shape = [1, img_h, img_w, 1]
    mask = tf.ones(mask_shape)

    # 在掩码中创建剪切区域
    paddings = [[0, 0], [bby1, img_h - bby2], [bbx1, img_w - bbx2], [0, 0]]
    cut_region = tf.ones([1, bby2 - bby1, bbx2 - bbx1, 1])
    cut_mask = tf.pad(cut_region, paddings)
    mask = mask - cut_mask

    # 应用CutMix
    mixed_images = images * mask + shuffled_images * (1 - mask)

    # 计算实际混合比例
    actual_lam = 1 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1 - actual_lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def mixup_cutmix_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    use_mixup: bool = USE_MIXUP,
    use_cutmix: bool = USE_CUTMIX,
    mixup_alpha: float = MIXUP_ALPHA,
    cutmix_alpha: float = CUTMIX_ALPHA,
    prob: float = MIXUP_CUTMIX_PROB
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    随机应用Mixup或CutMix
    Randomly apply Mixup or CutMix

    Args:
        images: 图像批次
        labels: 标签批次
        use_mixup: 是否使用Mixup
        use_cutmix: 是否使用CutMix
        mixup_alpha: Mixup的alpha参数
        cutmix_alpha: CutMix的alpha参数
        prob: 使用Mixup的概率（1-prob为CutMix）

    Returns:
        Tuple: (增强后的图像, 混合后的标签)
    """
    if not use_mixup and not use_cutmix:
        return images, labels

    if use_mixup and not use_cutmix:
        return mixup(images, labels, mixup_alpha)

    if use_cutmix and not use_mixup:
        return cutmix(images, labels, cutmix_alpha)

    # 随机选择Mixup或CutMix
    if tf.random.uniform([]) < prob:
        return mixup(images, labels, mixup_alpha)
    else:
        return cutmix(images, labels, cutmix_alpha)


# ==================== EMA (指数移动平均) ====================

class EMACallback(Callback):
    """
    指数移动平均回调
    Exponential Moving Average callback

    EMA通过维护模型权重的移动平均来提高模型的泛化能力
    在训练期间更新EMA权重，在评估/推理时使用EMA权重
    """

    def __init__(self, decay: float = 0.999):
        """
        Args:
            decay: EMA衰减率，越接近1越平滑
        """
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        self.backup_weights = None

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """训练开始时初始化EMA权重"""
        self.ema_weights = [tf.Variable(w, trainable=False)
                           for w in self.model.get_weights()]

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """每个batch结束后更新EMA权重"""
        if self.ema_weights is None:
            return

        for ema_w, model_w in zip(self.ema_weights, self.model.get_weights()):
            ema_w.assign(self.decay * ema_w + (1 - self.decay) * model_w)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """每个epoch结束时，使用EMA权重进行验证"""
        # 备份当前权重
        self.backup_weights = self.model.get_weights()
        # 应用EMA权重
        self.model.set_weights([w.numpy() for w in self.ema_weights])

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """每个epoch开始时，恢复训练权重"""
        if self.backup_weights is not None:
            self.model.set_weights(self.backup_weights)
            self.backup_weights = None

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """训练结束时，应用EMA权重作为最终权重"""
        if self.ema_weights is not None:
            self.model.set_weights([w.numpy() for w in self.ema_weights])
            logger.info("已应用EMA权重作为最终模型权重")

    def get_ema_weights(self) -> List:
        """获取EMA权重"""
        if self.ema_weights is None:
            return None
        return [w.numpy() for w in self.ema_weights]


# ==================== 梯度累积 ====================

class GradientAccumulationModel(tf.keras.Model):
    """
    梯度累积模型包装器
    Gradient Accumulation Model Wrapper

    用于在显存有限时模拟大batch训练
    通过累积多个小batch的梯度，然后一次性更新权重
    """

    def __init__(self, model: Model, accumulation_steps: int = 1):
        """
        Args:
            model: 原始Keras模型
            accumulation_steps: 梯度累积步数
        """
        super().__init__()
        self.inner_model = model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulator = None
        self.step_count = None

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        """重写compile以设置累积器"""
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        # 初始化梯度累积器
        self.gradient_accumulator = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.inner_model.trainable_variables
        ]
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, inputs, training=None):
        return self.inner_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # 缩放损失以平均梯度
            scaled_loss = loss / self.accumulation_steps

        # 计算梯度
        gradients = tape.gradient(scaled_loss, self.inner_model.trainable_variables)

        # 累积梯度
        for accum, grad in zip(self.gradient_accumulator, gradients):
            if grad is not None:
                accum.assign_add(grad)

        # 更新步数计数
        self.step_count.assign_add(1)

        # 检查是否应该应用梯度
        def apply_gradients():
            # 应用累积的梯度
            self.optimizer.apply_gradients(
                zip(self.gradient_accumulator, self.inner_model.trainable_variables)
            )
            # 重置累积器
            for accum in self.gradient_accumulator:
                accum.assign(tf.zeros_like(accum))
            self.step_count.assign(0)

        def no_op():
            pass

        tf.cond(
            self.step_count >= self.accumulation_steps,
            apply_gradients,
            no_op
        )

        # 更新指标
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @property
    def layers(self):
        return self.inner_model.layers

    def get_weights(self):
        return self.inner_model.get_weights()

    def set_weights(self, weights):
        self.inner_model.set_weights(weights)

    def save_weights(self, filepath, **kwargs):
        self.inner_model.save_weights(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        self.inner_model.load_weights(filepath, **kwargs)


def wrap_model_with_gradient_accumulation(
    model: Model,
    accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
) -> Model:
    """
    用梯度累积包装模型
    Wrap model with gradient accumulation

    Args:
        model: 原始模型
        accumulation_steps: 累积步数

    Returns:
        包装后的模型（如果accumulation_steps > 1）
    """
    if accumulation_steps <= 1:
        return model

    logger.info(f"启用梯度累积: {accumulation_steps} 步")
    return GradientAccumulationModel(model, accumulation_steps)


# ==================== 自定义回调 ====================

class TimeHistory(Callback):
    """
    自定义回调：记录每个epoch的训练时间
    Custom callback: record time for each epoch
    """

    def __init__(self):
        super().__init__()
        self.times: List[float] = []
        self.epoch_time_start: Optional[float] = None

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        self.times = []
        self.epoch_time_start = None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if self.epoch_time_start is not None:
            self.times.append(time.time() - self.epoch_time_start)


class LearningRateLogger(Callback):
    """
    学习率日志回调
    Learning rate logging callback
    """

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if logs is None:
            logs = {}

        # 获取当前学习率
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # 如果是调度器，获取当前步数的学习率
            current_lr = float(lr(self.model.optimizer.iterations))
        else:
            current_lr = float(tf.keras.backend.get_value(lr))

        logs['lr'] = current_lr
        logger.info(f"Epoch {epoch + 1}: 学习率 = {current_lr:.2e}")


# ==================== 回调函数配置 ====================

def get_callbacks(
    model_name: str,
    output_dir: str = OUTPUT_DIR,
    use_ema: bool = USE_EMA,
    ema_decay: float = EMA_DECAY
) -> List[Callback]:
    """
    获取训练回调函数
    Get training callbacks

    Args:
        model_name: 模型名称
        output_dir: 输出目录
        use_ema: 是否使用EMA
        ema_decay: EMA衰减率

    Returns:
        list: 回调函数列表
    """
    # 创建模型专属目录
    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))
    model_log_dir = ensure_dir(os.path.join(LOG_DIR, model_name))

    callbacks = [
        # 1. 模型检查点 - 保存最佳模型
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # 2. 早停 - 防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        # 3. TensorBoard日志
        TensorBoard(
            log_dir=model_log_dir,
            histogram_freq=1,
            write_graph=True
        ),

        # 4. CSV日志
        CSVLogger(
            os.path.join(model_output_dir, 'training_log.csv'),
            append=False
        ),

        # 5. 时间记录
        TimeHistory(),

        # 6. 学习率日志
        LearningRateLogger()
    ]

    # 7. EMA回调（可选）
    if use_ema:
        callbacks.append(EMACallback(decay=ema_decay))
        logger.info(f"已添加EMA回调 (衰减率: {ema_decay})")

    return callbacks


# ==================== 高级训练器 ====================

class AdvancedTrainer:
    """
    高级训练器
    Advanced Trainer with all optimization techniques

    集成功能:
    - 混合精度训练
    - 学习率调度
    - 梯度累积
    - 标签平滑
    - Mixup / CutMix
    - EMA
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        use_mixed_precision: bool = USE_MIXED_PRECISION,
        lr_schedule_type: str = LR_SCHEDULE_TYPE,
        gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
        label_smoothing: float = LABEL_SMOOTHING,
        use_mixup: bool = USE_MIXUP,
        use_cutmix: bool = USE_CUTMIX,
        use_ema: bool = USE_EMA,
        ema_decay: float = EMA_DECAY
    ):
        """
        Args:
            model: Keras模型
            model_name: 模型名称
            use_mixed_precision: 是否使用混合精度
            lr_schedule_type: 学习率调度类型
            gradient_accumulation_steps: 梯度累积步数
            label_smoothing: 标签平滑系数
            use_mixup: 是否使用Mixup
            use_cutmix: 是否使用CutMix
            use_ema: 是否使用EMA
            ema_decay: EMA衰减率
        """
        self.model = model
        self.model_name = model_name
        self.use_mixed_precision = use_mixed_precision
        self.lr_schedule_type = lr_schedule_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing = label_smoothing
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 设置混合精度
        if use_mixed_precision:
            setup_mixed_precision(True)

    def compile_model(
        self,
        steps_per_epoch: int,
        epochs: int,
        initial_lr: float = LEARNING_RATE
    ) -> None:
        """
        编译模型
        Compile model with advanced configurations

        Args:
            steps_per_epoch: 每epoch步数
            epochs: 总轮数
            initial_lr: 初始学习率
        """
        # 获取学习率调度器
        lr_schedule = get_lr_schedule(
            self.lr_schedule_type,
            steps_per_epoch,
            epochs,
            initial_lr
        )

        # 创建优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # 创建带标签平滑的损失函数
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.label_smoothing
        )

        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        logger.info(f"模型编译完成:")
        logger.info(f"  - 学习率调度: {self.lr_schedule_type}")
        logger.info(f"  - 标签平滑: {self.label_smoothing}")
        logger.info(f"  - 混合精度: {self.use_mixed_precision}")

    def create_augmented_dataset(
        self,
        dataset: tf.data.Dataset,
        training: bool = True
    ) -> tf.data.Dataset:
        """
        创建带Mixup/CutMix增强的数据集
        Create dataset with Mixup/CutMix augmentation

        Args:
            dataset: 原始数据集
            training: 是否为训练模式

        Returns:
            增强后的数据集
        """
        if not training or (not self.use_mixup and not self.use_cutmix):
            return dataset

        def augment(images, labels):
            return mixup_cutmix_augmentation(
                images, labels,
                use_mixup=self.use_mixup,
                use_cutmix=self.use_cutmix
            )

        return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    def train(
        self,
        train_generator: Any,
        val_generator: Any,
        epochs: int = EPOCHS,
        class_weights: Optional[Dict[int, float]] = None,
        output_dir: str = OUTPUT_DIR
    ) -> Tuple[Any, float]:
        """
        训练模型
        Train model with all advanced techniques

        Args:
            train_generator: 训练数据生成器
            val_generator: 验证数据生成器
            epochs: 训练轮数
            class_weights: 类别权重
            output_dir: 输出目录

        Returns:
            Tuple: (history, training_time)
        """
        print("\n" + "="*60)
        print(f"开始高级训练: {self.model_name}")
        print("="*60)
        print(f"训练配置:")
        print(f"  - 混合精度: {self.use_mixed_precision}")
        print(f"  - 学习率调度: {self.lr_schedule_type}")
        print(f"  - 梯度累积: {self.gradient_accumulation_steps} 步")
        print(f"  - 标签平滑: {self.label_smoothing}")
        print(f"  - Mixup: {self.use_mixup}")
        print(f"  - CutMix: {self.use_cutmix}")
        print(f"  - EMA: {self.use_ema} (衰减率: {self.ema_decay})")
        print("="*60)

        # 计算steps
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = val_generator.samples // val_generator.batch_size
        steps_per_epoch = max(1, steps_per_epoch)
        validation_steps = max(1, validation_steps)

        # 编译模型
        self.compile_model(steps_per_epoch, epochs)

        # 获取回调函数
        callbacks = get_callbacks(
            self.model_name,
            output_dir,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay
        )

        # 开始计时
        start_time = time.time()

        # 训练
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # 计算训练时间
        training_time = time.time() - start_time

        print(f"\n训练完成! 总耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")

        return history, training_time


# ==================== 原有训练函数（保持兼容性）====================

def train_model(
    model: Model,
    train_generator: Any,
    val_generator: Any,
    model_name: str,
    epochs: int = EPOCHS,
    class_weights: Optional[Dict[int, float]] = None,
    use_advanced_training: bool = True
) -> Tuple[Any, float]:
    """
    训练模型
    Train model

    Args:
        model: Keras模型
        train_generator: 训练数据生成器
        val_generator: 验证数据生成器
        model_name: 模型名称
        epochs: 训练轮数
        class_weights: 类别权重
        use_advanced_training: 是否使用高级训练技术

    Returns:
        Tuple: (history, training_time)
    """
    if use_advanced_training:
        # 使用高级训练器
        trainer = AdvancedTrainer(
            model=model,
            model_name=model_name
        )
        return trainer.train(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=epochs,
            class_weights=class_weights
        )

    # 原有简单训练流程
    print("\n" + "="*60)
    print(f"开始训练模型: {model_name}")
    print("="*60)

    # 获取回调函数
    callbacks = get_callbacks(model_name)

    # 计算steps
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = val_generator.samples // val_generator.batch_size
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)

    # 开始计时
    start_time = time.time()

    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 计算训练时间
    training_time = time.time() - start_time

    print(f"\n训练完成! 总耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")

    return history, training_time


def fine_tune_model(
    model: Model,
    train_generator: Any,
    val_generator: Any,
    model_name: str,
    fine_tune_epochs: int = 10,
    fine_tune_lr: float = 1e-5,
    num_layers_to_unfreeze: int = 20,
    use_label_smoothing: bool = True
) -> Any:
    """
    微调模型（解冻部分基模型层）
    Fine-tune model (unfreeze some base model layers)

    Args:
        model: 预训练后的模型
        train_generator: 训练数据生成器
        val_generator: 验证数据生成器
        model_name: 模型名称
        fine_tune_epochs: 微调轮数
        fine_tune_lr: 微调学习率（应该很小）
        num_layers_to_unfreeze: 要解冻的层数
        use_label_smoothing: 是否使用标签平滑

    Returns:
        history: 微调历史
    """
    print("\n" + "="*60)
    print(f"开始微调模型: {model_name}")
    print("="*60)

    # 解冻基模型的部分层
    unfreeze_base_model(model, num_layers_to_unfreeze=num_layers_to_unfreeze)

    # 计算steps用于学习率调度
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    steps_per_epoch = max(1, steps_per_epoch)

    # 创建Warmup + Cosine学习率调度
    lr_schedule = WarmupCosineDecay(
        initial_lr=fine_tune_lr,
        warmup_steps=steps_per_epoch * 2,  # 2个epoch的warmup
        decay_steps=steps_per_epoch * (fine_tune_epochs - 2),
        warmup_lr=fine_tune_lr / 10,
        min_lr=1e-8
    )

    # 创建损失函数
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING if use_label_smoothing else 0.0
    )

    # 使用更小的学习率重新编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss,
        metrics=['accuracy']
    )

    # 微调训练
    history = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            LearningRateLogger()
        ],
        verbose=1
    )

    return history


def save_training_results(
    model_name: str,
    history: Any,
    training_time: float,
    model_info: Dict[str, Any],
    test_results: Optional[Dict[str, Any]] = None,
    output_dir: str = OUTPUT_DIR,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    保存训练结果
    Save training results to JSON

    Args:
        model_name: 模型名称
        history: 训练历史
        training_time: 训练时间
        model_info: 模型信息
        test_results: 测试结果
        output_dir: 输出目录
        training_config: 训练配置（高级训练参数）

    Returns:
        Dict: 结果字典
    """
    model_output_dir = ensure_dir(os.path.join(output_dir, model_name))

    results = {
        'model_name': model_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'model_info': model_info,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'epochs_trained': len(history.history['accuracy'])
    }

    # 添加训练配置
    if training_config:
        results['training_config'] = training_config
    else:
        results['training_config'] = {
            'mixed_precision': USE_MIXED_PRECISION,
            'lr_schedule': LR_SCHEDULE_TYPE,
            'label_smoothing': LABEL_SMOOTHING,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'use_mixup': USE_MIXUP,
            'use_cutmix': USE_CUTMIX,
            'use_ema': USE_EMA,
            'ema_decay': EMA_DECAY
        }

    if test_results:
        results['test_results'] = test_results

    # 保存为JSON
    results_path = os.path.join(model_output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"训练结果已保存到: {results_path}")

    return results


def print_training_summary(history: Any) -> None:
    """
    打印训练摘要
    Print training summary

    Args:
        history: 训练历史对象
    """
    print("\n" + "="*60)
    print("训练摘要 (Training Summary)")
    print("="*60)
    print(f"训练轮数: {len(history.history['accuracy'])}")
    print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")
    print("="*60 + "\n")


def cleanup_memory(model: Optional[Model] = None) -> None:
    """
    清理内存，释放GPU资源
    Clean up memory and release GPU resources

    Args:
        model: 要释放的模型（可选）
    """
    if model is not None:
        del model

    # 清理Keras后端
    tf.keras.backend.clear_session()

    # 强制垃圾回收
    gc.collect()

    logger.info("内存清理完成")


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """
    获取GPU内存使用信息
    Get GPU memory usage information

    Returns:
        Dict: GPU内存信息，如果没有GPU则返回None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return None

    try:
        memory_info = {}
        for i, gpu in enumerate(gpus):
            memory_info[f'GPU:{i}'] = {
                'name': gpu.name,
                'device_type': gpu.device_type
            }
        return memory_info
    except Exception as e:
        logger.warning(f"无法获取GPU内存信息: {e}")
        return None


# ==================== 导出 ====================

__all__ = [
    # 混合精度
    'setup_mixed_precision',
    'get_mixed_precision_loss_scale_optimizer',
    # 学习率调度
    'WarmupCosineDecay',
    'OneCycleLR',
    'SGDRSchedule',
    'get_lr_schedule',
    # Mixup / CutMix
    'mixup',
    'cutmix',
    'mixup_cutmix_augmentation',
    # EMA
    'EMACallback',
    # 梯度累积
    'GradientAccumulationModel',
    'wrap_model_with_gradient_accumulation',
    # 回调
    'TimeHistory',
    'LearningRateLogger',
    'get_callbacks',
    # 训练器
    'AdvancedTrainer',
    'train_model',
    'fine_tune_model',
    # 工具函数
    'save_training_results',
    'print_training_summary',
    'cleanup_memory',
    'get_gpu_memory_info',
]
