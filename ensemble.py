"""
模型集成模块 (Model Ensemble Module)

实现多种模型集成策略，包括：
- 概率平均 (Probability Averaging)
- 加权平均 (Weighted Averaging)
- 投票法 (Voting)
- 堆叠法 (Stacking)
- Snapshot Ensemble
- 测试时增强 (Test Time Augmentation, TTA)

Author: Auto-generated
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.callbacks import Callback
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from scipy.optimize import minimize
from scipy.stats import mode
import warnings

from config import (
    NUM_CLASSES, IMG_SIZE, IMG_SHAPE, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, OUTPUT_DIR, ensure_dir
)

# 配置日志
logger = logging.getLogger(__name__)

__all__ = [
    'ModelEnsemble',
    'SnapshotEnsemble',
    'TestTimeAugmentation',
    'WeightOptimizer',
    'CyclicLearningRate',
    'create_ensemble_from_checkpoints',
    'ensemble_predict',
]


class ModelEnsemble:
    """
    模型集成类

    支持多种集成策略来组合多个模型的预测结果，提高整体性能和鲁棒性。

    Attributes:
        models: 模型列表
        strategy: 集成策略
        weights: 模型权重（用于加权平均）
        meta_model: 元模型（用于堆叠法）
    """

    SUPPORTED_STRATEGIES = ['average', 'weighted', 'voting', 'stacking']

    def __init__(
        self,
        models: List[tf.keras.Model],
        strategy: str = 'average',
        weights: Optional[List[float]] = None,
        meta_model: Optional[tf.keras.Model] = None
    ):
        """
        初始化模型集成

        Args:
            models: 模型列表，每个模型应该有相同的输入输出形状
            strategy: 集成策略
                - 'average': 概率平均，对所有模型的预测概率取平均
                - 'weighted': 加权平均，使用指定权重对预测概率加权平均
                - 'voting': 投票法，选择最多模型预测的类别
                - 'stacking': 堆叠法，使用元模型组合基模型预测
            weights: 模型权重列表，仅用于 'weighted' 策略
            meta_model: 元模型，仅用于 'stacking' 策略

        Raises:
            ValueError: 当策略不支持或参数不匹配时
        """
        if not models:
            raise ValueError("模型列表不能为空")

        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"不支持的集成策略: {strategy}. "
                f"支持的策略: {self.SUPPORTED_STRATEGIES}"
            )

        self.models = models
        self.strategy = strategy
        self.n_models = len(models)

        # 验证和设置权重
        if strategy == 'weighted':
            if weights is None:
                # 默认使用均等权重
                self.weights = np.ones(self.n_models) / self.n_models
                logger.warning("未指定权重，使用均等权重")
            else:
                if len(weights) != self.n_models:
                    raise ValueError(
                        f"权重数量 ({len(weights)}) 必须与模型数量 ({self.n_models}) 相同"
                    )
                # 归一化权重
                self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(self.n_models) / self.n_models

        # 验证元模型
        if strategy == 'stacking':
            if meta_model is None:
                logger.info("未指定元模型，将创建默认元模型")
                self.meta_model = self._create_default_meta_model()
            else:
                self.meta_model = meta_model
        else:
            self.meta_model = meta_model

        logger.info(
            f"创建模型集成: {self.n_models} 个模型, 策略={strategy}"
        )

    def _create_default_meta_model(self) -> tf.keras.Model:
        """
        创建默认的元模型用于堆叠法

        Returns:
            编译好的元模型
        """
        # 输入: 所有基模型的预测概率拼接
        input_dim = self.n_models * NUM_CLASSES

        meta_model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ], name='meta_model')

        meta_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return meta_model

    def _get_all_predictions(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> np.ndarray:
        """
        获取所有模型的预测结果

        Args:
            x: 输入数据
            verbose: 是否显示进度

        Returns:
            形状为 (n_models, n_samples, n_classes) 的预测数组
        """
        predictions = []
        for i, model in enumerate(self.models):
            if verbose:
                logger.info(f"模型 {i+1}/{self.n_models} 正在预测...")
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        return np.array(predictions)

    def predict(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> np.ndarray:
        """
        集成预测

        Args:
            x: 输入数据，可以是 numpy 数组或 tf.data.Dataset
            verbose: 是否显示详细信息

        Returns:
            预测概率，形状为 (n_samples, n_classes)
        """
        all_preds = self._get_all_predictions(x, verbose)

        if self.strategy == 'average':
            return self._average_predict(all_preds)
        elif self.strategy == 'weighted':
            return self._weighted_predict(all_preds)
        elif self.strategy == 'voting':
            return self._voting_predict(all_preds)
        elif self.strategy == 'stacking':
            return self._stacking_predict(all_preds)
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def _average_predict(self, all_preds: np.ndarray) -> np.ndarray:
        """
        概率平均预测

        Args:
            all_preds: 所有模型的预测，形状 (n_models, n_samples, n_classes)

        Returns:
            平均预测概率
        """
        return np.mean(all_preds, axis=0)

    def _weighted_predict(self, all_preds: np.ndarray) -> np.ndarray:
        """
        加权平均预测

        Args:
            all_preds: 所有模型的预测，形状 (n_models, n_samples, n_classes)

        Returns:
            加权平均预测概率
        """
        # 使用 einsum 进行加权平均: weights[i] * preds[i]
        weighted_preds = np.einsum('i,ijk->jk', self.weights, all_preds)
        return weighted_preds

    def _voting_predict(self, all_preds: np.ndarray) -> np.ndarray:
        """
        投票法预测

        Args:
            all_preds: 所有模型的预测，形状 (n_models, n_samples, n_classes)

        Returns:
            投票结果的 one-hot 编码形式
        """
        n_samples = all_preds.shape[1]
        n_classes = all_preds.shape[2]

        # 获取每个模型的预测类别
        all_classes = np.argmax(all_preds, axis=2)  # (n_models, n_samples)

        # 对每个样本进行投票
        voted_classes = np.zeros(n_samples, dtype=np.int32)
        vote_counts = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            votes = all_classes[:, i]
            # 统计每个类别的票数
            for v in votes:
                vote_counts[i, v] += 1
            # 选择票数最多的类别
            voted_classes[i] = np.argmax(vote_counts[i])

        # 归一化投票计数作为"概率"
        vote_probs = vote_counts / self.n_models
        return vote_probs

    def _stacking_predict(self, all_preds: np.ndarray) -> np.ndarray:
        """
        堆叠法预测

        Args:
            all_preds: 所有模型的预测，形状 (n_models, n_samples, n_classes)

        Returns:
            元模型的预测概率
        """
        if self.meta_model is None:
            raise ValueError("堆叠法需要元模型")

        # 将所有预测拼接为元模型的输入
        n_samples = all_preds.shape[1]
        stacked_input = all_preds.transpose(1, 0, 2).reshape(n_samples, -1)

        return self.meta_model.predict(stacked_input, verbose=0)

    def predict_classes(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> np.ndarray:
        """
        预测类别

        Args:
            x: 输入数据
            verbose: 是否显示详细信息

        Returns:
            预测类别索引
        """
        probs = self.predict(x, verbose)
        return np.argmax(probs, axis=1)

    def predict_with_uncertainty(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        带不确定性估计的预测

        通过分析各模型预测的差异来估计预测的不确定性。

        Args:
            x: 输入数据
            verbose: 是否显示详细信息

        Returns:
            tuple: (预测概率, 不确定性指标字典)
                - predictions: 集成预测概率
                - uncertainty: 包含以下不确定性指标的字典
                    - 'std': 预测标准差（模型间）
                    - 'entropy': 预测熵
                    - 'variance': 预测方差
                    - 'confidence': 最大预测概率
                    - 'disagreement': 模型间分歧度
        """
        all_preds = self._get_all_predictions(x, verbose)

        # 获取集成预测
        predictions = self.predict(x, verbose=0)

        # 计算各种不确定性指标
        uncertainty = {}

        # 1. 标准差（模型间）- 每个样本每个类别的预测标准差
        uncertainty['std'] = np.std(all_preds, axis=0)

        # 2. 预测熵 - 衡量预测分布的不确定性
        # H = -sum(p * log(p))
        eps = 1e-10  # 避免 log(0)
        uncertainty['entropy'] = -np.sum(
            predictions * np.log(predictions + eps), axis=1
        )

        # 3. 方差 - 模型间的预测方差
        uncertainty['variance'] = np.var(all_preds, axis=0)

        # 4. 置信度 - 最大预测概率
        uncertainty['confidence'] = np.max(predictions, axis=1)

        # 5. 分歧度 - 模型预测类别的一致性
        # 计算有多少模型预测了不同于多数的类别
        predicted_classes = np.argmax(all_preds, axis=2)  # (n_models, n_samples)
        n_samples = predictions.shape[0]
        disagreement = np.zeros(n_samples)

        for i in range(n_samples):
            votes = predicted_classes[:, i]
            majority_class = np.argmax(np.bincount(votes, minlength=NUM_CLASSES))
            disagreement[i] = 1.0 - np.sum(votes == majority_class) / self.n_models

        uncertainty['disagreement'] = disagreement

        # 6. 平均标准差（简化指标）
        uncertainty['mean_std'] = np.mean(uncertainty['std'], axis=1)

        return predictions, uncertainty

    def fit_stacking(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        训练堆叠法的元模型

        Args:
            x_val: 验证集输入
            y_val: 验证集标签（one-hot 编码）
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否显示训练进度

        Returns:
            训练历史
        """
        if self.strategy != 'stacking':
            raise ValueError("只有 'stacking' 策略需要训练元模型")

        logger.info("正在为元模型生成训练数据...")

        # 获取所有基模型的预测
        all_preds = self._get_all_predictions(x_val, verbose=0)
        n_samples = all_preds.shape[1]

        # 准备元模型的输入
        stacked_input = all_preds.transpose(1, 0, 2).reshape(n_samples, -1)

        logger.info(f"训练元模型，输入形状: {stacked_input.shape}")

        # 训练元模型
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.meta_model.fit(
            stacked_input, y_val,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=verbose
        )

        return history

    def set_weights(self, weights: List[float]) -> None:
        """
        设置模型权重

        Args:
            weights: 新的权重列表
        """
        if len(weights) != self.n_models:
            raise ValueError(
                f"权重数量 ({len(weights)}) 必须与模型数量 ({self.n_models}) 相同"
            )
        self.weights = np.array(weights) / np.sum(weights)
        logger.info(f"更新权重: {self.weights}")

    def get_model_contributions(
        self,
        x: Union[np.ndarray, tf.data.Dataset]
    ) -> Dict[str, np.ndarray]:
        """
        分析各模型对最终预测的贡献

        Args:
            x: 输入数据

        Returns:
            包含各模型贡献分析的字典
        """
        all_preds = self._get_all_predictions(x, verbose=0)
        ensemble_pred = self.predict(x)
        ensemble_classes = np.argmax(ensemble_pred, axis=1)

        contributions = {}

        # 各模型与集成结果的一致率
        agreement_rates = []
        for i, pred in enumerate(all_preds):
            model_classes = np.argmax(pred, axis=1)
            agreement = np.mean(model_classes == ensemble_classes)
            agreement_rates.append(agreement)

        contributions['agreement_rates'] = np.array(agreement_rates)
        contributions['predictions'] = all_preds
        contributions['weights'] = self.weights

        return contributions

    def save(self, save_dir: str) -> None:
        """
        保存集成模型

        Args:
            save_dir: 保存目录
        """
        ensure_dir(save_dir)

        # 保存各个基模型
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'model_{i}')
            model.save(model_path)

        # 保存元模型（如果有）
        if self.meta_model is not None:
            meta_path = os.path.join(save_dir, 'meta_model')
            self.meta_model.save(meta_path)

        # 保存配置
        config = {
            'strategy': self.strategy,
            'n_models': self.n_models,
            'weights': self.weights.tolist(),
        }
        config_path = os.path.join(save_dir, 'ensemble_config.npy')
        np.save(config_path, config)

        logger.info(f"集成模型已保存到: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> 'ModelEnsemble':
        """
        加载集成模型

        Args:
            save_dir: 保存目录

        Returns:
            加载的 ModelEnsemble 实例
        """
        # 加载配置
        config_path = os.path.join(save_dir, 'ensemble_config.npy')
        config = np.load(config_path, allow_pickle=True).item()

        # 加载基模型
        models_list = []
        for i in range(config['n_models']):
            model_path = os.path.join(save_dir, f'model_{i}')
            model = tf.keras.models.load_model(model_path)
            models_list.append(model)

        # 加载元模型（如果有）
        meta_model = None
        meta_path = os.path.join(save_dir, 'meta_model')
        if os.path.exists(meta_path):
            meta_model = tf.keras.models.load_model(meta_path)

        # 创建集成实例
        ensemble = cls(
            models=models_list,
            strategy=config['strategy'],
            weights=config['weights'],
            meta_model=meta_model
        )

        logger.info(f"集成模型已从 {save_dir} 加载")
        return ensemble


class WeightOptimizer:
    """
    模型权重优化器

    自动计算最优的模型权重以最大化集成性能。
    支持基于验证集的优化和贝叶斯优化。
    """

    def __init__(
        self,
        ensemble: ModelEnsemble,
        metric: str = 'accuracy'
    ):
        """
        初始化权重优化器

        Args:
            ensemble: ModelEnsemble 实例
            metric: 优化目标指标
                - 'accuracy': 准确率
                - 'log_loss': 对数损失
                - 'f1': F1 分数
        """
        self.ensemble = ensemble
        self.metric = metric
        self.optimization_history: List[Dict] = []

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        计算评估指标

        Args:
            y_true: 真实标签（one-hot 或类别索引）
            y_pred: 预测概率

        Returns:
            指标值（越大越好）
        """
        # 转换标签格式
        if len(y_true.shape) > 1:
            y_true_classes = np.argmax(y_true, axis=1)
        else:
            y_true_classes = y_true

        y_pred_classes = np.argmax(y_pred, axis=1)

        if self.metric == 'accuracy':
            return np.mean(y_true_classes == y_pred_classes)

        elif self.metric == 'log_loss':
            # 返回负对数损失（因为我们要最大化）
            eps = 1e-10
            n_samples = len(y_true_classes)
            log_loss = 0
            for i in range(n_samples):
                log_loss -= np.log(y_pred[i, y_true_classes[i]] + eps)
            return -log_loss / n_samples

        elif self.metric == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y_true_classes, y_pred_classes, average='macro')

        else:
            raise ValueError(f"不支持的指标: {self.metric}")

    def optimize_grid_search(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        n_points: int = 10,
        verbose: int = 1
    ) -> np.ndarray:
        """
        使用网格搜索优化权重

        Args:
            x_val: 验证集输入
            y_val: 验证集标签
            n_points: 每个维度的搜索点数
            verbose: 是否显示详细信息

        Returns:
            最优权重
        """
        n_models = self.ensemble.n_models

        if n_models == 2:
            # 两个模型的情况，简单网格搜索
            best_score = -np.inf
            best_weights = None

            all_preds = self.ensemble._get_all_predictions(x_val)

            for w1 in np.linspace(0, 1, n_points):
                w2 = 1 - w1
                weights = np.array([w1, w2])

                # 计算加权预测
                weighted_pred = np.einsum('i,ijk->jk', weights, all_preds)
                score = self._compute_metric(y_val, weighted_pred)

                self.optimization_history.append({
                    'weights': weights.copy(),
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()

            if verbose:
                logger.info(f"网格搜索完成: 最优分数={best_score:.4f}")

            return best_weights

        else:
            # 多个模型的情况，使用随机搜索
            return self.optimize_random_search(
                x_val, y_val, n_iterations=n_points**2, verbose=verbose
            )

    def optimize_random_search(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        n_iterations: int = 100,
        verbose: int = 1
    ) -> np.ndarray:
        """
        使用随机搜索优化权重

        Args:
            x_val: 验证集输入
            y_val: 验证集标签
            n_iterations: 迭代次数
            verbose: 是否显示详细信息

        Returns:
            最优权重
        """
        n_models = self.ensemble.n_models
        best_score = -np.inf
        best_weights = None

        all_preds = self.ensemble._get_all_predictions(x_val)

        for i in range(n_iterations):
            # 生成随机权重（使用 Dirichlet 分布确保和为 1）
            weights = np.random.dirichlet(np.ones(n_models))

            # 计算加权预测
            weighted_pred = np.einsum('i,ijk->jk', weights, all_preds)
            score = self._compute_metric(y_val, weighted_pred)

            self.optimization_history.append({
                'weights': weights.copy(),
                'score': score
            })

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

                if verbose:
                    logger.info(
                        f"迭代 {i+1}/{n_iterations}: "
                        f"新最优分数={best_score:.4f}, 权重={best_weights}"
                    )

        if verbose:
            logger.info(f"随机搜索完成: 最优分数={best_score:.4f}")

        return best_weights

    def optimize_scipy(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        method: str = 'SLSQP',
        verbose: int = 1
    ) -> np.ndarray:
        """
        使用 SciPy 优化器优化权重

        Args:
            x_val: 验证集输入
            y_val: 验证集标签
            method: 优化方法（SLSQP, L-BFGS-B 等）
            verbose: 是否显示详细信息

        Returns:
            最优权重
        """
        n_models = self.ensemble.n_models
        all_preds = self.ensemble._get_all_predictions(x_val)

        def objective(weights):
            """目标函数（返回负值因为 minimize 是最小化）"""
            weights = weights / np.sum(weights)  # 归一化
            weighted_pred = np.einsum('i,ijk->jk', weights, all_preds)
            score = self._compute_metric(y_val, weighted_pred)
            return -score

        # 约束：权重和为 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # 边界：权重在 [0, 1] 之间
        bounds = [(0, 1) for _ in range(n_models)]

        # 初始值：均等权重
        x0 = np.ones(n_models) / n_models

        if verbose:
            logger.info(f"开始 {method} 优化...")

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': verbose > 0}
        )

        optimal_weights = result.x / np.sum(result.x)

        if verbose:
            logger.info(
                f"优化完成: 最优分数={-result.fun:.4f}, "
                f"权重={optimal_weights}"
            )

        return optimal_weights

    def optimize_bayesian(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        n_iterations: int = 50,
        n_initial: int = 10,
        verbose: int = 1
    ) -> np.ndarray:
        """
        使用贝叶斯优化权重

        通过高斯过程建模来高效搜索最优权重组合。

        Args:
            x_val: 验证集输入
            y_val: 验证集标签
            n_iterations: 总迭代次数
            n_initial: 初始随机采样数
            verbose: 是否显示详细信息

        Returns:
            最优权重
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            logger.warning("sklearn 不可用，回退到随机搜索")
            return self.optimize_random_search(x_val, y_val, n_iterations, verbose)

        n_models = self.ensemble.n_models
        all_preds = self.ensemble._get_all_predictions(x_val)

        # 用于存储观测值
        X_observed = []
        y_observed = []

        def evaluate(weights):
            """评估给定权重的性能"""
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            weighted_pred = np.einsum('i,ijk->jk', weights, all_preds)
            return self._compute_metric(y_val, weighted_pred)

        # 初始随机采样
        if verbose:
            logger.info(f"贝叶斯优化: 初始采样 {n_initial} 个点...")

        for _ in range(n_initial):
            weights = np.random.dirichlet(np.ones(n_models))
            score = evaluate(weights)
            X_observed.append(weights)
            y_observed.append(score)

        # 高斯过程回归器
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        best_score = max(y_observed)
        best_weights = X_observed[np.argmax(y_observed)]

        # 贝叶斯优化循环
        for i in range(n_iterations - n_initial):
            # 拟合高斯过程
            gp.fit(np.array(X_observed), np.array(y_observed))

            # 采集函数（Expected Improvement）
            def acquisition(weights):
                weights = weights / np.sum(weights)
                mu, sigma = gp.predict([weights], return_std=True)

                # Expected Improvement
                with np.errstate(divide='warn'):
                    improvement = mu[0] - best_score
                    Z = improvement / (sigma[0] + 1e-9)
                    from scipy.stats import norm
                    ei = improvement * norm.cdf(Z) + sigma[0] * norm.pdf(Z)

                return -ei  # 负值因为 minimize

            # 搜索最大化采集函数的点
            best_next = None
            best_ei = -np.inf

            for _ in range(100):
                candidate = np.random.dirichlet(np.ones(n_models))
                ei = -acquisition(candidate)
                if ei > best_ei:
                    best_ei = ei
                    best_next = candidate

            # 评估新点
            score = evaluate(best_next)
            X_observed.append(best_next)
            y_observed.append(score)

            if score > best_score:
                best_score = score
                best_weights = best_next.copy()

                if verbose:
                    logger.info(
                        f"贝叶斯优化 {i+n_initial+1}/{n_iterations}: "
                        f"新最优={best_score:.4f}"
                    )

        if verbose:
            logger.info(f"贝叶斯优化完成: 最优分数={best_score:.4f}")

        return best_weights

    def optimize(
        self,
        x_val: Union[np.ndarray, tf.data.Dataset],
        y_val: np.ndarray,
        method: str = 'bayesian',
        **kwargs
    ) -> np.ndarray:
        """
        优化权重的统一接口

        Args:
            x_val: 验证集输入
            y_val: 验证集标签
            method: 优化方法
                - 'grid': 网格搜索
                - 'random': 随机搜索
                - 'scipy': SciPy 优化器
                - 'bayesian': 贝叶斯优化
            **kwargs: 传递给具体优化方法的参数

        Returns:
            最优权重
        """
        if method == 'grid':
            return self.optimize_grid_search(x_val, y_val, **kwargs)
        elif method == 'random':
            return self.optimize_random_search(x_val, y_val, **kwargs)
        elif method == 'scipy':
            return self.optimize_scipy(x_val, y_val, **kwargs)
        elif method == 'bayesian':
            return self.optimize_bayesian(x_val, y_val, **kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {method}")


class CyclicLearningRate(Callback):
    """
    循环学习率回调

    实现余弦退火循环学习率，用于 Snapshot Ensemble。
    学习率在每个周期内从最大值余弦衰减到最小值。
    """

    def __init__(
        self,
        base_lr: float = 0.001,
        max_lr: float = 0.1,
        step_size: int = 1000,
        mode: str = 'cosine',
        gamma: float = 0.99
    ):
        """
        初始化循环学习率

        Args:
            base_lr: 最小学习率
            max_lr: 最大学习率
            step_size: 每个周期的步数（batch 数）
            mode: 学习率变化模式
                - 'cosine': 余弦退火
                - 'triangular': 三角形
                - 'triangular2': 三角形（每周期减半）
            gamma: 每周期后最大学习率的衰减因子
        """
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        self.iteration = 0
        self.cycle = 0
        self.lr_history: List[float] = []

    def _compute_lr(self) -> float:
        """计算当前学习率"""
        cycle_position = self.iteration % self.step_size
        cycle_progress = cycle_position / self.step_size

        # 当前周期的最大学习率
        current_max_lr = self.max_lr * (self.gamma ** self.cycle)

        if self.mode == 'cosine':
            # 余弦退火
            lr = self.base_lr + (current_max_lr - self.base_lr) * \
                 (1 + np.cos(np.pi * cycle_progress)) / 2

        elif self.mode == 'triangular':
            # 三角形
            if cycle_progress < 0.5:
                lr = self.base_lr + (current_max_lr - self.base_lr) * 2 * cycle_progress
            else:
                lr = current_max_lr - (current_max_lr - self.base_lr) * 2 * (cycle_progress - 0.5)

        elif self.mode == 'triangular2':
            # 三角形（每周期减半）
            if cycle_progress < 0.5:
                lr = self.base_lr + (current_max_lr - self.base_lr) * 2 * cycle_progress
            else:
                lr = current_max_lr - (current_max_lr - self.base_lr) * 2 * (cycle_progress - 0.5)

        else:
            lr = self.max_lr

        return lr

    def on_batch_begin(self, batch, logs=None):
        """每个 batch 开始时更新学习率"""
        lr = self._compute_lr()
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.lr_history.append(lr)

        # 检查是否完成一个周期
        if self.iteration > 0 and self.iteration % self.step_size == 0:
            self.cycle += 1

        self.iteration += 1

    def on_epoch_end(self, epoch, logs=None):
        """记录当前学习率"""
        current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        if logs is not None:
            logs['lr'] = current_lr


class SnapshotCallback(Callback):
    """
    快照保存回调

    在循环学习率的每个周期最低点保存模型快照。
    """

    def __init__(
        self,
        save_dir: str,
        cyclic_lr: CyclicLearningRate,
        save_best_only: bool = True
    ):
        """
        初始化快照回调

        Args:
            save_dir: 快照保存目录
            cyclic_lr: CyclicLearningRate 实例
            save_best_only: 是否只保存验证集最优的快照
        """
        super().__init__()
        self.save_dir = save_dir
        self.cyclic_lr = cyclic_lr
        self.save_best_only = save_best_only

        self.snapshots: List[str] = []
        self.best_val_loss = np.inf
        self.snapshot_count = 0

        ensure_dir(save_dir)

    def on_batch_end(self, batch, logs=None):
        """在每个周期结束时保存快照"""
        # 检查是否在周期最低点（学习率最小）
        if (self.cyclic_lr.iteration > 0 and
            self.cyclic_lr.iteration % self.cyclic_lr.step_size == 0):

            self._save_snapshot(logs)

    def _save_snapshot(self, logs=None):
        """保存模型快照"""
        snapshot_path = os.path.join(
            self.save_dir,
            f'snapshot_{self.snapshot_count}.keras'
        )

        if self.save_best_only and logs is not None:
            val_loss = logs.get('val_loss', np.inf)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(snapshot_path)
                self.snapshots.append(snapshot_path)
                self.snapshot_count += 1
                logger.info(f"保存快照: {snapshot_path} (val_loss={val_loss:.4f})")
        else:
            self.model.save(snapshot_path)
            self.snapshots.append(snapshot_path)
            self.snapshot_count += 1
            logger.info(f"保存快照: {snapshot_path}")


class SnapshotEnsemble:
    """
    Snapshot Ensemble

    使用循环学习率训练单个模型，在每个周期的最低点保存快照，
    然后将这些快照组合成一个集成模型。

    Reference:
        Huang et al., "Snapshot Ensembles: Train 1, get M for free", ICLR 2017
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        n_snapshots: int = 5,
        snapshot_dir: str = None
    ):
        """
        初始化 Snapshot Ensemble

        Args:
            base_model: 基础模型（将被克隆和训练）
            n_snapshots: 要收集的快照数量
            snapshot_dir: 快照保存目录
        """
        self.base_model = base_model
        self.n_snapshots = n_snapshots
        self.snapshot_dir = snapshot_dir or os.path.join(OUTPUT_DIR, 'snapshots')

        self.snapshots: List[tf.keras.Model] = []
        self.ensemble: Optional[ModelEnsemble] = None

        ensure_dir(self.snapshot_dir)

    def train(
        self,
        train_data: Union[np.ndarray, tf.data.Dataset],
        val_data: Union[np.ndarray, tf.data.Dataset, Tuple] = None,
        epochs: int = None,
        batch_size: int = BATCH_SIZE,
        initial_lr: float = 0.1,
        min_lr: float = 0.001,
        verbose: int = 1,
        **fit_kwargs
    ) -> tf.keras.callbacks.History:
        """
        使用循环学习率训练并收集快照

        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 总训练轮数（默认为 n_snapshots * 一定轮数）
            batch_size: 批大小
            initial_lr: 初始（最大）学习率
            min_lr: 最小学习率
            verbose: 是否显示训练进度
            **fit_kwargs: 传递给 model.fit 的额外参数

        Returns:
            训练历史
        """
        # 计算每个快照需要的步数
        if hasattr(train_data, 'cardinality'):
            steps_per_epoch = int(train_data.cardinality().numpy())
        elif hasattr(train_data, '__len__'):
            steps_per_epoch = len(train_data) // batch_size
        else:
            steps_per_epoch = 100  # 默认值

        # 每个周期（快照）的 epoch 数
        epochs_per_cycle = 10
        if epochs is None:
            epochs = self.n_snapshots * epochs_per_cycle
        else:
            epochs_per_cycle = epochs // self.n_snapshots

        step_size = steps_per_epoch * epochs_per_cycle

        logger.info(
            f"Snapshot Ensemble 训练: {self.n_snapshots} 个快照, "
            f"每个周期 {epochs_per_cycle} epochs, "
            f"step_size={step_size}"
        )

        # 创建循环学习率回调
        cyclic_lr = CyclicLearningRate(
            base_lr=min_lr,
            max_lr=initial_lr,
            step_size=step_size,
            mode='cosine'
        )

        # 创建快照回调
        snapshot_callback = SnapshotCallback(
            save_dir=self.snapshot_dir,
            cyclic_lr=cyclic_lr,
            save_best_only=False
        )

        # 准备回调列表
        callbacks_list = [cyclic_lr, snapshot_callback]
        if 'callbacks' in fit_kwargs:
            callbacks_list.extend(fit_kwargs.pop('callbacks'))

        # 重新编译模型（使用初始学习率）
        self.base_model.compile(
            optimizer=optimizers.SGD(learning_rate=initial_lr, momentum=0.9),
            loss=self.base_model.loss,
            metrics=self.base_model.metrics
        )

        # 训练
        history = self.base_model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **fit_kwargs
        )

        # 加载所有快照
        self._load_snapshots(snapshot_callback.snapshots)

        return history

    def _load_snapshots(self, snapshot_paths: List[str]) -> None:
        """加载保存的快照"""
        self.snapshots = []

        for path in snapshot_paths[-self.n_snapshots:]:  # 只取最后 n 个
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                self.snapshots.append(model)
                logger.info(f"加载快照: {path}")

        if self.snapshots:
            # 创建集成模型
            self.ensemble = ModelEnsemble(
                models=self.snapshots,
                strategy='average'
            )
            logger.info(f"创建了 {len(self.snapshots)} 个模型的 Snapshot Ensemble")

    def predict(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> np.ndarray:
        """
        使用 Snapshot Ensemble 预测

        Args:
            x: 输入数据
            verbose: 是否显示详细信息

        Returns:
            预测概率
        """
        if self.ensemble is None:
            raise ValueError("尚未训练或加载快照")

        return self.ensemble.predict(x, verbose)

    def predict_with_uncertainty(
        self,
        x: Union[np.ndarray, tf.data.Dataset],
        verbose: int = 0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        带不确定性估计的预测

        Args:
            x: 输入数据
            verbose: 是否显示详细信息

        Returns:
            (预测概率, 不确定性指标)
        """
        if self.ensemble is None:
            raise ValueError("尚未训练或加载快照")

        return self.ensemble.predict_with_uncertainty(x, verbose)

    def load_snapshots_from_dir(self, snapshot_dir: str = None) -> None:
        """
        从目录加载快照

        Args:
            snapshot_dir: 快照目录
        """
        snapshot_dir = snapshot_dir or self.snapshot_dir

        snapshot_paths = sorted([
            os.path.join(snapshot_dir, f)
            for f in os.listdir(snapshot_dir)
            if f.startswith('snapshot_') and f.endswith('.keras')
        ])

        self._load_snapshots(snapshot_paths)


class TestTimeAugmentation:
    """
    测试时增强 (Test Time Augmentation, TTA)

    在测试时对输入图像应用多种增强变换，然后聚合所有变换后的预测结果。
    这可以提高模型的鲁棒性和准确性。
    """

    def __init__(
        self,
        model: tf.keras.Model,
        augmentations: List[str] = None,
        n_augmentations: int = 5,
        aggregation: str = 'mean'
    ):
        """
        初始化 TTA

        Args:
            model: 预测模型
            augmentations: 要应用的增强列表，支持:
                - 'horizontal_flip': 水平翻转
                - 'vertical_flip': 垂直翻转
                - 'rotate_90': 旋转90度
                - 'rotate_180': 旋转180度
                - 'rotate_270': 旋转270度
                - 'brightness': 亮度调整
                - 'contrast': 对比度调整
                - 'zoom_in': 放大
                - 'zoom_out': 缩小
            n_augmentations: 当 augmentations 为 None 时，随机选择的增强数量
            aggregation: 聚合方法
                - 'mean': 平均
                - 'max': 最大
                - 'geometric_mean': 几何平均
        """
        self.model = model
        self.aggregation = aggregation

        # 定义可用的增强
        self.available_augmentations = {
            'original': self._identity,
            'horizontal_flip': self._horizontal_flip,
            'vertical_flip': self._vertical_flip,
            'rotate_90': lambda x: self._rotate(x, 90),
            'rotate_180': lambda x: self._rotate(x, 180),
            'rotate_270': lambda x: self._rotate(x, 270),
            'brightness_up': lambda x: self._brightness(x, 1.2),
            'brightness_down': lambda x: self._brightness(x, 0.8),
            'contrast_up': lambda x: self._contrast(x, 1.2),
            'contrast_down': lambda x: self._contrast(x, 0.8),
            'zoom_in': lambda x: self._zoom(x, 1.1),
            'zoom_out': lambda x: self._zoom(x, 0.9),
        }

        if augmentations is None:
            # 默认使用常用的增强组合
            self.augmentations = [
                'original',
                'horizontal_flip',
                'rotate_90',
                'rotate_180',
                'rotate_270',
            ][:n_augmentations]
        else:
            self.augmentations = ['original'] + augmentations

        logger.info(f"TTA 增强: {self.augmentations}")

    @staticmethod
    def _identity(x: np.ndarray) -> np.ndarray:
        """恒等变换"""
        return x

    @staticmethod
    def _horizontal_flip(x: np.ndarray) -> np.ndarray:
        """水平翻转"""
        return np.flip(x, axis=-2)

    @staticmethod
    def _vertical_flip(x: np.ndarray) -> np.ndarray:
        """垂直翻转"""
        return np.flip(x, axis=-3)

    @staticmethod
    def _rotate(x: np.ndarray, angle: int) -> np.ndarray:
        """旋转图像"""
        k = angle // 90
        return np.rot90(x, k=k, axes=(-3, -2))

    @staticmethod
    def _brightness(x: np.ndarray, factor: float) -> np.ndarray:
        """调整亮度"""
        return np.clip(x * factor, 0, 1)

    @staticmethod
    def _contrast(x: np.ndarray, factor: float) -> np.ndarray:
        """调整对比度"""
        mean = np.mean(x, axis=(-3, -2, -1), keepdims=True)
        return np.clip((x - mean) * factor + mean, 0, 1)

    @staticmethod
    def _zoom(x: np.ndarray, factor: float) -> np.ndarray:
        """缩放图像"""
        if len(x.shape) == 4:
            # Batch of images
            batch_size, h, w, c = x.shape
            new_h, new_w = int(h * factor), int(w * factor)

            if factor > 1:
                # Zoom in: crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                zoomed = tf.image.resize(x, [new_h, new_w]).numpy()
                return zoomed[:, start_h:start_h+h, start_w:start_w+w, :]
            else:
                # Zoom out: pad
                zoomed = tf.image.resize(x, [new_h, new_w]).numpy()
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                padded = np.zeros_like(x)
                padded[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = zoomed
                return padded
        else:
            # Single image
            h, w, c = x.shape
            return TestTimeAugmentation._zoom(x[np.newaxis, ...], factor)[0]

    def _inverse_transform(
        self,
        predictions: np.ndarray,
        aug_name: str
    ) -> np.ndarray:
        """
        对预测结果应用逆变换（对于分类任务，预测是类别概率，不需要空间逆变换）

        对于分类任务，直接返回预测结果
        """
        return predictions

    def predict(
        self,
        x: np.ndarray,
        verbose: int = 0
    ) -> np.ndarray:
        """
        使用 TTA 进行预测

        Args:
            x: 输入图像，形状为 (batch_size, height, width, channels)
            verbose: 是否显示详细信息

        Returns:
            聚合后的预测概率
        """
        all_predictions = []

        for aug_name in self.augmentations:
            if verbose:
                logger.info(f"应用增强: {aug_name}")

            # 应用增强
            aug_func = self.available_augmentations.get(aug_name, self._identity)
            x_aug = aug_func(x.copy())

            # 预测
            pred = self.model.predict(x_aug, verbose=0)

            # 应用逆变换（对于分类任务通常不需要）
            pred = self._inverse_transform(pred, aug_name)

            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)

        # 聚合预测
        if self.aggregation == 'mean':
            return np.mean(all_predictions, axis=0)
        elif self.aggregation == 'max':
            return np.max(all_predictions, axis=0)
        elif self.aggregation == 'geometric_mean':
            return np.exp(np.mean(np.log(all_predictions + 1e-10), axis=0))
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation}")

    def predict_classes(
        self,
        x: np.ndarray,
        verbose: int = 0
    ) -> np.ndarray:
        """
        使用 TTA 预测类别

        Args:
            x: 输入图像
            verbose: 是否显示详细信息

        Returns:
            预测类别索引
        """
        probs = self.predict(x, verbose)
        return np.argmax(probs, axis=1)

    def predict_with_uncertainty(
        self,
        x: np.ndarray,
        verbose: int = 0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        带不确定性估计的 TTA 预测

        Args:
            x: 输入图像
            verbose: 是否显示详细信息

        Returns:
            (预测概率, 不确定性指标)
        """
        all_predictions = []

        for aug_name in self.augmentations:
            aug_func = self.available_augmentations.get(aug_name, self._identity)
            x_aug = aug_func(x.copy())
            pred = self.model.predict(x_aug, verbose=0)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)

        # 聚合预测
        predictions = np.mean(all_predictions, axis=0)

        # 计算不确定性
        uncertainty = {}
        uncertainty['std'] = np.std(all_predictions, axis=0)
        uncertainty['mean_std'] = np.mean(uncertainty['std'], axis=1)

        eps = 1e-10
        uncertainty['entropy'] = -np.sum(
            predictions * np.log(predictions + eps), axis=1
        )
        uncertainty['confidence'] = np.max(predictions, axis=1)

        return predictions, uncertainty


# ============== 便捷函数 ==============

def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    strategy: str = 'average',
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    从检查点文件创建集成模型

    Args:
        checkpoint_paths: 模型检查点路径列表
        strategy: 集成策略
        weights: 模型权重

    Returns:
        ModelEnsemble 实例
    """
    models_list = []

    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点不存在: {path}")

        model = tf.keras.models.load_model(path)
        models_list.append(model)
        logger.info(f"加载模型: {path}")

    return ModelEnsemble(
        models=models_list,
        strategy=strategy,
        weights=weights
    )


def ensemble_predict(
    models: List[tf.keras.Model],
    x: Union[np.ndarray, tf.data.Dataset],
    strategy: str = 'average',
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    快速集成预测（无需创建 ModelEnsemble 对象）

    Args:
        models: 模型列表
        x: 输入数据
        strategy: 集成策略
        weights: 模型权重

    Returns:
        预测概率
    """
    ensemble = ModelEnsemble(
        models=models,
        strategy=strategy,
        weights=weights
    )
    return ensemble.predict(x)


def create_tta_ensemble(
    model: tf.keras.Model,
    augmentations: List[str] = None,
    n_augmentations: int = 5
) -> TestTimeAugmentation:
    """
    创建 TTA 实例的便捷函数

    Args:
        model: 预测模型
        augmentations: 增强列表
        n_augmentations: 增强数量

    Returns:
        TestTimeAugmentation 实例
    """
    return TestTimeAugmentation(
        model=model,
        augmentations=augmentations,
        n_augmentations=n_augmentations
    )


# ============== 示例使用 ==============

if __name__ == '__main__':
    """
    示例：如何使用模型集成功能
    """
    import sys

    print("=" * 60)
    print("模型集成模块示例")
    print("=" * 60)

    # 示例 1: 创建简单集成
    print("\n1. 创建模型集成")
    print("-" * 40)

    # 创建示例模型（实际使用时应该加载训练好的模型）
    def create_dummy_model():
        model = models.Sequential([
            layers.Input(shape=IMG_SHAPE),
            layers.GlobalAveragePooling2D(),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    print("创建 3 个示例模型...")
    dummy_models = [create_dummy_model() for _ in range(3)]

    # 创建不同策略的集成
    print("\n各种集成策略:")

    # 平均集成
    avg_ensemble = ModelEnsemble(dummy_models, strategy='average')
    print(f"  - 平均集成: {avg_ensemble.n_models} 个模型")

    # 加权集成
    weighted_ensemble = ModelEnsemble(
        dummy_models,
        strategy='weighted',
        weights=[0.5, 0.3, 0.2]
    )
    print(f"  - 加权集成: 权重={weighted_ensemble.weights}")

    # 投票集成
    voting_ensemble = ModelEnsemble(dummy_models, strategy='voting')
    print(f"  - 投票集成: {voting_ensemble.n_models} 个模型")

    # 堆叠集成
    stacking_ensemble = ModelEnsemble(dummy_models, strategy='stacking')
    print(f"  - 堆叠集成: 包含元模型")

    # 示例 2: 预测
    print("\n2. 集成预测示例")
    print("-" * 40)

    # 创建随机输入
    dummy_input = np.random.rand(4, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    predictions = avg_ensemble.predict(dummy_input)
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  预测形状: {predictions.shape}")
    print(f"  预测类别: {np.argmax(predictions, axis=1)}")

    # 示例 3: 带不确定性的预测
    print("\n3. 不确定性估计")
    print("-" * 40)

    preds, uncertainty = avg_ensemble.predict_with_uncertainty(dummy_input)
    print(f"  预测熵: {uncertainty['entropy']}")
    print(f"  置信度: {uncertainty['confidence']}")
    print(f"  模型分歧度: {uncertainty['disagreement']}")

    # 示例 4: TTA
    print("\n4. 测试时增强 (TTA)")
    print("-" * 40)

    tta = TestTimeAugmentation(
        model=dummy_models[0],
        augmentations=['horizontal_flip', 'rotate_90', 'rotate_180'],
        aggregation='mean'
    )

    tta_preds = tta.predict(dummy_input)
    print(f"  TTA 预测形状: {tta_preds.shape}")
    print(f"  使用增强: {tta.augmentations}")

    # 示例 5: 权重优化
    print("\n5. 权重优化")
    print("-" * 40)

    optimizer = WeightOptimizer(weighted_ensemble, metric='accuracy')
    print("  支持的优化方法: grid, random, scipy, bayesian")

    # 示例 6: Snapshot Ensemble
    print("\n6. Snapshot Ensemble")
    print("-" * 40)

    snapshot_ensemble = SnapshotEnsemble(
        base_model=dummy_models[0],
        n_snapshots=3
    )
    print(f"  快照数量: {snapshot_ensemble.n_snapshots}")
    print(f"  保存目录: {snapshot_ensemble.snapshot_dir}")

    print("\n" + "=" * 60)
    print("示例完成！详细用法请参考各类的文档字符串。")
    print("=" * 60)
