"""协方差几何与正则化工具（EXP-V1-05 用）。

提供：
- 经验协方差（MLE）
- 向 scaled-identity 目标收缩的协方差，收缩强度可由
  Ledoit–Wolf 解析式或训练集留一交叉验证确定
- 特征值/特征向量结构、最小特征值方向的投影与距离贡献分解

注意：所有协方差估计与超参数选择都只使用训练数据；
测试数据只用于最终评价。
"""

from __future__ import annotations

import numpy as np


def center(matrix: np.ndarray) -> np.ndarray:
    """按列去均值。"""
    return np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(axis=0)


def empirical_covariance(centered: np.ndarray) -> np.ndarray:
    """经验协方差（最大似然，除以 n）。输入应已去均值。"""
    n = centered.shape[0]
    return np.atleast_2d(centered.T @ centered / n)


def shrinkage_covariance(centered: np.ndarray, delta: float) -> np.ndarray:
    """向 scaled-identity 目标收缩。

    delta = 0 → 经验协方差；delta = 1 → 纯目标（trace/p · I）。
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta 必须位于 [0, 1]")
    mle = empirical_covariance(centered)
    dimension = mle.shape[0]
    target = (np.trace(mle) / dimension) * np.eye(dimension)
    return (1.0 - delta) * mle + delta * target


def ledoit_wolf_delta(centered: np.ndarray) -> float:
    """Ledoit–Wolf (2004) 对 scaled-identity 目标的解析收缩强度。"""
    n = centered.shape[0]
    mle = empirical_covariance(centered)
    dimension = mle.shape[0]
    target = (np.trace(mle) / dimension) * np.eye(dimension)
    d_squared = float(np.sum((mle - target) ** 2))
    if d_squared <= 0:
        return 0.0
    b_squared = 0.0
    for row in centered:
        outer = np.outer(row, row)
        b_squared += float(np.sum((outer - mle) ** 2))
    b_squared /= n**2
    return float(np.clip(b_squared / d_squared, 0.0, 1.0))


def log_likelihood(sample: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """单个样本在多元正态下的对数似然（含常数项）。"""
    dimension = covariance.shape[0]
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("-inf")
    inverse = np.linalg.inv(covariance)
    centered = np.asarray(sample, dtype=float) - mean
    quadratic = float(centered @ inverse @ centered)
    return float(-0.5 * (dimension * np.log(2 * np.pi) + log_det + quadratic))


def select_shrinkage_by_loo(
    centered: np.ndarray, deltas: np.ndarray, jitter: float = 1e-12
) -> tuple[float, float]:
    """用训练集留一交叉验证选择收缩强度，返回 (delta, 平均对数似然)。"""
    n = centered.shape[0]
    best_delta, best_score = 0.0, float("-inf")
    for delta in np.asarray(deltas, dtype=float):
        scores = []
        for index in range(n):
            subset = np.delete(centered, index, axis=0)
            mean = subset.mean(axis=0)
            covariance = shrinkage_covariance(center(subset), float(delta))
            covariance = covariance + np.eye(covariance.shape[0]) * jitter
            scores.append(log_likelihood(centered[index], mean, covariance))
        score = float(np.mean(scores))
        if score > best_score:
            best_delta, best_score = float(delta), score
    return best_delta, best_score


def mahalanobis_distances(
    samples: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """相对给定均值/协方差的马氏距离。"""
    centered = np.asarray(samples, dtype=float) - mean
    inverse = np.linalg.inv(covariance)
    squared = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    return np.sqrt(np.clip(squared, 0.0, None))


def eigen_structure(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对称矩阵特征分解，特征值升序返回。"""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def canonicalize_direction(direction: np.ndarray) -> np.ndarray:
    """把特征向量符号规范化（绝对值最大的分量为正），便于跨组比较。"""
    direction = np.asarray(direction, dtype=float)
    index = int(np.argmax(np.abs(direction)))
    return direction if direction[index] >= 0 else -direction


def direction_contributions(
    samples: np.ndarray,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """把马氏距离平方分解到各特征方向上。

    返回：
    - squared_distance：每个样本的 D²
    - contributions：形状 (n_samples, n_directions)，满足按行求和 = D²
    - shares：各方向占比
    - projections：在原始方向上的投影（未标准化）
    - standardized_projections：投影 / √λ（即该方向的距离贡献）
    """
    centered = np.asarray(samples, dtype=float) - mean
    projections = centered @ eigenvectors
    contributions = projections**2 / eigenvalues
    squared_distance = contributions.sum(axis=1)
    shares = np.divide(
        contributions,
        squared_distance[:, None],
        out=np.zeros_like(contributions),
        where=squared_distance[:, None] > 0,
    )
    standardized = projections / np.sqrt(eigenvalues)[None, :]
    return {
        "squared_distance": squared_distance,
        "contributions": contributions,
        "shares": shares,
        "projections": projections,
        "standardized_projections": standardized,
    }
