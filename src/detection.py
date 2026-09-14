"""两种轻量异常检测方法。

方法 A：3 倍标准差阈值法（基线）——只用 RMS 一个特征。
方法 B：马氏距离法——综合多个特征，衡量新样本离"正常分布中心"的距离。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table.dropna(subset=columns).copy()


class ThresholdDetector:
    """方法 A：单特征统计阈值。"""

    def __init__(self, column: str = "rms", n_std: float = 3.0) -> None:
        self.column = column
        self.n_std = n_std
        self.mean_ = float("nan")
        self.std_ = float("nan")

    def fit(self, train_table: pd.DataFrame) -> "ThresholdDetector":
        data = _finite(train_table, [self.column])[self.column]
        self.mean_ = float(data.mean())
        self.std_ = float(data.std(ddof=1))
        if self.std_ == 0:
            raise ValueError("训练数据的标准差为 0，无法使用统计阈值")
        return self

    @property
    def threshold_value(self) -> float:
        return self.mean_ + self.n_std * self.std_

    def decision_function(self, table: pd.DataFrame) -> pd.Series:
        valid = _finite(table, [self.column])
        scores = (valid[self.column].to_numpy() - self.mean_) / self.std_
        return pd.Series(scores, index=valid.index, dtype=float)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        scores = self.decision_function(table)
        prediction = pd.Series(False, index=table.index, dtype=bool)
        prediction.loc[scores.index] = scores > self.n_std
        return prediction


class MahalanobisDetector:
    """方法 B：基于马氏距离的无监督异常检测。

    思路：假设正常窗口的特征服从多元正态分布，先估计正常数据的
    均值与协方差；新窗口离分布中心太远（超过正常训练窗口距离的
    指定分位数）就判为异常。
    """

    def __init__(
        self,
        features: list[str] | None = None,
        quantile: float = 0.99,
        regularization: float = 1e-6,
    ) -> None:
        self.features = features
        self.quantile = quantile
        self.regularization = regularization
        self.mean_ = None
        self.inv_covariance_ = None
        self.distance_limit_ = float("nan")

    def fit(self, train_table: pd.DataFrame) -> "MahalanobisDetector":
        if self.features is None:
            raise ValueError("请指定要使用的特征列")
        data = _finite(train_table, self.features)[self.features].to_numpy(dtype=float)
        if data.shape[0] <= data.shape[1]:
            raise ValueError("训练窗口太少，无法估计协方差")

        self.mean_ = data.mean(axis=0)
        covariance = np.atleast_2d(np.cov(data, rowvar=False, ddof=1))
        covariance += np.eye(covariance.shape[0]) * self.regularization
        self.inv_covariance_ = np.linalg.inv(covariance)
        train_distances = self.decision_function(train_table).to_numpy()
        self.distance_limit_ = float(np.quantile(train_distances, self.quantile))
        return self

    def decision_function(self, table: pd.DataFrame) -> pd.Series:
        if self.mean_ is None or self.inv_covariance_ is None:
            raise RuntimeError("请先调用 fit()")
        valid = _finite(table, self.features)
        data = valid[self.features].to_numpy(dtype=float)
        centered = data - self.mean_
        distances = np.sqrt(
            np.einsum("ij,jk,ik->i", centered, self.inv_covariance_, centered)
        )
        return pd.Series(distances, index=valid.index, dtype=float)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        distances = self.decision_function(table)
        prediction = pd.Series(False, index=table.index, dtype=bool)
        prediction.loc[distances.index] = distances > self.distance_limit_
        return prediction



