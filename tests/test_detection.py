"""两种检测器的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detection import MahalanobisDetector, ThresholdDetector


def _normal_table(seed: int = 0, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"rms": rng.normal(0.5, 0.05, n)})


def test_threshold_detector_flags_outlier():
    train = _normal_table()
    detector = ThresholdDetector(column="rms", n_std=3.0).fit(train)
    test = pd.concat(
        [
            _normal_table(seed=1),
            pd.DataFrame({"rms": [detector.mean_ + 10 * detector.std_]}),
        ],
        ignore_index=True,
    )
    pred = detector.predict(test)
    assert bool(pred.iloc[-1])
    assert not bool(pred.iloc[0])


def test_threshold_detector_zero_std_raises():
    train = pd.DataFrame({"rms": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError):
        ThresholdDetector(column="rms").fit(train)


def _multivariate_table(seed: int = 0, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n, 4))
    return pd.DataFrame(base, columns=["rms", "crest_factor", "kurtosis", "centroid_hz"])


def test_mahalanobis_distance_and_predict():
    features = ["rms", "crest_factor", "kurtosis", "centroid_hz"]
    train = _multivariate_table()
    detector = MahalanobisDetector(features=features, quantile=0.99).fit(train)
    # 训练数据本身几乎都在报警线内（99% 分位下 100 个点至多越线 1 个）
    assert int(detector.predict(train).sum()) <= 1
    # 极端偏移的窗口应被判定为异常
    outlier = train.copy()
    outlier.iloc[0] += 50.0
    assert bool(detector.predict(outlier).iloc[0])


def test_mahalanobis_requires_features():
    detector = MahalanobisDetector()
    with pytest.raises(ValueError):
        detector.fit(_multivariate_table())


def test_mahalanobis_too_few_rows():
    features = ["rms", "crest_factor", "kurtosis", "centroid_hz"]
    with pytest.raises(ValueError):
        MahalanobisDetector(features=features).fit(_multivariate_table(n=3))
