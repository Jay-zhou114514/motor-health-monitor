"""单特征马氏距离的单元测试。

单特征时协方差退化成一个标量，代码必须能正确处理，
否则 Feature Ablation（逐特征测试）无法进行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from detection import MahalanobisDetector


def test_single_feature_fit_and_predict():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"rms": rng.normal(0.5, 0.05, 40)})
    detector = MahalanobisDetector(features=["rms"], quantile=0.99).fit(train)

    test = pd.DataFrame({"rms": [0.5, 0.5, 5.0]})
    prediction = detector.predict(test)
    assert prediction.shape == (3,)
    assert not bool(prediction.iloc[0])
    assert bool(prediction.iloc[2])
    # 单特征时，马氏距离应等于标准化后的绝对偏移量
    distances = detector.decision_function(test).to_numpy()
    expected = np.abs(test["rms"].to_numpy() - detector.mean_[0]) / np.sqrt(
        (detector.inv_covariance_[0, 0]) ** -1
    )
    assert np.allclose(distances, expected)
