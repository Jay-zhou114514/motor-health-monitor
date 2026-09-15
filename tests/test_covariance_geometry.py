"""协方差几何工具的单元测试。"""

from __future__ import annotations

import numpy as np

from covariance_geometry import (
    center,
    direction_contributions,
    eigen_structure,
    empirical_covariance,
    ledoit_wolf_delta,
    mahalanobis_distances,
    select_shrinkage_by_loo,
    shrinkage_covariance,
)


def _ill_conditioned_data(seed: int = 0, n: int = 40):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    # 第二列几乎与第一列共线
    second = base + 1e-3 * rng.normal(0, 1, n)
    return center(np.column_stack([base, second]))


def test_shrinkage_reduces_condition_number():
    data = _ill_conditioned_data()
    empirical = empirical_covariance(data)
    shrunk = shrinkage_covariance(data, 0.5)
    assert np.linalg.cond(shrunk) < np.linalg.cond(empirical)
    assert np.allclose(shrunk, shrunk.T)


def test_ledoit_wolf_delta_in_range():
    data = _ill_conditioned_data()
    delta = ledoit_wolf_delta(data)
    assert 0.0 <= delta <= 1.0


def test_loo_selects_valid_delta():
    data = _ill_conditioned_data()
    delta, score = select_shrinkage_by_loo(data, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert 0.0 <= delta <= 1.0
    assert np.isfinite(score)


def test_direction_decomposition_sums_to_squared_distance():
    data = _ill_conditioned_data()
    covariance = empirical_covariance(data)
    eigenvalues, eigenvectors = eigen_structure(covariance)
    mean = np.zeros(data.shape[1])
    result = direction_contributions(data, mean, eigenvalues, eigenvectors)
    assert np.allclose(result["contributions"].sum(axis=1), result["squared_distance"])
    assert np.allclose(result["shares"].sum(axis=1), 1.0)
    distances = mahalanobis_distances(data, mean, covariance)
    assert np.allclose(result["squared_distance"], distances**2)
