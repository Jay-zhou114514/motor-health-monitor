"""包络谱故障诊断的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from diagnosis import (
    EnvelopeDiagnoser,
    envelope_spectrum,
    fault_feature_score,
)


def _impulse_train(freq_hz: float, sample_rate: float, seconds: float = 2.0):
    """模拟轴承故障：周期性冲击衰减振荡（共振载波 + 指数衰减）。"""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    resonance = 3000.0
    impulses = np.zeros_like(t)
    period = 1.0 / freq_hz
    decay = 40.0
    for start in np.arange(0, seconds, period):
        index = int(start * sample_rate)
        length = min(int(sample_rate * 0.01), t.size - index)
        if length <= 0:
            break
        local_t = np.arange(length) / sample_rate
        impulses[index : index + length] += np.exp(-decay * local_t) * np.sin(
            2 * np.pi * resonance * local_t
        )
    return t, impulses + 0.05 * np.random.default_rng(0).normal(0, 1, t.size)


def test_envelope_spectrum_shapes():
    signal = np.random.default_rng(1).normal(0, 1, 4096)
    frequencies, spectrum = envelope_spectrum(signal, 1024.0)
    assert frequencies[0] == 0.0
    assert np.isclose(frequencies[-1], 512.0)
    assert spectrum.shape == frequencies.shape
    assert (spectrum >= 0).all()


def test_fault_feature_score_detects_impulse_rate():
    sample_rate = 20000.0
    _, signal = _impulse_train(81.0, sample_rate)
    frequencies, spectrum = envelope_spectrum(signal, sample_rate)
    score_fault = fault_feature_score(frequencies, spectrum, 81.0)
    score_other = fault_feature_score(frequencies, spectrum, 119.0)
    assert score_fault > 5.0
    assert score_fault > 3 * score_other


def test_diagnoser_classifies_outer_and_inner():
    sample_rate = 20000.0
    char_freqs = {"BPFO": 81.0, "BPFI": 119.0, "FTF": 15.0, "BSF": 64.0}
    _, normal = _impulse_train(50.0, sample_rate)  # 50 Hz 与所有特征频率都不同
    diagnoser = EnvelopeDiagnoser().fit(
        [{"signal": normal, "sr": sample_rate, **char_freqs}]
    )
    assert diagnoser.baseline_score_ > 0

    _, outer = _impulse_train(81.0, sample_rate)
    _, inner = _impulse_train(119.0, sample_rate)
    assert diagnoser.diagnose(outer, sample_rate, char_freqs)["predicted"] == "outer_race_fault"
    assert diagnoser.diagnose(inner, sample_rate, char_freqs)["predicted"] == "inner_race_fault"
    assert diagnoser.diagnose(normal, sample_rate, char_freqs)["predicted"] == "normal"


def test_diagnoser_requires_normal_records():
    with pytest.raises(ValueError):
        EnvelopeDiagnoser().fit([])
