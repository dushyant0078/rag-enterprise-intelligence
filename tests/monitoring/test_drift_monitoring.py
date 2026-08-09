from __future__ import annotations

import random

import pytest

from src.monitoring.drift_monitoring import DriftMonitor, InsufficientDataError


def test_set_baseline_rejects_too_few_values() -> None:
    mon = DriftMonitor()
    with pytest.raises(InsufficientDataError):
        mon.set_baseline([0.1, 0.2])


def test_check_without_baseline_raises() -> None:
    mon = DriftMonitor()
    with pytest.raises(InsufficientDataError):
        mon.check([0.1] * 20)


def test_check_with_too_few_current_values_raises() -> None:
    mon = DriftMonitor()
    mon.set_baseline([0.5] * 20)
    with pytest.raises(InsufficientDataError):
        mon.check([0.5, 0.5])


def test_similar_distributions_not_flagged_as_drifted() -> None:
    random.seed(42)
    baseline = [random.gauss(0.8, 0.05) for _ in range(200)]
    current = [random.gauss(0.8, 0.05) for _ in range(100)]

    mon = DriftMonitor()
    mon.set_baseline(baseline)
    report = mon.check(current)

    assert report.drifted is False
    assert report.p_value > 0.05


def test_shifted_distribution_is_flagged_as_drifted() -> None:
    random.seed(42)
    baseline = [random.gauss(0.8, 0.05) for _ in range(200)]
    current = [random.gauss(0.4, 0.05) for _ in range(100)]  # genuinely different mean

    mon = DriftMonitor()
    mon.set_baseline(baseline)
    report = mon.check(current)

    assert report.drifted is True
    assert report.p_value < 0.05


def test_has_baseline_reflects_state() -> None:
    mon = DriftMonitor()
    assert mon.has_baseline() is False
    mon.set_baseline([0.5] * 20)
    assert mon.has_baseline() is True
