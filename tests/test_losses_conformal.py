"""CPU tests for conformal calibration: coverage property on synthetic errors."""
from __future__ import annotations

import numpy as np
import pytest

from pharos.losses.conformal import calibrate, coverage


def test_in_sample_coverage_at_least_target():
    rng = np.random.default_rng(0)
    n = 20_000
    sigma = rng.uniform(0.5, 2.0, size=n)
    errors = np.abs(rng.normal(0.0, sigma))  # well-specified: err ~ N(0, sigma)
    alpha = 0.1
    q = calibrate(sigma, errors, alpha=alpha)
    assert q > 0
    # split-conformal guarantees in-sample coverage >= 1 - alpha by construction
    assert coverage(sigma, errors, q) >= 1 - alpha


def test_holdout_coverage_near_target():
    rng = np.random.default_rng(1)
    n = 40_000
    sigma = rng.uniform(0.5, 2.0, size=n)
    errors = np.abs(rng.normal(0.0, sigma))
    cal_s, test_s = sigma[: n // 2], sigma[n // 2 :]
    cal_e, test_e = errors[: n // 2], errors[n // 2 :]
    alpha = 0.1
    q = calibrate(cal_s, cal_e, alpha=alpha)
    cov = coverage(test_s, test_e, q)
    # marginal coverage ~ 1 - alpha; allow finite-sample slack
    assert cov >= 1 - alpha - 0.03
    assert cov <= 1 - alpha + 0.05


def test_larger_sigma_scale_increases_coverage():
    rng = np.random.default_rng(2)
    n = 10_000
    sigma = np.ones(n)
    errors = np.abs(rng.normal(0.0, 1.0, size=n))
    q90 = calibrate(sigma, errors, alpha=0.1)
    q99 = calibrate(sigma, errors, alpha=0.01)
    assert q99 > q90  # tighter alpha -> larger scale


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        calibrate([1.0, 2.0], [0.5], alpha=0.1)
