import numpy as np
import pytest

from distributionshift import shift_report, spectrum_distance


def test_distance_zero_for_identical():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((32, 32))
    assert spectrum_distance(A, A) == pytest.approx(0.0, abs=1e-12)


def test_distance_positive_for_different_decay():
    rng = np.random.default_rng(1)
    u, _, vt = np.linalg.svd(rng.standard_normal((32, 32)), full_matrices=False)
    ranks = np.arange(1, 33)
    fast = (u * (1.0 / ranks ** 2)) @ vt
    slow = (u * (1.0 / ranks ** 0.5)) @ vt
    assert spectrum_distance(fast, slow) > 0.01


def test_distance_scale_invariant():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((16, 16))
    assert spectrum_distance(A, A * 1000.0) == pytest.approx(0.0, abs=1e-9)


def test_distance_rejects_bad_bins():
    A = np.eye(4)
    with pytest.raises(ValueError):
        spectrum_distance(A, A, bins=1)


def test_shift_report_flags_most_shifted():
    rng = np.random.default_rng(3)
    base = {"l1": rng.standard_normal((16, 16)), "l2": rng.standard_normal((16, 16))}
    u, _, vt = np.linalg.svd(base["l2"], full_matrices=False)
    ranks = np.arange(1, 17)
    shifted_l2 = (u * (1.0 / ranks ** 3)) @ vt
    cand = {"l1": base["l1"], "l2": shifted_l2}
    report = shift_report(base, cand)
    assert report["most_shifted"] == "l2"
    assert report["l2"] > report["l1"]


def test_shift_report_rejects_disjoint():
    A = np.eye(4)
    with pytest.raises(ValueError):
        shift_report({"a": A}, {"b": A})
