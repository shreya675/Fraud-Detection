import numpy as np
import pytest

from fraudguard.evaluate import compute_metrics, select_threshold, threshold_sweep


def _toy():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.05, 0.10, 0.20, 0.45, 0.55, 0.80, 0.90, 0.60, 0.30, 0.15])
    return y, p


def test_compute_metrics_counts_are_consistent():
    y, p = _toy()
    m = compute_metrics(y, p, 0.5)
    assert m["tp"] + m["fn"] == y.sum()
    assert m["fp"] + m["tn"] == (y == 0).sum()
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["roc_auc"] <= 1


def test_compute_metrics_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1])
    assert compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["f1"] == 1.0
    m = compute_metrics(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.5)  # predicts nothing
    assert m["tp"] == 0 and m["recall"] == 0.0 and m["precision"] == 0.0


def test_compute_metrics_validates_inputs():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1], 0.5)
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1, 0.9], 1.5)


def test_threshold_sweep_matches_compute_metrics():
    y, p = _toy()
    sweep = threshold_sweep(y, p)
    for _, row in sweep.iloc[::7].iterrows():
        m = compute_metrics(y, p, row["threshold"])
        assert row["tp"] == m["tp"] and row["fp"] == m["fp"] and row["fn"] == m["fn"]
        assert row["f1"] == pytest.approx(m["f1"])


def test_threshold_sweep_monotone_recall():
    y, p = _toy()
    sweep = threshold_sweep(y, p)
    assert (np.diff(sweep["recall"]) <= 1e-12).all()  # recall never increases with threshold


def test_select_threshold_f1_and_plateau_midpoint():
    y = np.array([0] * 50 + [1] * 5)
    p = np.r_[np.linspace(0.0, 0.30, 50), np.linspace(0.70, 1.0, 5)]
    chosen = select_threshold(y, p)
    assert 0.31 <= chosen["threshold"] <= 0.70  # inside the perfect-F1 plateau
    assert chosen["validation_f1"] == 1.0
    assert "plateau" in chosen["rule"]


def test_select_threshold_min_precision():
    y, p = _toy()
    chosen = select_threshold(y, p, min_precision=1.0)
    assert chosen["validation_precision"] == 1.0
    with pytest.raises(ValueError):
        select_threshold(y, np.zeros_like(p), min_precision=0.5)
