import numpy as np

from gbm_early_stopping import features_from_prefix


def test_too_short_returns_none():
    assert features_from_prefix([1.0, 0.9], prefix_min=5) is None


def test_normalized_by_first_value():
    curve = [2.0, 1.6, 1.4, 1.3, 1.25, 1.22]
    feats = features_from_prefix(curve, prefix_min=5)
    assert feats is not None
    assert feats["p_last"] == 1.22 / 2.0
    assert feats["k"] == len(curve)


def test_decreasing_curve_has_negative_slope():
    curve = list(np.linspace(1.0, 0.5, 10))
    feats = features_from_prefix(curve, prefix_min=5)
    assert feats["slope_all"] < 0
