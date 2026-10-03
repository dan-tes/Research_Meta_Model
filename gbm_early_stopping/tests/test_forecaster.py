import numpy as np
import pytest

from gbm_early_stopping import GBMForecaster


def _synthetic_curves(n=40, length=40, seed=0):
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(0.05, 0.3)
        c = rng.uniform(0.05, 0.2)
        t = np.arange(length)
        noise = rng.normal(0, 0.01, size=length)
        curve = a * np.exp(-b * t) + c + noise
        curves.append(np.clip(curve, 1e-3, None))
    return curves


def test_fit_predict_roundtrip(tmp_path):
    curves = _synthetic_curves()
    forecaster = GBMForecaster(horizon=5, prefix_min=5).fit(curves)

    still_falling = curves[0][:10]
    rel_gain = forecaster.predict_rel_gain(still_falling)
    assert rel_gain is not None
    assert 0.0 <= rel_gain <= 1.0

    plateau = forecaster.predict_plateau(still_falling)
    assert plateau is not None
    assert plateau >= 0.0

    path = tmp_path / "forecaster.pkl"
    forecaster.save(path)
    loaded = GBMForecaster.load(path)
    assert loaded.horizon == forecaster.horizon
    assert loaded.predict_rel_gain(still_falling) == pytest.approx(rel_gain)


def test_predict_before_fit_raises():
    forecaster = GBMForecaster()
    with pytest.raises(RuntimeError):
        forecaster.predict_rel_gain([1.0] * 10)


def test_short_prefix_returns_none_after_fit():
    curves = _synthetic_curves()
    forecaster = GBMForecaster(horizon=5, prefix_min=5).fit(curves)
    assert forecaster.predict_rel_gain([1.0, 0.9]) is None
