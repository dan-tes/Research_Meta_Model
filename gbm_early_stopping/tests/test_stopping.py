import warnings

import numpy as np
import pytest

from gbm_early_stopping import GBMEarlyStopping, GBMForecaster


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


def test_stops_on_flat_curve_with_fallback():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = GBMEarlyStopping(min_epochs=8, patience=1, epoch_penalty=0.01)

    # быстрый спад, затем явное плато — окно fallback_window=6 должно попасть
    # целиком на плоский хвост и дать rel_gain около нуля
    curve = [1.0, 0.8, 0.65, 0.55, 0.5, 0.499, 0.4985, 0.498, 0.4978, 0.4977, 0.4976]
    stopped_at = None
    for epoch, val_loss in enumerate(curve):
        if es.step(epoch, val_loss):
            stopped_at = epoch
            break

    assert stopped_at is not None
    assert es.should_stop


def test_never_stops_before_min_epochs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = GBMEarlyStopping(min_epochs=5, patience=1, epoch_penalty=0.5)

    for epoch, val_loss in enumerate([1.0, 0.5, 0.49, 0.489]):
        assert es.step(epoch, val_loss) is False


def test_uses_trained_forecaster():
    curves = _synthetic_curves()
    forecaster = GBMForecaster(horizon=5, prefix_min=5).fit(curves)
    es = GBMEarlyStopping(forecaster=forecaster, min_epochs=5, patience=1,
                           epoch_penalty=0.02)

    # прогоняем полную кривую из того же распределения, что и обучающие —
    # на длинном, уже сошедшемся хвосте прогнозист должен решить остановиться
    stopped = False
    for epoch, val_loss in enumerate(curves[0]):
        if es.step(epoch, val_loss):
            stopped = True
            break
    assert stopped


def test_no_forecaster_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # без forecaster.pkl рядом
    with pytest.warns(UserWarning):
        GBMEarlyStopping()


def test_default_model_path_autoloads(tmp_path, monkeypatch):
    curves = _synthetic_curves()
    GBMForecaster(horizon=5, prefix_min=5).fit(curves).save(tmp_path / "forecaster.pkl")
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # без явного model_path — не должно быть warning
        es = GBMEarlyStopping(min_epochs=5, patience=1)
    assert es.forecaster is not None


def test_missing_explicit_model_path_raises(tmp_path):
    with pytest.raises(Exception):
        GBMEarlyStopping(model_path=str(tmp_path / "missing.pkl"))
