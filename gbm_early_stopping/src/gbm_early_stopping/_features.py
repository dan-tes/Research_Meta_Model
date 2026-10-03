"""Признаки префикса кривой ``val_loss`` для GBM-прогнозиста.

Кривая нормируется делением на ``val_loss[0]``, поэтому признаки безразмерны
и одинаково применимы к кривым с разным масштабом лосса (CE, MSE, ...).
"""
import numpy as np

DEFAULT_FEATURES = [
    "k", "p_last", "p_min", "drop_so_far", "gap_to_min", "gap_to_min_frac",
    "epochs_since_best", "frac_since_best",
    "slope_all", "curv_all", "slope_5", "slope_3",
    "impr_5", "impr_3", "impr_ratio",
    "last_delta", "mean_delta_5", "std_delta_5", "sign_changes_8", "acc",
    "slope_ratio",
]


def features_from_prefix(prefix, prefix_min=5):
    """``prefix`` — сырой ``val_loss[:k]`` (list/array). -> dict признаков или
    ``None``, если префикс короче ``prefix_min`` эпох."""
    v = np.asarray(prefix, dtype=np.float64)
    k = len(v)
    if k < prefix_min:
        return None
    v0 = max(float(v[0]), 1e-8)
    p = (v / v0).astype(np.float64)

    x = np.arange(k)
    p_min = float(p.min())
    best_idx = int(np.argmin(p))
    d = np.diff(p)

    def _slope(a):
        return float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) > 1 else 0.0

    r5, r3 = p[-5:], p[-3:]
    impr_5 = float(p[-min(6, k)] - p[-1])
    impr_3 = float(p[-min(4, k)] - p[-1])
    d8 = d[-8:]
    nz = d8[d8 != 0]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0)) if len(nz) > 1 else 0
    slope_all = _slope(p)

    return {
        "k": float(k),
        "p_last": float(p[-1]),
        "p_min": p_min,
        "drop_so_far": 1.0 - p_min,
        "gap_to_min": float(p[-1]) - p_min,
        "gap_to_min_frac": (float(p[-1]) - p_min) / (1.0 - p_min + 1e-8),
        "epochs_since_best": float(k - 1 - best_idx),
        "frac_since_best": (k - 1 - best_idx) / k,
        "slope_all": slope_all,
        "curv_all": float(np.polyfit(x, p, 2)[0]) if k > 2 else 0.0,
        "slope_5": _slope(r5),
        "slope_3": _slope(r3),
        "impr_5": impr_5,
        "impr_3": impr_3,
        "impr_ratio": impr_3 / (impr_5 + 1e-6),
        "last_delta": float(d[-1]) if k > 1 else 0.0,
        "mean_delta_5": float(np.mean(d[-5:])) if k > 1 else 0.0,
        "std_delta_5": float(np.std(d[-5:])) if k > 2 else 0.0,
        "sign_changes_8": float(sign_changes),
        "acc": float(p[-1] - 2 * p[-2] + p[-3]) if k > 2 else 0.0,
        "slope_ratio": _slope(r5) / (slope_all - 1e-9),
    }
