"""Важность всех признаков мета-прогнозиста (relgain и plateau).

HistGradientBoostingRegressor не даёт встроенной feature_importances_, поэтому
считаем permutation importance на held-out кривых (EVAL_PATH): насколько растёт
MAE, если перемешать один признак.

    python plot_feature_importance.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

from meta_forecaster import EVAL_PATH, FEATURES, MODEL_PATH, build_dataset, load_meta

OUT = "results/fig_feature_importance.png"


def main():
    b = load_meta(MODEL_PATH)
    X, y_rg, y_pl, _ = build_dataset(EVAL_PATH)
    print(f"held-out: {len(X)} окон")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.35 * len(FEATURES))))
    for ax, (name, y) in zip(axes, (("relgain", y_rg), ("plateau", y_pl))):
        r = permutation_importance(b[name], X, y, scoring="neg_mean_absolute_error",
                                   n_repeats=5, random_state=0, n_jobs=-1)
        order = np.argsort(r.importances_mean)
        ax.barh(np.array(FEATURES)[order], r.importances_mean[order],
                xerr=r.importances_std[order], color="#4C72B0")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}: permutation importance (held-out)")
        ax.set_xlabel("рост MAE при перемешивании признака")
        for f, m in zip(np.array(FEATURES)[order][::-1], r.importances_mean[order][::-1]):
            print(f"  [{name}] {f:18s} {m:+.5f}")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"-> {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
