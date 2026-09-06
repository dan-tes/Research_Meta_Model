"""Вариант C — табличный GBM-прогнозист «сколько ещё осталось», сделанный правильно.

Предыстория. Первый табличный мета-прогнозист (`_forecast_meta` в
`pytorch_version.py`) был обучен на `data/final.csv` (почти сошедшиеся
MNIST-Keras кривые) и предсказывал одношаговое относительное изменение loss,
которое затем авторегрессионно раскатывалось на несколько эпох. На свежих,
быстро падающих кривых он систематически выдавал «изменение ≈ 0», из-за чего
«умная» остановка всегда срабатывала на полу `min_epochs + patience`.

Здесь чинятся ровно две вещи; архитектура (градиентный бустинг над таблицей
признаков префикса) — прежняя:

1. ДАННЫЕ. Кривые берутся из того же PyTorch-пайплайна, что и весь бенчмарк
   (`gen_curves.py` -> `data/curves_train.jsonl`, 500-2000 кривых), а не из
   постороннего `final.csv`. Обучающие и held-out задачи не пересекаются.

2. ТАРГЕТ. Вместо одношаговой дельты + авторегрессии модель напрямую
   предсказывает величины, нужные для решения об остановке:
     * `relgain`  — какое относительное улучшение val_loss (к текущему лучшему)
                    ещё доступно за следующие HORIZON эпох;
     * `plateau`  — через сколько эпох кривая выйдет на плато (best-so-far
                    подойдёт на EPS к итоговому минимуму остатка кривой).

    python meta_forecaster.py            # train + eval
    python meta_forecaster.py train
    python meta_forecaster.py eval
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

try:                                    # xgboost не обязателен; если есть — можно
    from xgboost import XGBRegressor    # переключить MODEL_KIND = "xgb"
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

# --- представление префикса (совпадает с curve_forecaster для сопоставимости) ---
PREFIX_MIN = 5           # меньше эпох на входе — прогноз не считаем
HORIZON = 10             # на сколько эпох вперёд оцениваем ещё доступное улучшение
PLATEAU_EPS = 0.02       # «плато»: best-so-far в пределах EPS*loss0 от минимума остатка
PLATEAU_CAP = 40         # таргет plateau обрезаем сверху этим числом эпох
STEP = 1                 # шаг по k при нарезке окон из одной кривой

TRAIN_PATH = "data/curves_train.jsonl"
EVAL_PATH = "data/curves_eval.jsonl"
MODEL_PATH = "models/meta_forecaster.pkl"
MODEL_KIND = "hgb"       # "hgb" (sklearn HistGradientBoosting) или "xgb"

FEATURES = [
    "k", "p_last", "p_min", "drop_so_far", "gap_to_min", "gap_to_min_frac",
    "epochs_since_best", "frac_since_best",
    "slope_all", "curv_all", "slope_5", "slope_3",
    "impr_5", "impr_3", "impr_ratio",
    "last_delta", "mean_delta_5", "std_delta_5", "sign_changes_8", "acc",
    "slope_ratio",
]


# ------------------------------------------------------------------ признаки
def features_from_prefix(prefix):
    """prefix — сырой val_loss[:k] (list/array). -> dict признаков или None.

    Нормировка — деление на prefix[0], та же, что у GRU-прогнозиста: кривая
    всегда стартует с 1.0, признаки безразмерны и одинаковы для CE и MSE.
    """
    v = np.asarray(prefix, dtype=np.float64)
    k = len(v)
    if k < PREFIX_MIN:
        return None
    v0 = max(float(v[0]), 1e-8)
    p = (v / v0).astype(np.float64)

    x = np.arange(k)
    p_min = float(p.min())
    best_idx = int(np.argmin(p))
    d = np.diff(p)                                   # длина k-1

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


# ------------------------------------------------------------------ таргеты
def _targets_at(v_norm, k):
    """v_norm — вся нормированная кривая, k — длина префикса.

    relgain = ещё доступное относительное улучшение к текущему лучшему за
              следующие HORIZON эпох (>= 0);
    plateau = число будущих эпох до выхода best-so-far на EPS-окрестность
              итогового минимума остатка кривой (обрезано PLATEAU_CAP).
    """
    m0 = float(v_norm[:k].min())                     # лучший val_loss на префиксе
    fut = v_norm[k:k + HORIZON]
    fut_best = float(fut.min()) if len(fut) else m0
    relgain = max(0.0, (m0 - min(m0, fut_best)) / (m0 + 1e-8))

    tail_min = float(v_norm[k - 1:].min())           # чего вообще можно достичь дальше
    thr = tail_min + PLATEAU_EPS
    running = m0
    plateau = float(PLATEAU_CAP)
    for step in range(1, min(PLATEAU_CAP, len(v_norm) - k) + 1):
        running = min(running, float(v_norm[k - 1 + step]))
        if running <= thr:
            plateau = float(step)
            break
    if m0 <= thr:
        plateau = 0.0
    return relgain, plateau


def curve_to_rows(rec):
    """Одна запись jsonl -> список dict (признаки + relgain + plateau + task)."""
    v = np.asarray(rec["val_loss"], dtype=np.float64)
    if len(v) < PREFIX_MIN + 3:
        return []
    v0 = max(float(v[0]), 1e-8)
    v_norm = v / v0
    rows = []
    for k in range(PREFIX_MIN, len(v) - 2, STEP):
        feats = features_from_prefix(v[:k])
        if feats is None:
            continue
        relgain, plateau = _targets_at(v_norm, k)
        feats["relgain"] = relgain
        feats["plateau"] = plateau
        feats["task"] = rec.get("task", "?")
        rows.append(feats)
    return rows


def build_dataset(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.extend(curve_to_rows(json.loads(line)))
    df = pd.DataFrame(rows)
    X = df[FEATURES].astype(np.float64)
    return X, df["relgain"], df["plateau"], df["task"].to_numpy()


# ------------------------------------------------------------------ модель
def _make_model():
    if MODEL_KIND == "xgb" and _HAVE_XGB:
        return XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            random_state=0)
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)


# ------------------------------------------------------------------ baseline
def trend_rel_gain(prefix, window=6, future_steps=HORIZON):
    """Оценка ещё доступного улучшения линейной экстраполяцией — ровно то, что
    делает `SmartEarlyStoppingMultiStep._forecast_trend` в дефолтном режиме."""
    y = np.asarray(prefix, dtype=np.float64)
    v0 = max(float(y[0]), 1e-8)
    p = y / v0
    w = int(min(window, len(p)))
    slope = float(np.polyfit(np.arange(w), p[-w:], 1)[0]) if w > 1 else 0.0
    steps = np.arange(1, future_steps + 1)
    preds = np.clip(p[-1] + slope * steps, 0.0, None)
    if slope >= 0:
        preds[:] = p[-1]
    m0 = float(p.min())
    projected = min(m0, float(p[-1]), float(preds.min()))
    return float(np.clip((m0 - projected) / (m0 + 1e-8), 0.0, 1.0))


# ------------------------------------------------------------------ train / eval
def train(train_path=TRAIN_PATH, out_path=MODEL_PATH):
    if not os.path.exists(train_path):
        sys.exit(f"нет {train_path} — сначала: N_PER_TASK=100 python gen_curves.py")
    import joblib

    X, y_rg, y_pl, groups = build_dataset(train_path)
    n_curves = sum(1 for _ in open(train_path))
    print(f"train: {n_curves} кривых -> {len(X)} окон, {X.shape[1]} признаков, "
          f"{len(set(groups))} задач: {sorted(set(groups))}")

    # честная оценка: GroupKFold по задаче (окна одной задачи не текут в val)
    for name, y in (("relgain", y_rg), ("plateau", y_pl)):
        gkf = GroupKFold(n_splits=min(5, len(set(groups))))
        maes, r2s = [], []
        for tr, va in gkf.split(X, y, groups):
            m = _make_model()
            m.fit(X.iloc[tr], y.iloc[tr])
            pr = m.predict(X.iloc[va])
            maes.append(mean_absolute_error(y.iloc[va], pr))
            r2s.append(r2_score(y.iloc[va], pr))
        print(f"  [{name:7s}] GroupKFold  MAE {np.mean(maes):.4f}  R2 {np.mean(r2s):+.3f}")

    rg_model = _make_model().fit(X, y_rg)
    pl_model = _make_model().fit(X, y_pl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"relgain": rg_model, "plateau": pl_model, "features": FEATURES,
                 "config": {"prefix_min": PREFIX_MIN, "horizon": HORIZON,
                            "plateau_cap": PLATEAU_CAP, "model_kind": MODEL_KIND}},
                out_path)
    print(f"-> {out_path}")


_BUNDLE = None


def load_meta(path=MODEL_PATH):
    global _BUNDLE
    if _BUNDLE is None or _BUNDLE.get("_path") != path:
        import joblib
        b = joblib.load(path)
        b["_path"] = path
        _BUNDLE = b
    return _BUNDLE


def predict_rel_gain(prefix, path=MODEL_PATH):
    """Ещё доступное относительное улучшение за HORIZON эпох по префиксу кривой."""
    feats = features_from_prefix(prefix)
    if feats is None:
        return None
    b = load_meta(path)
    row = pd.DataFrame([feats])[b["features"]].astype(np.float64)
    return float(np.clip(b["relgain"].predict(row)[0], 0.0, 1.0))


def predict_plateau(prefix, path=MODEL_PATH):
    feats = features_from_prefix(prefix)
    if feats is None:
        return None
    b = load_meta(path)
    row = pd.DataFrame([feats])[b["features"]].astype(np.float64)
    return float(max(0.0, b["plateau"].predict(row)[0]))


def evaluate(model_path=MODEL_PATH, eval_path=EVAL_PATH):
    if not os.path.exists(model_path):
        sys.exit(f"нет {model_path} — сначала: python meta_forecaster.py train")
    if not os.path.exists(eval_path):
        sys.exit(f"нет {eval_path}")
    b = load_meta(model_path)
    X, y_rg, y_pl, groups = build_dataset(eval_path)
    print(f"\nHELD-OUT eval: {eval_path}  ({len(X)} окон, задачи {sorted(set(groups))})")

    pred_rg = np.clip(b["relgain"].predict(X), 0.0, 1.0)
    pred_pl = np.clip(b["plateau"].predict(X), 0.0, PLATEAU_CAP)

    # baseline: линейный тренд по тем же префиксам (пересобираем из сырых кривых)
    trend_rg = []
    for line in open(eval_path):
        rec = json.loads(line)
        v = np.asarray(rec["val_loss"], dtype=np.float64)
        if len(v) < PREFIX_MIN + 3:
            continue
        for k in range(PREFIX_MIN, len(v) - 2, STEP):
            trend_rg.append(trend_rel_gain(v[:k]))
    trend_rg = np.asarray(trend_rg[:len(y_rg)])

    print("\n  -- relgain (ещё доступное относит. улучшение за 10 эпох) --")
    print(f"    GBM     MAE {mean_absolute_error(y_rg, pred_rg):.4f}   "
          f"R2 {r2_score(y_rg, pred_rg):+.3f}")
    print(f"    trend   MAE {mean_absolute_error(y_rg, trend_rg):.4f}   "
          f"R2 {r2_score(y_rg, trend_rg):+.3f}   (baseline)")

    print("\n  -- plateau (эпох до выхода на плато, cap 30) --")
    print(f"    GBM     MAE {mean_absolute_error(y_pl, pred_pl):.4f}   "
          f"R2 {r2_score(y_pl, pred_pl):+.3f}")

    print("\n  -- решение continue? (relgain >= epoch_penalty * future_steps) --")
    print(f"    {'penalty':>8} {'budget':>7} {'GBM acc':>9} {'trend acc':>10} "
          f"{'base rate':>10}")
    for pen in (0.006, 0.012, 0.02, 0.03):
        budget = pen * 5
        truth = (y_rg.to_numpy() >= budget)
        acc_g = np.mean((pred_rg >= budget) == truth)
        acc_t = np.mean((trend_rg >= budget) == truth)
        print(f"    {pen:>8.3f} {budget:>7.3f} {acc_g:>9.3f} {acc_t:>10.3f} "
              f"{truth.mean():>10.3f}")

    print("\n  -- relgain MAE по held-out задачам --")
    df = pd.DataFrame({"task": groups, "y": y_rg.to_numpy(),
                       "gbm": pred_rg, "trend": trend_rg})
    for t, g in df.groupby("task"):
        print(f"    {t:12s}  GBM {mean_absolute_error(g.y, g.gbm):.4f}   "
              f"trend {mean_absolute_error(g.y, g.trend):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "train", "eval"])
    a = ap.parse_args()
    if a.cmd in ("all", "train"):
        train()
    if a.cmd in ("all", "eval"):
        evaluate()
