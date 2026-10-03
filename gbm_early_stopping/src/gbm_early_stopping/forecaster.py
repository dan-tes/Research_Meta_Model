"""``GBMForecaster`` — табличный градиентный бустинг, который по префиксу кривой
``val_loss`` предсказывает:

* ``relgain``  — ещё доступное относительное улучшение ``val_loss`` (к текущему
  лучшему) за следующие ``horizon`` эпох;
* ``plateau``  — через сколько эпох ``best-so-far`` выйдет на ``plateau_eps``-
  окрестность итогового минимума остатка кривой (диагностический таргет).

Прогнозист обучается на кривых из вашего собственного пайплайна обучения —
``fit()`` принимает список последовательностей ``val_loss``. Точность прогноза
существенно зависит от того, насколько кривые обучения похожи на те, на
которых он был обучен (см. README о переносе между пайплайнами).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from ._features import DEFAULT_FEATURES, features_from_prefix


def _as_paths(path_or_paths):
    return [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)


class GBMForecaster:
    """Обучаемый прогнозист остаточного улучшения ``val_loss``.

    Параметры
    ---------
    horizon : сколько эпох вперёд оценивается ``relgain`` (должно совпадать с
        ``epoch_penalty * horizon`` — бюджетом, который использует
        :class:`gbm_early_stopping.GBMEarlyStopping`).
    prefix_min : минимальная длина префикса, с которой считается прогноз.
    plateau_eps / plateau_cap : параметры диагностического таргета ``plateau``.
    model_kind : ``"hgb"`` (sklearn ``HistGradientBoostingRegressor``, по
        умолчанию) или ``"xgb"`` (нужен пакет ``xgboost``, extra ``[xgboost]``).
    """

    def __init__(self, horizon=10, prefix_min=5, plateau_eps=0.02, plateau_cap=40,
                 features=None, model_kind="hgb"):
        self.horizon = horizon
        self.prefix_min = prefix_min
        self.plateau_eps = plateau_eps
        self.plateau_cap = plateau_cap
        self.features = list(features) if features else list(DEFAULT_FEATURES)
        self.model_kind = model_kind
        self.relgain_model_ = None
        self.plateau_model_ = None

    # ------------------------------------------------------------ модель
    def _make_model(self):
        if self.model_kind == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                 random_state=0)
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0)

    # ------------------------------------------------------------ таргеты
    def _targets_at(self, v_norm, k):
        m0 = float(v_norm[:k].min())
        fut = v_norm[k:k + self.horizon]
        fut_best = float(fut.min()) if len(fut) else m0
        relgain = max(0.0, (m0 - min(m0, fut_best)) / (m0 + 1e-8))

        tail_min = float(v_norm[k - 1:].min())
        thr = tail_min + self.plateau_eps
        running = m0
        plateau = float(self.plateau_cap)
        for step in range(1, min(self.plateau_cap, len(v_norm) - k) + 1):
            running = min(running, float(v_norm[k - 1 + step]))
            if running <= thr:
                plateau = float(step)
                break
        if m0 <= thr:
            plateau = 0.0
        return relgain, plateau

    def _curve_to_rows(self, val_loss, task="?"):
        v = np.asarray(val_loss, dtype=np.float64)
        if len(v) < self.prefix_min + 3:
            return []
        v0 = max(float(v[0]), 1e-8)
        v_norm = v / v0
        rows = []
        for k in range(self.prefix_min, len(v) - 2):
            feats = features_from_prefix(v[:k], self.prefix_min)
            if feats is None:
                continue
            relgain, plateau = self._targets_at(v_norm, k)
            feats["relgain"] = relgain
            feats["plateau"] = plateau
            feats["task"] = task
            rows.append(feats)
        return rows

    def _build_dataset(self, curves, groups=None):
        rows = []
        for i, v in enumerate(curves):
            task = groups[i] if groups is not None else "?"
            rows.extend(self._curve_to_rows(v, task))
        if not rows:
            raise ValueError(
                "не набралось окон для обучения — нужны кривые длиной хотя бы "
                f"prefix_min + 5 = {self.prefix_min + 5} эпох")
        df = pd.DataFrame(rows)
        X = df[self.features].astype(np.float64)
        return X, df["relgain"], df["plateau"], df["task"].to_numpy()

    # ------------------------------------------------------------ обучение
    def fit(self, curves, groups=None, verbose=False):
        """Обучить прогнозист на списке кривых ``val_loss``.

        ``groups`` — необязательный список идентификаторов задачи/датасета
        (по одному на кривую), нужен только для честной оценки через
        ``GroupKFold`` при ``verbose=True``, на само обучение не влияет.
        """
        X, y_rg, y_pl, task_col = self._build_dataset(curves, groups)

        if verbose:
            n_groups = len(set(task_col))
            print(f"обучение: {len(curves)} кривых -> {len(X)} окон, "
                  f"{X.shape[1]} признаков, {n_groups} групп")
            if n_groups > 1:
                for name, y in (("relgain", y_rg), ("plateau", y_pl)):
                    gkf = GroupKFold(n_splits=min(5, n_groups))
                    maes, r2s = [], []
                    for tr, va in gkf.split(X, y, task_col):
                        m = self._make_model()
                        m.fit(X.iloc[tr], y.iloc[tr])
                        pr = m.predict(X.iloc[va])
                        maes.append(mean_absolute_error(y.iloc[va], pr))
                        r2s.append(r2_score(y.iloc[va], pr))
                    print(f"  [{name:7s}] GroupKFold  MAE {np.mean(maes):.4f}  "
                          f"R2 {np.mean(r2s):+.3f}")

        self.relgain_model_ = self._make_model().fit(X, y_rg)
        self.plateau_model_ = self._make_model().fit(X, y_pl)
        return self

    @classmethod
    def fit_from_jsonl(cls, paths, val_loss_key="val_loss", task_key="task", **kwargs):
        """Обучить прогнозист из jsonl-файлов вида
        ``{"val_loss": [...], "task": "..."}`` (одна строка — одна кривая)."""
        curves, groups = [], []
        for p in _as_paths(paths):
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    curves.append(rec[val_loss_key])
                    groups.append(rec.get(task_key, "?"))
        forecaster = cls(**kwargs)
        forecaster.fit(curves, groups=groups, verbose=True)
        return forecaster

    # ------------------------------------------------------------ инференс
    def predict_rel_gain(self, prefix):
        """Ещё доступное относительное улучшение ``val_loss`` за ``horizon``
        эпох по префиксу кривой. ``None``, если префикс короче ``prefix_min``."""
        if self.relgain_model_ is None:
            raise RuntimeError("прогнозист не обучен — вызовите fit()/fit_from_jsonl() "
                                "или загрузите обученную модель через load()")
        feats = features_from_prefix(prefix, self.prefix_min)
        if feats is None:
            return None
        row = pd.DataFrame([feats])[self.features].astype(np.float64)
        return float(np.clip(self.relgain_model_.predict(row)[0], 0.0, 1.0))

    def predict_plateau(self, prefix):
        if self.plateau_model_ is None:
            raise RuntimeError("прогнозист не обучен — вызовите fit()/fit_from_jsonl() "
                                "или загрузите обученную модель через load()")
        feats = features_from_prefix(prefix, self.prefix_min)
        if feats is None:
            return None
        row = pd.DataFrame([feats])[self.features].astype(np.float64)
        return float(max(0.0, self.plateau_model_.predict(row)[0]))

    # ------------------------------------------------------------ сохранение
    def save(self, path):
        import joblib
        dirname = os.path.dirname(os.path.abspath(path))
        os.makedirs(dirname, exist_ok=True)
        joblib.dump({
            "relgain": self.relgain_model_,
            "plateau": self.plateau_model_,
            "features": self.features,
            "config": {
                "prefix_min": self.prefix_min,
                "horizon": self.horizon,
                "plateau_eps": self.plateau_eps,
                "plateau_cap": self.plateau_cap,
                "model_kind": self.model_kind,
            },
        }, path)

    @classmethod
    def load(cls, path):
        import joblib
        b = joblib.load(path)
        cfg = b.get("config", {})
        obj = cls(
            horizon=cfg.get("horizon", 10),
            prefix_min=cfg.get("prefix_min", 5),
            plateau_eps=cfg.get("plateau_eps", 0.02),
            plateau_cap=cfg.get("plateau_cap", 40),
            features=b["features"],
            model_kind=cfg.get("model_kind", "hgb"),
        )
        obj.relgain_model_ = b["relgain"]
        obj.plateau_model_ = b.get("plateau")
        return obj
