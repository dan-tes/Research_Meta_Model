"""Стратегии ранней остановки обучения.

Три коллбэка, которые вызываются в конце каждой эпохи и решают, пора ли
останавливаться:

* ``SimpleEarlyStopping`` — классический ``patience`` (N эпох без улучшения);
* ``SmartEarlyStoppingMultiStep`` — оценивает ещё доступное относительное
  улучшение ``val_loss`` за горизонт и стопает, когда оно перестаёт окупать
  эпохи. Прогноз — линейная экстраполяция недавнего окна либо (``use_meta=True``)
  табличный GBM из ``meta_forecaster.py`` (вариант C);
* ``ParametricEarlyStopping`` — то же решение по фиту экспоненты ``a·e^(−bt)+c``.

Компромисс «скорость ↔ качество» задаёт ``EPOCH_PENALTY`` — минимальное
относительное улучшение ``val_loss`` (к текущему лучшему), оправдывающее одну
лишнюю эпоху. Бенчмарк стратегий — ``eval_early_stopping.py``.
"""
import numpy as np
import pandas as pd

# ========================
# ГЛОБАЛЬНЫЙ КОМПРОМИСС СКОРОСТЬ / КАЧЕСТВО
# ========================
# Минимальное относительное улучшение val_loss (к текущему лучшему), которое
# должна приносить одна дополнительная эпоха, чтобы обучение продолжалось.
# Большое значение = сильный штраф за лишние эпохи = ранняя остановка.
EPOCH_PENALTY = 0.006


# ========================
# EARLY STOPPING CALLBACKS
# ========================

class SmartEarlyStoppingMultiStep:
    """Early Stopping, который прогнозирует ближайший кусок кривой val_loss и
    останавливает обучение, как только ожидаемое улучшение перестаёт окупать
    эпохи, которые оно стоит.

    Компромисс задаётся параметром ``epoch_penalty`` — это минимальное
    относительное (к текущему лучшему val_loss) улучшение, которое должна
    приносить одна дополнительная эпоха, чтобы обучение продолжалось.
    Чем больше ``epoch_penalty`` — тем раньше происходит остановка.
    """

    def __init__(self, model_meta=None, train_size=None, future_steps=5,
                 min_epochs=10, patience=2, epoch_penalty=EPOCH_PENALTY,
                 forecast_window=6, restore_best_weights=True, use_meta=False,
                 meta_path="models/meta_forecaster.pkl"):
        self.model_meta = model_meta
        self.train_size = train_size
        self.future_steps = future_steps
        self.min_epochs = min_epochs
        self.patience = patience
        self.epoch_penalty = epoch_penalty
        self.forecast_window = forecast_window
        self.restore_best_weights = restore_best_weights
        # Вариант C: табличный GBM-прогнозист из meta_forecaster.py, обученный на
        # кривых ИЗ ЭТОГО ЖЕ пайплайна (gen_curves.py) и предсказывающий НАПРЯМУЮ
        # ещё доступное относительное улучшение за горизонт (а не одношаговую
        # дельту с авторегрессией, как сломанная модель на final.csv). Включается
        # явно через use_meta=True; при отсутствии .pkl тихо откатываемся на тренд.
        self.meta = None
        self.meta_horizon = None
        if use_meta:
            try:
                import meta_forecaster as MF
                self.meta = MF.load_meta(meta_path)
                self.meta_horizon = self.meta["config"].get("horizon", 10)
            except Exception:
                self.meta = None
        self.use_meta = self.meta is not None

        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        """Вызывается в конце каждой эпохи. Возвращает True, если пора стоп."""
        val_loss = float(val_loss)
        self.val_loss.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if len(self.val_loss) < self.min_epochs:
            return False

        if self.use_meta:
            rel_gain = self._forecast_meta_gain()                # прямой прогноз GBM
            budget = self.epoch_penalty * self.meta_horizon      # горизонт GBM (10)
        else:
            preds = self._forecast()
            projected = min([self.val_loss[-1], *preds])
            denom = max(self.best_val_loss, 1e-8)
            rel_gain = (self.best_val_loss - projected) / denom  # ещё доступное улучшение
            budget = self.epoch_penalty * self.future_steps      # сколько мы готовы "заплатить"

        if rel_gain < budget:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    # ------------------------------------------------------------------
    # Прогноз будущих значений
    # ------------------------------------------------------------------
    def _forecast(self):
        return self._forecast_trend()

    def _forecast_trend(self):
        """Устойчивая линейная экстраполяция по короткому недавнему окну.

        Линейный тренд на последних эпохах показывает, продолжает ли кривая
        снижаться, но без "разбегания" квадратичной экстраполяции (которая на
        падающей кривой уводит прогноз в минус и мешает остановке).
        """
        y = np.asarray(self.val_loss, dtype=float)
        w = int(min(self.forecast_window, len(y)))
        yr = y[-w:]
        slope = np.polyfit(np.arange(w), yr, 1)[0] if w > 1 else 0.0

        steps = np.arange(1, self.future_steps + 1)
        preds = yr[-1] + slope * steps
        preds = np.clip(preds, 0.0, None)
        if slope >= 0:                       # тренд вверх/плато -> дальше улучшений нет
            preds[:] = yr[-1]
        return preds.tolist()

    def _forecast_meta_gain(self):
        """Вариант C: табличный GBM напрямую оценивает ещё доступное
        относительное улучшение val_loss за meta_horizon эпох по признакам
        текущего префикса. При сбое — откат на линейный тренд."""
        try:
            import meta_forecaster as MF
            feats = MF.features_from_prefix(np.asarray(self.val_loss, dtype=np.float64))
            row = pd.DataFrame([feats])[self.meta["features"]].astype(np.float64)
            return float(np.clip(self.meta["relgain"].predict(row)[0], 0.0, 1.0))
        except Exception:
            preds = self._forecast_trend()
            projected = min([self.val_loss[-1], *preds])
            return (self.best_val_loss - projected) / max(self.best_val_loss, 1e-8)


class SimpleEarlyStopping:
    """Стандартный Early Stopping"""

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


class ParametricEarlyStopping:
    """Параметрический Early Stopping: подгоняет экспоненту a*exp(-b*t)+c к
    кривой val_loss и останавливается, когда прогноз обещает меньше улучшения,
    чем ``epoch_penalty`` (относительно лучшего val_loss) на эпоху."""

    def __init__(self, min_epochs=10, patience=2, epoch_penalty=EPOCH_PENALTY, future_steps=5):
        self.min_epochs = min_epochs
        self.patience = patience
        self.epoch_penalty = epoch_penalty
        self.future_steps = future_steps
        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        val_loss = float(val_loss)
        self.val_loss.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if len(self.val_loss) < self.min_epochs:
            return False

        try:
            from scipy.optimize import curve_fit

            y = np.asarray(self.val_loss)
            x = np.arange(len(y))

            def exp_func(t, a, b, c):
                return a * np.exp(-b * t) + c

            popt, _ = curve_fit(exp_func, x, y, bounds=(0, [10, 10, 10]), maxfev=5000)

            future_epochs = np.arange(len(y), len(y) + self.future_steps)
            preds = exp_func(future_epochs, *popt)

            projected = float(min(preds.min(), y[-1]))
            rel_gain = (self.best_val_loss - projected) / max(self.best_val_loss, 1e-8)

            if rel_gain < self.epoch_penalty * self.future_steps:
                self.counter += 1
            else:
                self.counter = 0

            if self.counter >= self.patience:
                self.should_stop = True
                return True
        except Exception:
            pass

        return False
