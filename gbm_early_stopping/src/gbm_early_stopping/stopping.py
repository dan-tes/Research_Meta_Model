"""``GBMEarlyStopping`` — коллбэк ранней остановки на основе прогноза
:class:`gbm_early_stopping.GBMForecaster`."""
import os
import warnings

import numpy as np

from .forecaster import GBMForecaster

DEFAULT_MODEL_PATH = "forecaster.pkl"


class GBMEarlyStopping:
    """Останавливает обучение, когда прогнозируемое остаточное улучшение
    ``val_loss`` перестаёт окупать эпохи, которые оно бы заняло.

    На каждом шаге :class:`GBMForecaster` оценивает ``rel_gain`` — ещё
    доступное относительное улучшение ``val_loss`` (к текущему лучшему) за
    ``forecaster.horizon`` эпох вперёд. Обучение останавливается, если

        rel_gain < epoch_penalty * horizon

    подряд ``patience`` раз. ``epoch_penalty`` — минимальное относительное
    улучшение на эпоху, оправдывающее продолжение: больше значение — раньше
    стоп.

    Если прогнозист не передан (или падает на конкретном префиксе), коллбэк
    откатывается на устойчивую линейную экстраполяцию последних
    ``fallback_window`` эпох на ``fallback_steps`` эпох вперёд — это работает
    всегда, но менее точно, чем обученный прогнозист.

    Параметры
    ---------
    forecaster : обученный :class:`GBMForecaster` (или ``None``).
    model_path : путь к ``.pkl``, сохранённому через ``GBMForecaster.save()`` —
        альтернатива передаче готового ``forecaster``. Если не указан, ищется
        файл ``DEFAULT_MODEL_PATH`` (``"forecaster.pkl"`` в текущей директории);
        если его тоже нет — используется fallback на линейную экстраполяцию.
        Если ``model_path`` указан явно, а файла нет — поднимается ошибка (в
        отличие от тихого отката, когда путь не задавался вовсе).
    epoch_penalty : минимальное относительное улучшение ``val_loss`` за эпоху,
        оправдывающее продолжение обучения.
    min_epochs : жёсткий пол — раньше этой эпохи остановка не рассматривается.
    patience : сколько раз подряд решение "не окупается" нужно принять перед
        остановкой.
    fallback_window / fallback_steps : окно и горизонт линейной экстраполяции,
        используемой при отсутствии/сбое прогнозиста.
    """

    def __init__(self, forecaster=None, model_path=None, epoch_penalty=0.003,
                 min_epochs=10, patience=2, fallback_window=6, fallback_steps=5):
        if forecaster is None:
            if model_path is not None:
                forecaster = GBMForecaster.load(model_path)          # путь задан явно -> falls loudly
            elif os.path.exists(DEFAULT_MODEL_PATH):
                forecaster = GBMForecaster.load(DEFAULT_MODEL_PATH)  # неявный дефолтный путь
        if forecaster is None:
            warnings.warn(
                "GBMEarlyStopping создан без прогнозиста (forecaster/model_path) и "
                f"без файла {DEFAULT_MODEL_PATH!r} в текущей директории — будет "
                "использоваться только линейная экстраполяция как fallback. "
                "Обучите GBMForecaster на своих кривых для точного прогноза.")
        self.forecaster = forecaster
        self.epoch_penalty = epoch_penalty
        self.min_epochs = min_epochs
        self.patience = patience
        self.fallback_window = fallback_window
        self.fallback_steps = fallback_steps

        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        """Вызывается в конце каждой эпохи. Возвращает ``True``, если пора
        остановить обучение (и восстановить веса лучшей эпохи, если они у вас
        сохраняются по ``best_epoch``)."""
        val_loss = float(val_loss)
        self.val_loss.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if len(self.val_loss) < self.min_epochs:
            return False

        rel_gain, horizon = self._estimate_rel_gain()
        budget = self.epoch_penalty * horizon

        if rel_gain < budget:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False

    def reset(self):
        """Сбросить состояние коллбэка для повторного использования (например,
        между запусками кросс-валидации)."""
        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    # ------------------------------------------------------------------
    def _estimate_rel_gain(self):
        if self.forecaster is not None:
            try:
                rg = self.forecaster.predict_rel_gain(self.val_loss)
                if rg is not None:
                    return rg, self.forecaster.horizon
            except Exception as exc:
                warnings.warn(f"GBMForecaster упал на текущем префиксе ({exc}), "
                               "откат на линейную экстраполяцию")
        return self._trend_rel_gain(), self.fallback_steps

    def _trend_rel_gain(self):
        y = np.asarray(self.val_loss, dtype=float)
        w = int(min(self.fallback_window, len(y)))
        yr = y[-w:]
        slope = np.polyfit(np.arange(w), yr, 1)[0] if w > 1 else 0.0

        steps = np.arange(1, self.fallback_steps + 1)
        preds = np.clip(yr[-1] + slope * steps, 0.0, None)
        if slope >= 0:
            preds[:] = yr[-1]

        projected = min(yr[-1], float(preds.min()))
        return (self.best_val_loss - projected) / max(self.best_val_loss, 1e-8)
