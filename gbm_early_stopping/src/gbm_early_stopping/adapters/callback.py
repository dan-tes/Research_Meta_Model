"""Универсальный адаптер для циклов обучения с колбэками "конец эпохи +
словарь метрик" (Keras-style ``logs``, PyTorch Ignite, самописный trainer)."""
from ..stopping import GBMEarlyStopping


class EpochEndEarlyStopping:
    """Оборачивает :class:`GBMEarlyStopping` в вызов ``on_epoch_end(epoch, logs)``.

    ``logs`` — словарь метрик текущей эпохи, должен содержать ключ ``monitor``
    (по умолчанию ``"val_loss"``). Возвращает ``True``, когда пора остановить
    обучение — так же, как это делают колбэки в Keras.

    Пример с PyTorch Ignite::

        es = EpochEndEarlyStopping(model_path="forecaster.pkl")

        @engine.on(Events.EPOCH_COMPLETED)
        def _check_stop(engine):
            logs = {"val_loss": engine.state.metrics["val_loss"]}
            if es.on_epoch_end(engine.state.epoch, logs):
                engine.terminate()
    """

    def __init__(self, monitor="val_loss", **kwargs):
        self.monitor = monitor
        self._es = GBMEarlyStopping(**kwargs)

    @property
    def should_stop(self):
        return self._es.should_stop

    @property
    def best_epoch(self):
        return self._es.best_epoch

    def on_epoch_end(self, epoch, logs):
        if self.monitor not in logs:
            raise KeyError(
                f"logs не содержит ключ monitor={self.monitor!r}: {list(logs)}")
        return self._es.step(epoch, logs[self.monitor])
