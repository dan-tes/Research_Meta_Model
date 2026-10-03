"""Коллбэк для Keras/TensorFlow. Требует ``tensorflow``
(``pip install gbm-early-stopping[keras]``)."""
try:
    from tensorflow import keras
except ImportError as exc:
    raise ImportError(
        "GBMEarlyStoppingCallback (Keras) требует tensorflow: "
        "pip install gbm-early-stopping[keras]"
    ) from exc

from ..stopping import GBMEarlyStopping


class GBMEarlyStoppingCallback(keras.callbacks.Callback):
    """Keras-коллбэк поверх :class:`GBMEarlyStopping`.

    После каждой эпохи читает ``logs[monitor]`` и выставляет
    ``self.model.stop_training = True``, когда прогнозист решает, что
    оставшееся улучшение не окупает эпохи — так же, как это делает
    ``tf.keras.callbacks.EarlyStopping``.

    Пример::

        model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            callbacks=[GBMEarlyStoppingCallback(monitor="val_loss",
                                                 model_path="forecaster.pkl")],
        )
    """

    def __init__(self, monitor="val_loss", **kwargs):
        super().__init__()
        self.monitor = monitor
        self._es = GBMEarlyStopping(**kwargs)

    @property
    def should_stop(self):
        return self._es.should_stop

    @property
    def best_epoch(self):
        return self._es.best_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor not in logs:
            return
        val = float(logs[self.monitor])
        if self._es.step(epoch, val):
            self.model.stop_training = True
