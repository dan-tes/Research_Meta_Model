"""Коллбэк для PyTorch Lightning. Требует ``pytorch-lightning`` или пакет
``lightning`` (``pip install gbm-early-stopping[lightning]``)."""
try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning.pytorch as pl
    except ImportError as exc:
        raise ImportError(
            "GBMEarlyStoppingCallback требует pytorch-lightning или lightning: "
            "pip install gbm-early-stopping[lightning]"
        ) from exc

from ..stopping import GBMEarlyStopping


class GBMEarlyStoppingCallback(pl.Callback):
    """Lightning-коллбэк поверх :class:`GBMEarlyStopping`.

    После каждой валидационной эпохи читает ``trainer.callback_metrics[monitor]``
    и выставляет ``trainer.should_stop = True``, когда прогнозист решает, что
    оставшееся улучшение не окупает эпохи.

    Пример::

        trainer = pl.Trainer(callbacks=[
            GBMEarlyStoppingCallback(monitor="val_loss", model_path="forecaster.pkl"),
        ])
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

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        val = float(metrics[self.monitor])
        if self._es.step(trainer.current_epoch, val):
            trainer.should_stop = True
