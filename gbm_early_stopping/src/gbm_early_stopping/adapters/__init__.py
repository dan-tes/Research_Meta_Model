"""Адаптеры :class:`gbm_early_stopping.GBMEarlyStopping` под конкретные
экосистемы. Импортируйте их напрямую из подмодулей, чтобы не тянуть
необязательные зависимости (например, ``pytorch-lightning`` или
``tensorflow``) при обычном импорте пакета:

>>> from gbm_early_stopping.adapters.lightning import GBMEarlyStoppingCallback
>>> from gbm_early_stopping.adapters.keras import GBMEarlyStoppingCallback
>>> from gbm_early_stopping.adapters.callback import EpochEndEarlyStopping
"""
