import pytest


def test_lightning_adapter_raises_clear_import_error():
    try:
        import pytorch_lightning  # noqa: F401
        import lightning  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pytorch-lightning/lightning установлены — гвард не сработает")
    with pytest.raises(ImportError, match="pytorch-lightning или lightning"):
        import gbm_early_stopping.adapters.lightning  # noqa: F401


def test_keras_adapter_raises_clear_import_error():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("tensorflow установлен — гвард не сработает")
    with pytest.raises(ImportError, match="tensorflow"):
        import gbm_early_stopping.adapters.keras  # noqa: F401
