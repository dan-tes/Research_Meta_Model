# gbm-early-stopping

Ранняя остановка обучения, которая прогнозирует, **сколько улучшения `val_loss`
ещё осталось**, вместо того чтобы просто ждать `patience` эпох без улучшения.

Ниже — быстрый старт. Подробное руководство со всеми параметрами, адаптерами,
разбором ошибок и справочником API — [docs/user_guide.md](docs/user_guide.md).

Табличный градиентный бустинг (`GBMForecaster`) по короткому префиксу кривой
`val_loss` оценивает `rel_gain` — ещё доступное относительное улучшение за
`horizon` эпох вперёд — и обучение останавливается, когда

```
rel_gain < epoch_penalty * horizon
```

`epoch_penalty` — минимальное относительное улучшение `val_loss` за эпоху,
оправдывающее продолжение. Библиотека не зависит от архитектуры сети или
фреймворка — на вход коллбэку нужны только числа `val_loss` по эпохам;
адаптеры под конкретные экосистемы — в `gbm_early_stopping.adapters`.

> Прогнозист даёт заметно более точную оценку `rel_gain`, чем линейная
> экстраполяция кривой, но это **не значит**, что он экономит больше эпох при
> той же просадке качества — калибруйте `epoch_penalty` под свой пайплайн и
> сравнивайте по факту (эпохи ↔ качество), а не по точности прогноза самой
> по себе.

## Установка

Пакет пока не публикуется в PyPI — ставится локально:

```bash
pip install -e ./gbm_early_stopping
# с адаптером под PyTorch Lightning:
pip install -e "./gbm_early_stopping[lightning]"
# с адаптером под Keras/TensorFlow:
pip install -e "./gbm_early_stopping[keras]"
```

## Быстрый старт

### 1. Обучите прогнозист на своих кривых

`GBMForecaster` учится на *ваших* кривых `val_loss` — точность прогноза
существенно зависит от того, насколько кривые обучения похожи на те, на
которых он обучен (архитектура, оптимизатор, датасет и т.д.).

```python
from gbm_early_stopping import GBMForecaster

# curves — список последовательностей val_loss, накопленных за прошлые прогоны
# обучения (чем больше и разнообразнее, тем лучше)
forecaster = GBMForecaster(horizon=10, prefix_min=5).fit(curves)
forecaster.save("forecaster.pkl")  # дефолтный путь, который ищет GBMEarlyStopping
```

Либо из jsonl-файлов вида `{"val_loss": [...], "task": "..."}` (одна строка —
одна кривая; `task` нужен только для честной кросс-валидации при обучении):

```python
forecaster = GBMForecaster.fit_from_jsonl("curves_train.jsonl", horizon=10)
```

### 2. Используйте прогнозист в цикле обучения

```python
from gbm_early_stopping import GBMEarlyStopping

early_stop = GBMEarlyStopping(epoch_penalty=0.003, min_epochs=10, patience=2)

for epoch in range(max_epochs):
    train_one_epoch(model, ...)
    val_loss = validate(model, ...)
    if early_stop.step(epoch, val_loss):
        break

# early_stop.best_epoch — эпоха с лучшим val_loss (для восстановления чекпойнта)
```

Если `model_path`/`forecaster` не переданы, `GBMEarlyStopping` сам ищет файл
`forecaster.pkl` в текущей директории и загружает его, если он есть. Если ни
прогнозист, ни файл по умолчанию не найдены — с предупреждением откатывается
на линейную экстраполяцию последних эпох (обучение при этом не падает, но
решение об остановке менее точное). Если же `model_path` передан явно, а
файла по этому пути нет — это ошибка, а не тихий откат.

## Адаптеры

### PyTorch Lightning

```python
from gbm_early_stopping.adapters.lightning import GBMEarlyStoppingCallback

trainer = pl.Trainer(callbacks=[
    GBMEarlyStoppingCallback(monitor="val_loss", model_path="forecaster.pkl",
                              epoch_penalty=0.003),
])
```

### Keras / TensorFlow

```python
from gbm_early_stopping.adapters.keras import GBMEarlyStoppingCallback

model.fit(
    x_train, y_train, validation_data=(x_val, y_val),
    callbacks=[GBMEarlyStoppingCallback(monitor="val_loss",
                                         model_path="forecaster.pkl")],
)
```

Требует `tensorflow` (`pip install gbm-early-stopping[keras]`).

### Произвольный цикл с колбэками по эпохам (PyTorch Ignite и другие)

```python
from gbm_early_stopping.adapters.callback import EpochEndEarlyStopping

early_stop = EpochEndEarlyStopping(monitor="val_loss", model_path="forecaster.pkl")

@engine.on(Events.EPOCH_COMPLETED)
def _check_stop(engine):
    logs = {"val_loss": engine.state.metrics["val_loss"]}
    if early_stop.on_epoch_end(engine.state.epoch, logs):
        engine.terminate()
```

## Как это устроено

`GBMForecaster` строит из префикса кривой (нормированного делением на
`val_loss[0]`) ~20 безразмерных признаков — наклоны на разных окнах, кривизну,
улучшение за последние эпохи, эпохи с последнего минимума, смены знака дельты
и т.д. (`gbm_early_stopping.features_from_prefix`) — и обучает
`sklearn.HistGradientBoostingRegressor` (или `xgboost`, `model_kind="xgb"`)
предсказывать `rel_gain` напрямую, без авторегрессионного прогноза кривой
по шагам.

Помимо `rel_gain`, `GBMForecaster.predict_plateau()` оценивает, через сколько
эпох `best-so-far` выйдет на плато — диагностическая величина, в решение об
остановке не входит, но полезна для анализа.

## Ограничения

* Прогнозист — не замена валидации на своих данных: обучайте его на кривых
  из своего пайплайна и проверяйте на отложенных задачах/архитектурах, иначе
  оценка `rel_gain` будет завышенной.
* Более точный прогноз `rel_gain` не гарантирует, что стратегия остановки
  экономит больше эпох при той же просадке качества относительно honest
  baseline (`patience`-based early stopping, доведённый почти до oracle) —
  сравнивайте по итоговому компромиссу «эпохи ↔ качество» на своих задачах.
* На кривых, которые ещё долго и гладко падают (например, глубокие сети без
  выхода на плато за разумное число эпох), любая стратегия остановки по
  прогнозу будет систематически терять больше качества, чем `patience`.

## Тесты

```bash
pip install -e "./gbm_early_stopping[dev]"
pytest gbm_early_stopping/tests
```
