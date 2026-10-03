# Руководство пользователя: gbm-early-stopping

Полный справочник по библиотеке. Краткий старт — в [README.md](../README.md);
здесь — подробности по каждому классу, параметрам и типичным ситуациям.

## Содержание

1. [Идея метода](#1-идея-метода)
2. [Установка](#2-установка)
3. [`GBMForecaster` — обучение прогнозиста](#3-gbmforecaster--обучение-прогнозиста)
4. [`GBMEarlyStopping` — коллбэк остановки](#4-gbmearlystopping--коллбэк-остановки)
5. [Адаптеры под экосистему](#5-адаптеры-под-экосистему)
6. [Как подбирать параметры](#6-как-подбирать-параметры)
7. [Диагностика и типичные ошибки](#7-диагностика-и-типичные-ошибки)
8. [Ограничения метода](#8-ограничения-метода)
9. [Справочник API](#9-справочник-api)

---

## 1. Идея метода

Классический `patience`-based early stopping останавливает обучение, когда
`val_loss` не улучшается N эпох подряд. Это надёжно, но не экономит эпохи там,
где кривая ещё чуть-чуть улучшается, но уже не окупает время обучения.

`gbm-early-stopping` вместо этого оценивает **прогноз**: сколько относительного
улучшения `val_loss` ещё доступно за горизонт в `horizon` эпох вперёд. Оценку
даёт табличный градиентный бустинг (`GBMForecaster`), обученный на признаках
префикса кривой (наклоны, кривизна, улучшение за последние эпохи и т.д.).
Обучение останавливается, когда

```
rel_gain < epoch_penalty * horizon
```

подряд `patience` раз. `epoch_penalty` — минимальное относительное улучшение
`val_loss` за эпоху, которое оправдывает продолжение обучения; больше
`epoch_penalty` — раньше остановка.

Важно: точный прогноз `rel_gain` **не гарантирует**, что стратегия остановки
экономит больше эпох при той же просадке качества, чем честный
`patience`-based baseline. Это калибруется отдельно под вашу задачу (см.
[§6](#6-как-подбирать-параметры)).

## 2. Установка



```bash
pip install -e ./gbm_early_stopping
# с адаптером под PyTorch Lightning
pip install -e "./gbm_early_stopping[lightning]"
# с адаптером под Keras/TensorFlow
pip install -e "./gbm_early_stopping[keras]"
# с поддержкой xgboost как альтернативной модели
pip install -e "./gbm_early_stopping[xgboost]"
# для запуска тестов
pip install -e "./gbm_early_stopping[dev]"
```

Пакет не публикуется в PyPI — ставится только локально, из исходников.

## 3. `GBMForecaster` — обучение прогнозиста

### 3.1 Формат входных данных

`GBMForecaster.fit()` принимает **список кривых** — по одной последовательности
`val_loss` на прогон обучения:

```python
curves = [
    [1.8, 1.2, 0.9, 0.75, 0.68, 0.65, 0.64, 0.635, ...],  # прогон 1
    [2.1, 1.5, 1.1, 0.95, 0.88, 0.85, ...],                # прогон 2
    ...
]
```

Требования к кривой:

* длина хотя бы `prefix_min + 5` эпох (иначе она не даёт ни одного обучающего
  окна и молча пропускается);
* значения — сырой `val_loss` по эпохам, без нормировки (нормировка на
  `val_loss[0]` происходит внутри).

Чем больше и разнообразнее кривые (разные архитектуры, датасеты, гиперпараметры,
seed'ы), тем устойчивее прогноз. **Прогнозист учится именно на ваших кривых** —
если обучить его на MLP и применять к трансформерам, прогноз, скорее всего,
будет плохим (см. [§8](#8-ограничения-метода)).

Если у вас уже есть кривые в виде jsonl (по одной записи `{"val_loss": [...],
"task": "..."}` на строку), используйте `fit_from_jsonl`:

```python
from gbm_early_stopping import GBMForecaster

forecaster = GBMForecaster.fit_from_jsonl(
    ["curves_train.jsonl", "curves_train_extra.jsonl"],  # можно несколько файлов
    horizon=10, prefix_min=5,
)
```

Поле `task` используется только для честной кросс-валидации при обучении
(`GroupKFold` по задаче — окна одной задачи не текут между train/val в отчёте);
на само обучение итоговой модели не влияет. Если поля `task` нет, все кривые
считаются одной группой.

### 3.2 Конструктор

```python
GBMForecaster(
    horizon=10,       # на сколько эпох вперёд оценивается rel_gain
    prefix_min=5,      # минимальная длина префикса, с которой считается прогноз
    plateau_eps=0.02,  # см. предсказание plateau ниже
    plateau_cap=40,    # верхняя граница таргета plateau
    features=None,     # список признаков; None -> использовать набор по умолчанию
    model_kind="hgb",  # "hgb" (sklearn HistGradientBoostingRegressor) или "xgb"
)
```

`horizon` — самый важный параметр: он должен совпадать с тем, на сколько эпох
вперёд вы готовы "смотреть" при принятии решения об остановке. Если поменяете
`horizon`, модель нужно переобучить — старый `.pkl` с другим `horizon` будет
загружен корректно (он хранит свой `horizon` в конфиге), но семантика прогноза
у него останется прежней.

### 3.3 Обучение и оценка

```python
forecaster = GBMForecaster(horizon=10, prefix_min=5)
forecaster.fit(curves, groups=None, verbose=True)
```

При `verbose=True` и наличии `groups` (списка идентификаторов задачи, по
одному на кривую) выводится честная `GroupKFold`-оценка `MAE`/`R²` для обеих
голов (`relgain`, `plateau`) — так вы видите, насколько прогнозист хорош
*до* того, как он попадёт в цикл обучения. Без `groups` (или с одной группой)
кросс-валидация пропускается — обучение произойдёт, но без отчёта о качестве.

Оценку на **отложенных** задачах (не пересекающихся с обучающими) стоит делать
отдельно — прогнать `predict_rel_gain` на кривых из `curves_eval.jsonl` и
сравнить с истинным `rel_gain`, посчитанным тем же способом, что и таргет при
обучении. Простейший вариант — использовать `GBMForecaster._build_dataset()`
(приватный метод, но полезен для быстрой проверки) на eval-кривых и сравнить
`forecaster.relgain_model_.predict(X)` с `y_rg`.

### 3.4 Инференс

```python
prefix = val_loss_so_far          # список/массив val_loss с начала обучения
rel_gain = forecaster.predict_rel_gain(prefix)   # float в [0, 1] или None
plateau  = forecaster.predict_plateau(prefix)    # float >= 0 или None
```

`None` возвращается, если `len(prefix) < prefix_min` — то есть слишком рано
делать прогноз. `GBMEarlyStopping` учитывает это сам (см. ниже); при прямом
использовании `GBMForecaster` в своём коде проверяйте `None` вручную.

`predict_plateau` — диагностическая величина (через сколько эпох `best-so-far`
выйдет на `plateau_eps`-окрестность итогового минимума), в решение об
остановке не входит, но полезна для анализа кривых постфактум.

### 3.5 Сохранение и загрузка

```python
forecaster.save("forecaster.pkl")     # создаст промежуточные директории при необходимости
forecaster = GBMForecaster.load("forecaster.pkl")
```

Файл — `joblib`-дамп словаря с обеими моделями, списком признаков и конфигом.
Переносить `.pkl` между версиями Python/scikit-learn, вообще говоря,
небезопасно — держите версии окружения обучения и инференса согласованными
или переобучайте модель при обновлении зависимостей.

## 4. `GBMEarlyStopping` — коллбэк остановки

### 4.1 Конструктор

```python
GBMEarlyStopping(
    forecaster=None,       # готовый GBMForecaster
    model_path=None,       # путь к .pkl — альтернатива forecaster
    epoch_penalty=0.003,   # минимальное относит. улучшение за эпоху, оправдывающее продолжение
    min_epochs=10,         # жёсткий пол — раньше не останавливаемся
    patience=2,            # сколько раз подряд "не окупается" перед остановкой
    fallback_window=6,     # окно линейной экстраполяции (fallback)
    fallback_steps=5,      # горизонт линейной экстраполяции (fallback)
)
```

Порядок разрешения прогнозиста:

1. Если передан `forecaster` — используется он.
2. Иначе, если передан `model_path` — загружается `GBMForecaster.load(model_path)`.
   **Если файла по этому пути нет — поднимается исключение** (вы явно попросили
   конкретный файл, тихий откат здесь был бы опаснее сообщения об ошибке).
3. Иначе ищется файл `forecaster.pkl` в текущей рабочей директории
   (`gbm_early_stopping.stopping.DEFAULT_MODEL_PATH`). Если он есть — загружается
   молча.
4. Если ничего не найдено — коллбэк выдаёт `warnings.warn(...)` и работает
   только на fallback-прогнозе (см. [§4.3](#43-fallback-без-обученного-прогнозиста)).

### 4.2 Использование в цикле обучения

```python
from gbm_early_stopping import GBMEarlyStopping

early_stop = GBMEarlyStopping(epoch_penalty=0.003, min_epochs=10, patience=2)

for epoch in range(max_epochs):
    train_one_epoch(model, ...)
    val_loss = validate(model, ...)

    if early_stop.step(epoch, val_loss):
        break

# восстановить лучший чекпойнт (если вы сохраняли его по эпохам)
best_epoch = early_stop.best_epoch
```

`step(epoch, val_loss)` — единственный метод, который нужно вызывать на
каждой эпохе. Возвращает `True`, когда пора остановиться. Внутреннее состояние
(`val_loss`, `best_val_loss`, `best_epoch`, `counter`, `should_stop`) доступно
как публичные атрибуты для логирования/отладки.

Если нужно использовать один и тот же объект для нескольких независимых
прогонов (например, в кросс-валидации), сбросьте состояние между ними:

```python
early_stop.reset()
```

### 4.3 Fallback без обученного прогнозиста

Если прогнозист недоступен (не передан и файла по умолчанию нет) или падает
на конкретном префиксе (например, версия sklearn несовместима с сохранённой
моделью), `GBMEarlyStopping` откатывается на устойчивую линейную экстраполяцию
последних `fallback_window` эпох на `fallback_steps` эпох вперёд. Это тот же
принцип решения (`rel_gain < epoch_penalty * horizon`), но с горизонтом
`fallback_steps` вместо `forecaster.horizon` — и с заметно менее точной
оценкой `rel_gain`, чем у обученного GBM.

Каждый такой откат из-за сбоя прогнозиста сопровождается `warnings.warn` —
если вы видите эти предупреждения регулярно в проде, скорее всего, версии
окружения обучения прогнозиста и окружения инференса разошлись, либо кривые
слишком отличаются от тех, на которых он обучен.

## 5. Адаптеры под экосистему

Ядро (`GBMForecaster`, `GBMEarlyStopping`) не зависит от фреймворка и работает
с любым циклом обучения, который умеет сообщить `val_loss` по эпохам. Адаптеры —
тонкие обёртки под конкретные API колбэков.

### 5.1 PyTorch Lightning

```python
from gbm_early_stopping.adapters.lightning import GBMEarlyStoppingCallback

callback = GBMEarlyStoppingCallback(
    monitor="val_loss",       # ключ в trainer.callback_metrics
    epoch_penalty=0.003, min_epochs=10, patience=2,
)
trainer = pl.Trainer(callbacks=[callback])
```

Требует `pytorch-lightning` или пакет `lightning`
(`pip install gbm-early-stopping[lightning]`) — при отсутствии обеих
библиотек импорт модуля падает с понятным `ImportError`, а не абстрактной
ошибкой атрибута. Колбэк читает метрику после каждой валидационной эпохи и
выставляет `trainer.should_stop = True`, когда пора остановиться. Все
остальные аргументы конструктора (`forecaster`, `model_path`, `epoch_penalty`,
...) — те же, что у `GBMEarlyStopping`, и передаются как `**kwargs`.

### 5.2 Keras / TensorFlow

```python
from gbm_early_stopping.adapters.keras import GBMEarlyStoppingCallback

callback = GBMEarlyStoppingCallback(
    monitor="val_loss",       # ключ в logs, который передаёт Keras
    epoch_penalty=0.003, min_epochs=10, patience=2,
)
model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[callback])
```

Требует `tensorflow` (`pip install gbm-early-stopping[keras]`) — так же, как
и адаптер под Lightning, при отсутствии зависимости падает понятным
`ImportError` уже на импорте модуля. Наследуется от
`tf.keras.callbacks.Callback`, реализует `on_epoch_end(epoch, logs)` и
выставляет `self.model.stop_training = True` — то есть ведёт себя как штатный
`tf.keras.callbacks.EarlyStopping`, только решение об остановке принимает
GBM-прогнозист. Если в `logs` нет ключа `monitor`, эпоха молча пропускается
(Keras иногда не логирует часть метрик на первых шагах).

### 5.3 Произвольный цикл с колбэками по эпохам (PyTorch Ignite и другие)

```python
from gbm_early_stopping.adapters.callback import EpochEndEarlyStopping

early_stop = EpochEndEarlyStopping(monitor="val_loss", epoch_penalty=0.003)

for epoch in range(max_epochs):
    ...
    logs = {"val_loss": val_loss, "val_acc": val_acc}
    if early_stop.on_epoch_end(epoch, logs):
        break
```

С PyTorch Ignite:

```python
early_stop = EpochEndEarlyStopping(monitor="val_loss")

@engine.on(Events.EPOCH_COMPLETED)
def _check_stop(engine):
    logs = {"val_loss": engine.state.metrics["val_loss"]}
    if early_stop.on_epoch_end(engine.state.epoch, logs):
        engine.terminate()
```

`monitor` — ключ, который ищется в словаре `logs`; `KeyError`, если его там
нет (намеренно без тихого пропуска — опечатка в имени метрики иначе осталась
бы незамеченной).

## 6. Как подбирать параметры

* **`epoch_penalty`** — главный рычаг компромисса «эпохи ↔ качество». Больше
  значение — раньше стоп, меньше экономии по качеству. Подбирайте свипом по
  сетке значений на отложенных прогонах, сравнивая (а) долю сэкономленных
  эпох относительно `patience`-baseline и (б) просадку итоговой метрики
  относительно oracle (лучший чекпойнт полного прогона без остановки).
* **`min_epochs`** — жёсткий пол. Ставьте не ниже той эпохи, после которой
  кривая вашей задачи обычно уже даёт содержательный сигнал (для медленно
  сходящихся регрессий может понадобиться 15–20).
* **`patience`** — сколько раз подряд решение "не окупается" нужно принять
  перед остановкой; сглаживает случайные провалы прогноза на шумных кривых.
  `1`–`3` обычно достаточно.
* **`horizon` (в `GBMForecaster`)** — фиксируется при обучении прогнозиста,
  но должен быть согласован с тем, на сколько эпох вперёд вы реально готовы
  "смотреть" при принятии решения. Смена `horizon` требует переобучения.

Единого универсально верного набора значений нет — калибруйте на своих
задачах, а не переносите числа один в один из другого проекта.

## 7. Диагностика и типичные ошибки

| Симптом | Вероятная причина | Что делать |
|---|---|---|
| `RuntimeError: прогнозист не обучен` | вызвали `predict_rel_gain`/`predict_plateau` на `GBMForecaster()` без `fit()`/`load()` | обучите или загрузите прогнозист перед инференсом |
| `GBMEarlyStopping` тихо всегда стопает на `min_epochs + patience` | прогнозист недоступен и включился fallback, либо `epoch_penalty` слишком велик | проверьте `warnings`, убедитесь что `forecaster.pkl` находится там, где ожидается, либо передайте `model_path`/`forecaster` явно |
| `ImportError` при импорте `adapters.lightning` | не установлен `pytorch-lightning`/`lightning` | `pip install gbm-early-stopping[lightning]` |
| `ImportError` при импорте `adapters.keras` | не установлен `tensorflow` | `pip install gbm-early-stopping[keras]` |
| `predict_rel_gain` возвращает `None` | префикс короче `prefix_min` эпох | это ожидаемо на первых эпохах; `GBMEarlyStopping` обрабатывает это сам через `min_epochs` |
| Явно указанный `model_path` вызывает исключение при создании `GBMEarlyStopping` | файла по пути нет | это осознанное поведение (см. [§4.1](#41-конструктор)) — проверьте путь |
| Прогноз резко хуже на новых данных | прогнозист обучен на кривых другого пайплайна/архитектуры | переобучите на кривых из своего пайплайна (см. [§8](#8-ограничения-метода)) |

## 8. Ограничения метода

* Прогнозист — не замена валидации на своих данных. Обучайте его на кривых
  из своего пайплайна и проверяйте на отложенных задачах/архитектурах;
  перенос между сильно разными пайплайнами (другой оптимизатор, другая
  форма кривой) снижает точность прогноза.
* Более точный прогноз `rel_gain` сам по себе не гарантирует лучшую
  стратегию остановки — решение принимается ради итоговой метрики качества,
  а прогнозируется `val_loss`; разрыв между ними может съедать выигрыш в
  точности прогноза.
* На кривых, которые долго и гладко продолжают падать (например, глубокие
  сети без выхода на плато за разумное число эпох), любая стратегия
  остановки по прогнозу теряет больше качества, чем классический `patience`
  — это фундаментальное ограничение подхода, а не баг конкретной реализации.
* `.pkl`-файлы моделей не гарантированно переносимы между несовместимыми
  версиями `scikit-learn`/`joblib`.

## 9. Справочник API

### `gbm_early_stopping.features_from_prefix(prefix, prefix_min=5) -> dict | None`
Признаки префикса кривой (нормированные на `prefix[0]`). `None`, если
`len(prefix) < prefix_min`.

### `class gbm_early_stopping.GBMForecaster`
* `__init__(horizon=10, prefix_min=5, plateau_eps=0.02, plateau_cap=40, features=None, model_kind="hgb")`
* `fit(curves, groups=None, verbose=False) -> self`
* `classmethod fit_from_jsonl(paths, val_loss_key="val_loss", task_key="task", **kwargs) -> GBMForecaster`
* `predict_rel_gain(prefix) -> float | None`
* `predict_plateau(prefix) -> float | None`
* `save(path)`
* `classmethod load(path) -> GBMForecaster`
* Атрибуты после `fit()`/`load()`: `relgain_model_`, `plateau_model_` (обученные регрессоры sklearn/xgboost)

### `class gbm_early_stopping.GBMEarlyStopping`
* `__init__(forecaster=None, model_path=None, epoch_penalty=0.003, min_epochs=10, patience=2, fallback_window=6, fallback_steps=5)`
* `step(epoch, val_loss) -> bool`
* `reset()`
* Атрибуты: `val_loss` (список), `best_val_loss`, `best_epoch`, `counter`, `should_stop`, `forecaster`

### `class gbm_early_stopping.adapters.lightning.GBMEarlyStoppingCallback(pl.Callback)`
* `__init__(monitor="val_loss", **kwargs)` — `**kwargs` пробрасываются в `GBMEarlyStopping`
* Атрибуты: `should_stop`, `best_epoch`

### `class gbm_early_stopping.adapters.keras.GBMEarlyStoppingCallback(tf.keras.callbacks.Callback)`
* `__init__(monitor="val_loss", **kwargs)` — `**kwargs` пробрасываются в `GBMEarlyStopping`
* `on_epoch_end(epoch, logs=None)` — выставляет `self.model.stop_training = True`
* Атрибуты: `should_stop`, `best_epoch`

### `class gbm_early_stopping.adapters.callback.EpochEndEarlyStopping`
* `__init__(monitor="val_loss", **kwargs)` — `**kwargs` пробрасываются в `GBMEarlyStopping`
* `on_epoch_end(epoch, logs) -> bool`
* Атрибуты: `should_stop`, `best_epoch`
