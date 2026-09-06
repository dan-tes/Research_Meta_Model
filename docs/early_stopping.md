# Ранняя остановка: стратегии, прогнозист кривой, бенчмарк

## Что это

Три стратегии остановки обучения и обвязка для их сравнения:

| стратегия | как решает, когда стоп |
|---|---|
| `SimpleEarlyStopping` | классический `patience`: N эпох без улучшения val_loss |
| `SmartEarlyStoppingMultiStep` | прогнозирует ближайший кусок кривой val_loss и стопает, когда ожидаемое относительное улучшение за горизонт меньше `epoch_penalty · future_steps` |
| `ParametricEarlyStopping` | то же правило, но прогноз — фит экспоненты `a·e^(−bt)+c` |

В этой ветке `SmartEarlyStoppingMultiStep` решает об остановке так:
- **`smart_trend`** (дефолт) — линейная экстраполяция val_loss по окну из 6 эпох,
  `projected = min(последнее, экстраполяция)`, `rel_gain = (best − projected) / best`.
- **`smart_meta`** (`use_meta=True`, **вариант C**) — табличный GBM из
  `meta_forecaster.py` предсказывает `rel_gain` НАПРЯМУЮ: «сколько относительного
  улучшения val_loss ещё доступно за следующие 10 эпох». Модель обучена на
  кривых из этого же пайплайна (`gen_curves.py`), а не из `final.csv`, и на
  прямом таргете, а не на одношаговой дельте с авторегрессией. Включается, только
  если обучен `models/meta_forecaster.pkl`; иначе — тихий откат на тренд.

Старый сломанный `_forecast_meta` (final.csv + одношаговая дельта + авторегрессия,
систематически предсказывал «≈0» на свежих кривых) удалён — его заменил
`_forecast_meta_gain`. RNN-прогноз кривой (`_forecast_rnn`, GRU) живёт в отдельной
ветке с прогнозистом; здесь его нет.

**Текущий вывод бенчмарка:** на held-out задачах `SimpleEarlyStopping(patience=5)` держится
вплотную к oracle; «умные» стратегии (`smart_trend`, `param`) — это размен эпох на качество.
Из двух прогнозистов, участвующих в решении об остановке, линейный тренд заметно точнее
параметрического фита экспоненты на held-out кривых (fig4: rel-MAE ниже в 3–7×), поэтому
`param` при дефолтном штрафе останавливается раньше и теряет больше качества.

## Файлы

| файл | что |
|---|---|
| `pytorch_version.py` | стратегии остановки + `EPOCH_PENALTY` + модели/лоадеры |
| `curve_forecaster.py` | GRU-прогнозист кривой: модель, препроцессинг, `load_forecaster`. **В решении об остановке на этой ветке не участвует**; нужен только для `rnn`-колонки в fig4/fig5, если она включена |
| `train_curve_forecaster.py` | обучение прогнозиста → `models/curve_forecaster.pt` (нужно только для RNN-ветки) |
| `gen_curves.py` | генерация датасетов кривых → `data/curves_{train,eval}.jsonl`. `N_PER_TASK` (дефолт 100, env-переопределяемо) кривых на задачу; 8 train-задач → ~800 кривых для варианта C |
| `meta_forecaster.py` | **вариант C**: табличный GBM-прогнозист. Строит из кривых пары `(фичи префикса → таргет)`, где таргет — `relgain` (ещё доступное относит. улучшение за 10 эпох) и `plateau` (эпох до плато). `train` → `models/meta_forecaster.pkl`, `eval` — held-out метрики + сравнение с линейным трендом |
| `eval_early_stopping.py` | бенчмарк стратегий → `results/*.csv`, `results/example_curves.json` |
| `plot_early_stopping.py` | отрисовка → `results/fig1..5*.png` |
| `test.py` | старый бенчмарк на MNIST/CIFAR/Wine (`runner_pytorch` / `analyze_results`) |

## Пайплайн

```
gen_curves.py              # 1. кривые обучения (train / eval — непересекающиеся задачи)
   ↓  data/curves_train.jsonl (~800), data/curves_eval.jsonl (~300)
meta_forecaster.py train    # 2. обучить табличный GBM-прогнозист (вариант C)
   ↓  models/meta_forecaster.pkl
train_curve_forecaster.py  # 2b. обучить GRU-прогнозист (только для RNN-ветки)             [опц.]
   ↓  models/curve_forecaster.pt
eval_early_stopping.py      # 3. прогнать стратегии (early / smart_trend / smart_meta / param)
   ↓  results/*.csv, results/example_curves.json
plot_early_stopping.py      # 4. нарисовать графики
   ↓  results/fig1..5.png
```

Для `smart_meta` нужны шаги 1–2 (иначе стратегия тихо откатывается на тренд и
не попадает в бенчмарк). `smart_trend` / `param` работают без них.

```bash
N_PER_TASK=100 python gen_curves.py     # ~20-40 мин на GPU
python meta_forecaster.py               # train + held-out eval, секунды
```

```bash
python eval_early_stopping.py            # ~4 мин на GPU
python plot_early_stopping.py            # секунды
python eval_early_stopping.py --quick    # черновой прогон (меньше повторов)
python eval_early_stopping.py --only sweep examples   # пересчитать часть
python plot_early_stopping.py --only fig3 fig5
```

## Если ты что-то меняешь

### …насколько агрессивно останавливаться
`EPOCH_PENALTY` в `pytorch_version.py` (сейчас `0.03`). Это минимальное
относительное улучшение val_loss за эпоху, оправдывающее продолжение. Больше →
раньше стоп. По свипу: `0.03` почти везде упирает «умные» стратегии в пол
(`min_epochs + patience`); адаптивное поведение начинается с `~0.006` и ниже.
Значение подхватывают и `eval_early_stopping.py` (`PENALTY`, `SWEEP_PENALTIES`),
и `test.py`.

### …`min_epochs` / `patience`
Аргументы `SmartEarlyStoppingMultiStep.__init__` и `ParametricEarlyStopping.__init__`
в `pytorch_version.py` (сейчас `min_epochs=10, patience=2`). `min_epochs` — жёсткий
пол; для регрессии/медленных кривых имеет смысл 15–20 (Wine до ~15 эпох даёт R²<0).
`SimpleEarlyStopping(patience=5)` задаётся при создании колбэка.

### …окно/горизонт прогноза линейного тренда
`forecast_window` (сейчас 6) и `future_steps` (сейчас 5) — аргументы
`SmartEarlyStoppingMultiStep`. `budget = epoch_penalty · future_steps`.

### …RNN-прогноз
На этой ветке в `SmartEarlyStoppingMultiStep` нет параметра `use_rnn` и метода `_forecast_rnn`.
`eval_early_stopping.py` это определяет по сигнатуре (`_SMART_HAS_RNN`) и:
- не гоняет отдельную стратегию `smart_rnn` (была бы копией `smart_trend`);
- в `forecast_mae`/`example_curves` считает только `trend` и `param`.
Полноценный RNN-путь — в ветке с прогнозистом кривой.

### …добавить задачу в бенчмарк
`eval_early_stopping.py`: добавь в `TASKS` пару `(loader, held_out?)` и в `SIZES`
список размеров. `loader` возвращает кортеж
`(X_train, y_train, X_test, y_test, is_classifier, out_dim)` — используй готовые
`gen_curves._img`, `gen_curves._sk_clf`, `gen_curves._sk_reg` или напиши свой.
**Дисциплина held-out:** если задача пойдёт в оценку прогнозиста, её НЕ должно
быть в `gen_curves.TRAIN_TASKS`.

### …поменять архитектуру/представление прогнозиста
`curve_forecaster.py`:
- `HORIZON` (10) — на сколько эпох вперёд прогноз. Меняешь → **переобучить**.
- `MAX_LEN` (40) — сколько последних эпох подаётся на вход.
- `MIN_PREFIX` (5) — минимум эпох, иначе прогноз не считается (фолбэк на тренд).
- `N_FEATURES` (2: `[v_norm, delta]`) — если добавляешь фичи, правь и
  `_features_from_norm`, и `prefix_to_input`, и `make_windows`.
- `CurveForecaster` — сама GRU (`hidden=64`, 1 слой). Конфиг сохраняется в чекпойнт.
Нормировка кривой — деление на `val_loss[0]`; в этой ветке GRU используется только
как `rnn`-прогнозист в `eval_early_stopping.py` (`rnn_fc`), не в `pytorch_version.py`.

### …переобучить прогнозист
`python train_curve_forecaster.py`. Источник — `data/curves_train.jsonl` +
доля `FINAL_CSV_FRACTION` (0.35) кривых из `data/final.csv` как augmentation
(MNIST-Keras, вне eval-набора). Валидация считается **только** по разнородным
кривым. Сохраняется лучший по val чекпойнт в `models/curve_forecaster.pt`.

### …пересобрать датасет кривых
`N_PER_TASK=100 python gen_curves.py` → `data/curves_train.jsonl` (из
`TRAIN_TASKS`) и `data/curves_eval.jsonl` (из `EVAL_TASKS`). `configs_for` задаёт
сетку гиперпараметров на задачу (`widths/lrs/wds/drops/sizes/noises`); если
комбинаций меньше, чем `N_PER_TASK`, они добираются повторами с новым seed.
`run_curve` гоняет один MLP `max_epochs=140` и пишет `val_loss` + `val_metric`
по эпохам. Одна кривая ≈ 1–2 с на GPU.
**KMNIST/CIFAR-100 выпали** — мёртвые/медленные зеркала; если чинить — добавь
обратно в `EVAL_TASKS` и убедись, что скачивание проходит.

### …табличный GBM-прогнозист (вариант C)
`meta_forecaster.py`:
- `PREFIX_MIN` (5), `HORIZON` (10) — окно префикса и горизонт таргета `relgain`.
  Меняешь → **переобучить** (`python meta_forecaster.py train`). `HORIZON` должен
  совпадать с тем, на что умножается `epoch_penalty` в `smart_meta`
  (`SmartEarlyStoppingMultiStep` берёт `meta_horizon` из конфига `.pkl`).
- `FEATURES` — список признаков префикса; правишь → правь `features_from_prefix`.
- `MODEL_KIND` — `"hgb"` (sklearn `HistGradientBoostingRegressor`) или `"xgb"`
  (если установлен `xgboost`). Архитектура именно градиентный бустинг.
- Таргеты в `_targets_at`: `relgain` (для решения об остановке) и `plateau`
  (эпох до плато, `PLATEAU_EPS`/`PLATEAU_CAP`).
- `evaluate()` меряет на `data/curves_eval.jsonl` (held-out задачи) и сравнивает
  с линейным трендом — тем же, что в `smart_trend`.

### …параметры бенчмарка
`eval_early_stopping.py`, блок «КОНФИГ»: `PENALTY`, `FULL_EPOCHS`, `MLP`
(гиперпараметры сети, на которой меряем), `SIZES`, `N_RUNS`, `SWEEP_*`,
`EXAMPLE_TASKS`.

### …внешний вид графиков
`plot_early_stopping.py`: `COL`/`MRK`/`LBL` (цвета/маркеры/подписи стратегий),
функции `fig1..fig5`. Каждая читает свой CSV из `--dir` и пишет туда же PNG.

## Что рисуют графики

| фигура | из чего | показывает |
|---|---|---|
| `fig1_vs_size` | `bench_vs_size.csv` | эпохи и итоговое качество каждой стратегии vs `train_size`, пунктир — oracle |
| `fig2_tradeoff` | `bench_vs_size.csv` | scatter «эпохи ↔ разрыв до oracle», идеал — левый-верх |
| `fig3_sweep` | `holdout_sweep.csv` | как `EPOCH_PENALTY` двигает компромисс `smart_trend` / `smart_meta` на held-out, early — референс |
| `fig4_forecast` | `forecast_mae.csv` | rel-MAE прогноза кривой по горизонту на held-out: текущий метод (линейный тренд) vs параметрический (exp-fit) |
| `fig5_examples` | `example_curves.json` | примеры кривых: прогнозы обоих методов из точки решения + вертикали, где остановилась каждая стратегия (подписи снизу у оси X) |

## Готчи

- **held-out дисциплина.** `data/curves_eval.jsonl` и `EVAL_TASKS` не должны
  пересекаться с `data/curves_train.jsonl` / `TRAIN_TASKS` / `final.csv`. Иначе
  оценка прогнозиста завышена.
- **CIFAR в бенчмарке — это MLP на flatten(3072)**, не CNN. Кривые формой похожи
  на реальные (переобучение после ~50 эпох), но абсолютная accuracy низкая — это
  ок, меряем стратегию остановки, а не CIFAR SOTA.
- **california как прогноз-кейс разваливается**: при высоком lr MLP-регрессия
  численно расходится (loss после минимума растёт в 3–4×), такой скачок не
  предсказуем из префикса ничем. На решение об остановке не влияет — кривая
  сходится к ~10-й эпохе.
- **`oracle`** в CSV = метрика на лучшем по val_loss чекпойнте за полный прогон
  без остановки (`strat="none"`, `FULL_EPOCHS`), а не `max(val_metric)`.
- **Старая** табличная мета-модель (`_forecast_meta` на `final.csv`, одношаговая
  дельта + авторегрессия) systematically прогнозировала «изменений ≈ 0» на свежих
  кривых → остановка всегда на полу. Удалена. Вариант C (`meta_forecaster.py`,
  `smart_meta`) — её замена: те же деревья, но данные из `gen_curves.py` и прямой
  таргет `relgain`. `use_meta=True` без обученного `.pkl` = тихий откат на тренд.
- **`smart_meta` в бенчмарке появляется только если есть `models/meta_forecaster.pkl`.**
  `eval_early_stopping.py` проверяет это (`_SMART_HAS_META`); нет файла — колонки
  `smart_meta` в CSV/графиках не будет.
