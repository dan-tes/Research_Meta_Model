"""Симуляция стратегий ранней остановки на уже посчитанных кривых LCBench
held-out (`data/curves_eval_lcbench.jsonl`) — без реального обучения: кривые
уже полностью посчитаны Auto-PyTorch, так что это прямая замена `run()` из
`eval_early_stopping.py`, но без GPU-тренировки.

Отвечает на вопрос: раз GBM с LCBench в обучении заметно точнее как
*прогнозист* (см. `meta_forecaster.py eval`), становится ли он лучше и как
*стратегия остановки*? (см. `docs/variant_c_forecaster.md`, раздел LCBench —
короткий ответ: нет, паритет со smart_trend, как и без LCBench).

    python sim_stopping_lcbench.py
"""
import json

import numpy as np
import pandas as pd

from pytorch_version import ParametricEarlyStopping, SimpleEarlyStopping, SmartEarlyStoppingMultiStep

EVAL_PATH = "data/curves_eval_lcbench.jsonl"
PENALTY = {"smart_trend": 0.006, "smart_meta": 0.003, "param": 0.010}

STRATS = [
    ("early", None),
    ("smart_trend", None),
    ("smart_meta_baseline", "models/meta_forecaster_baseline.pkl"),
    ("smart_meta_pluslc", "models/meta_forecaster.pkl"),
    ("param", None),
]


def make_cb(strat, meta_path=None):
    if strat == "early":
        return SimpleEarlyStopping(patience=5)
    if strat == "smart_trend":
        return SmartEarlyStoppingMultiStep(epoch_penalty=PENALTY["smart_trend"])
    if strat == "param":
        return ParametricEarlyStopping(epoch_penalty=PENALTY["param"])
    if strat.startswith("smart_meta"):
        return SmartEarlyStoppingMultiStep(epoch_penalty=PENALTY["smart_meta"],
                                           use_meta=True, meta_path=meta_path)
    raise ValueError(strat)


def simulate(val_loss, val_metric, strat, meta_path=None):
    """Реплицирует run() из eval_early_stopping.py на уже готовой кривой:
    quality = val_metric на эпохе с лучшим val_loss до точки остановки."""
    cb = make_cb(strat, meta_path)
    best_vl, best_vm, stop_ep = np.inf, val_metric[0], len(val_loss)
    for ep, (vl, vm) in enumerate(zip(val_loss, val_metric)):
        if vl < best_vl:
            best_vl, best_vm = vl, vm
        stop = cb.step(vl) if strat == "early" else cb.step(ep, vl)
        if stop:
            stop_ep = ep + 1
            break
    return stop_ep, best_vm


def main():
    rows = []
    with open(EVAL_PATH) as fh:
        for line in fh:
            rec = json.loads(line)
            vl, vm = rec["val_loss"], rec["val_metric"]
            oracle = max(vm)
            for strat, meta_path in STRATS:
                ep, q = simulate(vl, vm, strat, meta_path)
                rows.append(dict(task=rec["task"], strat=strat, epochs=ep,
                                 quality=q, oracle=oracle,
                                 gap_pct=(oracle - q) / oracle * 100 if oracle > 0 else 0.0))

    df = pd.DataFrame(rows)
    g = df.groupby("strat").agg(epochs=("epochs", "mean"), gap_pct=("gap_pct", "mean"))
    g["epochs_ratio_vs_early"] = g["epochs"] / g.loc["early", "epochs"]
    print(g.reindex([s for s, _ in STRATS]))


if __name__ == "__main__":
    main()
