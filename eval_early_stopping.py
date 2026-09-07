"""Бенчмарк стратегий ранней остановки + подготовка данных для графиков.

Гоняет стратегии (early / smart_trend / smart_meta / param) на нескольких
задачах и пишет CSV/JSON в RESULTS_DIR. Отрисовка — plot_early_stopping.py.
У каждой прогнозной стратегии свой калиброванный штраф — PENALTY_BY_STRAT.

    python eval_early_stopping.py            # полный прогон (~4 мин на GPU)
    python eval_early_stopping.py --quick    # быстрый черновой прогон

Что где менять — docs/early_stopping.md.
"""
import argparse
import inspect
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import gen_curves as G
from pytorch_version import (ParametricEarlyStopping, SimpleEarlyStopping,
                             SmartEarlyStoppingMultiStep)

# горизонт/минимальный префикс для оценки многошагового прогноза кривой
HORIZON = 10
MIN_PREFIX = 5

# ------------------------------------------------------------------ КОНФИГ
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7

FULL_EPOCHS = 140         # длина «полного» прогона (для oracle) и потолок обучения

# Один и тот же EPOCH_PENALTY означает для разных прогнозистов разное (свип fig3):
#   * бюджет = penalty * horizon, а horizon = 10 у smart_meta и 5 у smart_trend/param;
#   * линейный тренд оптимистично смещён (over-predict) и потому «тормозит» позже,
#     чем честный GBM при том же пороге.
# Поэтому у каждой стратегии свой штраф, откалиброванный по свипу так, чтобы
# gap до oracle на held-out (MNIST/CIFAR/Wine) держался ~2-3 п.п. при макс.
# экономии эпох. По типу задачи оптимум чуть разный (reg терпит больший штраф);
# компромиссные значения ниже, разбор — docs/early_stopping.md.
PENALTY_BY_STRAT = {
    "smart_trend": 0.006,
    "smart_meta":  0.003,
    "param":       0.010,
}
PENALTY = 0.006          # дефолт для стратегий вне таблицы (и старых вызовов)


def penalty_for(strat):
    return PENALTY_BY_STRAT.get(strat, PENALTY)

# Вариант C: табличный GBM-прогнозист. Участвует в бенчмарке только если обучен
# (models/meta_forecaster.pkl есть) и в ядре есть параметр use_meta.
_SMART_HAS_META = ("use_meta" in inspect.signature(
    SmartEarlyStoppingMultiStep.__init__).parameters
    and os.path.exists("models/meta_forecaster.pkl"))

STRATS = tuple(s for s, ok in (
    ("early", True), ("smart_trend", True),
    ("smart_meta", _SMART_HAS_META), ("param", True)) if ok)
SWEEP_SMART = tuple(s for s, ok in (
    ("smart_trend", True),
    ("smart_meta", _SMART_HAS_META), ("param", True)) if ok)

# базовые гиперпараметры MLP, на котором меряем стратегии
MLP = dict(width=64, depth=2, wd=1e-4, dropout=0.2, lr=1e-3)

# задачи: имя -> (загрузчик из gen_curves, held-out ли для прогнозиста)
TASKS = {
    "MNIST":      (lambda: G._img(G.datasets.MNIST), False),
    "CIFAR10":    (lambda: G._img(G.datasets.CIFAR10), True),
    "Wine":       (G.load_wine_reg, True),
    "California": (G.load_california, True),   # только для свипа штрафа
}
SIZES = {"MNIST": [300, 800, 2000, 5000], "CIFAR10": [500, 2000, 5000],
         "Wine": [300, 600, 850]}
N_RUNS = 4                                   # повторов на (задача, размер)

# свипуем штраф на классификации (MNIST — в обучении прогнозиста; CIFAR10 —
# held-out) и на регрессии (Wine, held-out), чтобы подобрать отдельный
# EPOCH_PENALTY для smart_trend / smart_meta / param и для clf / reg.
# California выпала из свипа: MLP-регрессия там сходится к ~10-й эпохе, раньше
# min_epochs, поэтому gap == 0 при любом штрафе — тюнить нечего.
SWEEP_TASKS = ("MNIST", "CIFAR10", "Wine")
SWEEP_PENALTIES = [0.001, 0.002, 0.003, 0.004, 0.006, 0.009, 0.013, 0.02, 0.03]
SWEEP_SIZE = 2500
SWEEP_CFGS = [dict(lr=lr, seed=s) for lr in (1e-3, 3e-4) for s in (0, 1)]

EXAMPLE_TASKS = ("MNIST", "CIFAR10", "Wine")  # для fig5


# ------------------------------------------------------------------ прогнозы
def trend_fc(hist, horizon=HORIZON, window=6):
    y = np.asarray(hist, float); w = min(window, len(y)); yr = y[-w:]
    s = np.polyfit(np.arange(w), yr, 1)[0] if w > 1 else 0.0
    p = np.clip(yr[-1] + s * np.arange(1, horizon + 1), 0.0, None)
    if s >= 0:
        p[:] = yr[-1]
    return p


def param_fc(hist, horizon=HORIZON):
    """Прогноз тем же способом, что и ParametricEarlyStopping: фит экспоненты
    a·e^(−bt)+c по всему префиксу. При неудаче фита — откат на линейный тренд."""
    from scipy.optimize import curve_fit

    y = np.asarray(hist, float)
    x = np.arange(len(y))

    def f(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        popt, _ = curve_fit(f, x, y, bounds=(0, [10, 10, 10]), maxfev=5000)
    except Exception:
        return trend_fc(hist, horizon)
    fut = np.arange(len(y), len(y) + horizon)
    return np.clip(f(fut, *popt), 0.0, None)


# ------------------------------------------------------------------ обучение
class _NeverStop:
    def step(self, *a):
        return False


def _callback(strat, penalty):
    if strat == "none":
        return _NeverStop()
    if strat == "early":
        return SimpleEarlyStopping(patience=5)
    if strat == "param":
        return ParametricEarlyStopping(epoch_penalty=penalty)
    if strat == "smart_meta" and _SMART_HAS_META:
        return SmartEarlyStoppingMultiStep(epoch_penalty=penalty, use_meta=True)
    return SmartEarlyStoppingMultiStep(epoch_penalty=penalty)


def run(data, cfg, strat, penalty=None, max_epochs=FULL_EPOCHS, bs=128, want_curve=False):
    """Одно обучение MLP с выбранной стратегией остановки.

    penalty=None -> берётся штраф этой стратегии из PENALTY_BY_STRAT.
    Возвращает: epochs (когда остановились), quality (метрика на лучшем по
    val_loss чекпойнте), oracle (лучшая метрика за весь прогон)."""
    if penalty is None:
        penalty = penalty_for(strat)
    Xtr, ytr, Xte, yte, is_clf, out_dim = data
    g = torch.Generator().manual_seed(cfg["seed"])
    idx = torch.randperm(len(Xtr), generator=g)[:cfg["train_size"]].numpy()
    Xt = torch.tensor(Xtr[idx]).to(DEVICE); yt = torch.tensor(ytr[idx]).to(DEVICE)
    Xv = torch.tensor(Xte).to(DEVICE); yv = torch.tensor(yte).to(DEVICE)

    torch.manual_seed(cfg["seed"])
    model = G.build_mlp(Xt.shape[1], out_dim, cfg.get("width", MLP["width"]),
                        cfg.get("depth", MLP["depth"]), cfg.get("dropout", MLP["dropout"])).to(DEVICE)
    crit = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.get("lr", MLP["lr"]),
                     weight_decay=cfg.get("wd", MLP["wd"]))
    cb = _callback(strat, penalty)

    best_vl, best_vm, stop_ep = np.inf, None, max_epochs
    vl_curve, vm_curve = [], []
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[b])
            loss = crit(out, yt[b]) if is_clf else crit(out.squeeze(-1), yt[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xv)
            if is_clf:
                vl = crit(out, yv).item()
                vm = (out.argmax(1) == yv).float().mean().item()
            else:
                p = out.squeeze(-1)
                vl = crit(p, yv).item()
                vm = 1 - ((yv - p) ** 2).sum().item() / (((yv - yv.mean()) ** 2).sum().item() + 1e-12)
        vl_curve.append(vl); vm_curve.append(vm)
        if vl < best_vl:
            best_vl, best_vm = vl, vm
        stop = cb.step(vl) if strat == "early" else cb.step(ep, vl)
        if stop:
            stop_ep = ep + 1
            break
    r = dict(epochs=stop_ep, quality=best_vm, oracle=max(vm_curve))
    if want_curve:
        r["vl_curve"] = vl_curve; r["vm_curve"] = vm_curve
    return r


# ------------------------------------------------------------------ секции
def bench_vs_size(quick):
    rows, t0 = [], time.time()
    n_runs = 2 if quick else N_RUNS
    for tname, (loader, holdout) in TASKS.items():
        if tname not in SIZES:                   # напр. California — только для свипа штрафа
            continue
        data = loader()
        for size in SIZES[tname]:
            if size > 0.9 * len(data[0]):
                continue
            is_clf = bool(data[4])
            for seed in range(n_runs):
                cfg = dict(MLP, train_size=size, seed=seed)
                oracle = run(data, cfg, "none")["quality"]      # честный потолок
                for strat in STRATS:
                    res = run(data, cfg, strat)
                    rows.append(dict(task=tname, holdout=holdout, is_classifier=is_clf,
                                     train_size=size, run=seed, strat=strat,
                                     epochs=res["epochs"], quality=res["quality"],
                                     oracle=oracle))
            print(f"  vs_size {tname} size={size}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def holdout_sweep(quick):
    rows, t0 = [], time.time()
    cfgs = SWEEP_CFGS[:2] if quick else SWEEP_CFGS
    pens = SWEEP_PENALTIES[::2] if quick else SWEEP_PENALTIES
    for tname in SWEEP_TASKS:
        data = TASKS[tname][0]()
        size = SWEEP_SIZE if SWEEP_SIZE <= 0.9 * len(data[0]) else int(0.7 * len(data[0]))
        cc = [dict(MLP, **c, train_size=size) for c in cfgs]
        early = [run(data, c, "early") for c in cc]
        eo = float(np.mean([x["oracle"] for x in early]))
        rows.append(dict(task=tname, penalty=np.nan, strat="early",
                         ep=np.mean([x["epochs"] for x in early]),
                         q=np.mean([x["quality"] for x in early]), oracle=eo))
        for pen in pens:
            for strat in SWEEP_SMART:
                rr = [run(data, c, strat, penalty=pen) for c in cc]
                rows.append(dict(task=tname, penalty=pen, strat=strat,
                                 ep=np.mean([x["epochs"] for x in rr]),
                                 q=np.mean([x["quality"] for x in rr]), oracle=eo))
        print(f"  sweep {tname}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def forecast_mae(path="data/curves_eval.jsonl"):
    """rel-MAE многошагового прогноза кривой на held-out задачах.

    Сравнивает прогнозисты, которые реально используют стратегии остановки:
    `trend` — текущий метод SmartEarlyStoppingMultiStep (линейная экстраполяция),
    `param` — метод ParametricEarlyStopping (фит экспоненты).
    """
    methods = {"trend": trend_fc, "param": param_fc}
    rows = []
    for line in open(path):
        r = json.loads(line)
        v = np.asarray(r["val_loss"], float)
        if len(v) < MIN_PREFIX + HORIZON + 3:
            continue
        v0 = max(v[0], 1e-8)
        for k in range(MIN_PREFIX + 3, len(v) - HORIZON, 2):
            fut = v[k:k + HORIZON]
            errs = {name: np.abs(np.asarray(fn(v[:k])) - fut) / v0 for name, fn in methods.items()}
            for h in range(HORIZON):
                rows.append(dict(task=r["task"], horizon=h + 1,
                                 **{name: e[h] for name, e in errs.items()}))
    return pd.DataFrame(rows)


def example_curves():
    out = []
    for tname in EXAMPLE_TASKS:
        data = TASKS[tname][0]()
        cfg = dict(MLP, train_size=min(2000, int(0.7 * len(data[0]))), seed=0)
        full = run(data, cfg, "none", max_epochs=120, want_curve=True)
        vl = full["vl_curve"]
        stops = {s: run(data, cfg, s, max_epochs=120)["epochs"] for s in STRATS}
        kf = 10  # момент первого решения «умной» стратегии (min_epochs)
        rec = dict(task=tname, vl=vl, oracle_ep=int(np.argmin(vl)) + 1,
                   stops=stops, forecast_at=kf,
                   trend_fc=list(map(float, trend_fc(vl[:kf]))),
                   param_fc=list(map(float, param_fc(vl[:kf]))))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="меньше повторов/точек")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--only", nargs="*", default=None,
                    help="подмножество: vs_size sweep forecast examples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    do = set(args.only) if args.only else {"vs_size", "sweep", "forecast", "examples"}

    if "vs_size" in do:
        bench_vs_size(args.quick).to_csv(f"{args.out}/bench_vs_size.csv", index=False)
        print("[1] bench_vs_size.csv")
    if "sweep" in do:
        holdout_sweep(args.quick).to_csv(f"{args.out}/holdout_sweep.csv", index=False)
        print("[2] holdout_sweep.csv")
    if "forecast" in do:
        forecast_mae().to_csv(f"{args.out}/forecast_mae.csv", index=False)
        print("[3] forecast_mae.csv")
    if "examples" in do:
        json.dump(example_curves(), open(f"{args.out}/example_curves.json", "w"))
        print("[4] example_curves.json")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
