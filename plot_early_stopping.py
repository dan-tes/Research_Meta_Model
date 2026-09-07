"""Отрисовка графиков по данным eval_early_stopping.py.

    python eval_early_stopping.py      # сначала посчитать
    python plot_early_stopping.py      # потом нарисовать -> results/fig*.png

Что где менять — docs/early_stopping.md.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

COL = {"early": "#666666", "smart_trend": "#1f77b4",
       "smart_meta": "#9467bd", "param": "#2ca02c"}
MRK = {"early": "o", "smart_trend": "s", "smart_meta": "P", "param": "D"}
# Обобщённые метки: не привязаны к конкретному механизму прогноза. «Умная»
# стратегия по линейному тренду подписана просто «Smart»; smart_meta
# встречается только там, где обучен табличный GBM-прогнозист (вариант C).
LBL = {"early": "Early stopping", "smart_trend": "Smart (trend)",
       "smart_meta": "Smart (GBM)", "param": "Parametric"}
SHORT = {"early": "early", "smart_trend": "smart",
         "smart_meta": "s-gbm", "param": "param"}
STRATS = ["early", "smart_trend", "smart_meta", "param"]

# Прогнозисты кривой для fig4/fig5: (цвет, маркер, подпись).
FC_STYLE = {
    "trend": (COL["smart_trend"], "s", "линейный тренд (текущий)"),
    "param": (COL["param"], "D", "параметрический (exp-fit)"),
}


def _present(df):
    """Стратегии из STRATS, реально присутствующие в данных (в порядке STRATS)."""
    have = set(df["strat"].unique())
    return [s for s in STRATS if s in have]


def _tasks(df):
    order = ["MNIST", "CIFAR10", "Wine"]
    return [t for t in order if t in df.task.unique()] + \
           [t for t in df.task.unique() if t not in order]


def fig_vs_size(D, out):
    df = pd.read_csv(f"{D}/bench_vs_size.csv")
    tasks = _tasks(df)
    g = df.groupby(["task", "train_size", "strat"]).agg(
        epochs=("epochs", "mean"), quality=("quality", "mean"),
        oracle=("oracle", "mean")).reset_index()
    fig, ax = plt.subplots(len(tasks), 2, figsize=(11, 3.1 * len(tasks)), squeeze=False)
    for i, t in enumerate(tasks):
        sub = g[g.task == t]
        hold = bool(df[df.task == t]["holdout"].iloc[0])
        orc = sub.groupby("train_size")["oracle"].mean()
        ax[i, 1].plot(orc.index, orc.values, "--", color="k", lw=1.3, label="oracle (полный прогон)")
        for strat in _present(df):
            d = sub[sub.strat == strat].sort_values("train_size")
            ax[i, 0].plot(d.train_size, d.epochs, "o-", color=COL[strat], label=LBL[strat])
            ax[i, 1].plot(d.train_size, d.quality, "o-", color=COL[strat], label=LBL[strat])
        is_clf = bool(df[df.task == t].get("is_classifier", pd.Series([True])).iloc[0])
        metric = "val accuracy" if is_clf else "R²"
        tag = "held-out для прогнозиста" if hold else "в обучении прогнозиста"
        ax[i, 0].set_title(f"{t} — эпохи до остановки  ({tag})")
        ax[i, 1].set_title(f"{t} — итоговое качество ({metric})")
        ax[i, 0].set_ylabel("эпохи"); ax[i, 1].set_ylabel(metric)
        for c in (0, 1):
            ax[i, c].set_xlabel("train_size")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Стратегии остановки vs размер обучающей выборки", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(f"{out}/fig1_vs_size.png"); plt.close(fig)


def fig_tradeoff(D, out):
    df = pd.read_csv(f"{D}/bench_vs_size.csv")
    tasks = _tasks(df)
    g = df.groupby(["task", "train_size", "strat"]).agg(
        epochs=("epochs", "mean"), quality=("quality", "mean"),
        oracle=("oracle", "mean")).reset_index()
    g["gap"] = g["oracle"] - g["quality"]
    fig, ax = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    for j, t in enumerate(tasks):
        sub = g[g.task == t]
        for strat in _present(df):
            d = sub[sub.strat == strat]
            ax[0, j].scatter(d.epochs, d.gap, s=85, color=COL[strat], marker=MRK[strat],
                             label=LBL[strat], edgecolor="w", lw=.6, alpha=.85, zorder=3)
        ax[0, j].axhline(0, color="k", lw=.8)
        ax[0, j].set_title(t); ax[0, j].set_xlabel("эпохи (меньше = быстрее)")
        ax[0, j].set_ylabel("потеря качества до oracle")
        ax[0, j].invert_yaxis()
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Компромисс скорость / качество: левый-верхний угол — идеал", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(f"{out}/fig2_tradeoff.png"); plt.close(fig)


def fig_sweep(D, out):
    df = pd.read_csv(f"{D}/holdout_sweep.csv")
    df["gap"] = df["oracle"] - df["q"]
    tasks = _tasks(df)
    fig, ax = plt.subplots(2, len(tasks), figsize=(5.2 * len(tasks), 7), squeeze=False)
    for j, t in enumerate(tasks):
        sub = df[df.task == t]
        early = sub[sub.strat == "early"].iloc[0]
        for strat in [s for s in ("smart_trend", "smart_meta", "param") if s in set(sub.strat.unique())]:
            d = sub[sub.strat == strat].sort_values("penalty")
            ax[0, j].plot(d.penalty, d.ep, "o-", color=COL[strat], label=LBL[strat])
            ax[1, j].plot(d.penalty, d.gap, "o-", color=COL[strat], label=LBL[strat])
        ax[0, j].axhline(early.ep, color=COL["early"], ls="--", label="early")
        ax[1, j].axhline(early.oracle - early.q, color=COL["early"], ls="--", label="early")
        pens = sorted(sub[sub.strat == "smart_trend"].penalty.dropna().unique())
        for r in (0, 1):
            ax[r, j].set_xscale("log"); ax[r, j].set_xlabel("EPOCH_PENALTY")
            ax[r, j].set_xticks(pens); ax[r, j].set_xticklabels([f"{p:g}" for p in pens], fontsize=8)
            ax[r, j].minorticks_off()
        ax[0, j].set_title(f"{t} — эпохи vs штраф"); ax[1, j].set_title(f"{t} — gap vs штраф")
        ax[0, j].set_ylabel("эпохи"); ax[1, j].set_ylabel("gap до oracle")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Held-out: как штраф двигает компромисс (early — пунктир-референс)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(f"{out}/fig3_sweep.png"); plt.close(fig)


def fig_forecast(D, out):
    df = pd.read_csv(f"{D}/forecast_mae.csv")
    methods = [m for m in ("trend", "param") if m in df.columns]
    g = df.groupby(["task", "horizon"])[methods].mean().reset_index()
    tasks = sorted(g.task.unique())
    fig, ax = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), squeeze=False)
    for j, t in enumerate(tasks):
        d = g[g.task == t]
        for m in methods:
            color, mk, lbl = FC_STYLE[m]
            ax[0, j].plot(d.horizon, d[m], marker=mk, ls="-", color=color, label=lbl)
        ax[0, j].set_title(t); ax[0, j].set_xlabel("шаг прогноза вперёд")
        ax[0, j].set_ylabel("rel. MAE (в долях val_loss[0])")
    ax[0, 0].legend(fontsize=9)
    fig.suptitle("Точность прогноза кривой на HELD-OUT задачах: текущий метод vs параметрический",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(f"{out}/fig4_forecast.png"); plt.close(fig)


def fig_examples(D, out):
    ex = json.load(open(f"{D}/example_curves.json"))
    fig, ax = plt.subplots(len(ex), 1, figsize=(10, 3.4 * len(ex)), squeeze=False)
    ax = ax[:, 0]
    for a, e in zip(ax, ex):
        vl = np.array(e["vl"]); ep = np.arange(1, len(vl) + 1)
        lo, hi = vl.min(), vl.max(); pad = (hi - lo) * 0.12
        a.plot(ep, vl, color="#222", lw=1.8, label="val_loss (факт, полный прогон)")
        kf = e["forecast_at"]
        for key in ("trend_fc", "param_fc"):
            if key not in e:
                continue
            color, mk, lbl = FC_STYLE[key[:-3]]
            fc = e[key]
            fx = np.arange(kf + 1, kf + 1 + len(fc))
            a.plot(fx, np.clip(fc, lo - pad, hi + pad), ls="--", marker=mk, color=color,
                   ms=4, lw=2, label=f"прогноз из эп. {kf}: {lbl}")
        a.axvline(e["oracle_ep"], color="gold", lw=2.6, label=f"argmin val_loss ({e['oracle_ep']})")
        buckets = {}
        for strat, se in e["stops"].items():
            buckets.setdefault(se, []).append(strat)
        for se, strats in buckets.items():
            a.axvline(se, color=COL[strats[0]] if len(strats) == 1 else "#555", lw=1.6, alpha=.8)
            # подписи снизу, у оси X — иначе их перекрывает легенда (CIFAR)
            a.text(se, lo - pad * 0.35, " ".join(SHORT[s] for s in strats), fontsize=7.5,
                   ha="center", va="top",
                   color=COL[strats[0]] if len(strats) == 1 else "#333")
        a.set_ylim(lo - pad * 1.7, hi + pad * 1.3)
        a.set_title(f"{e['task']}  — где остановилась каждая стратегия")
        a.set_xlabel("эпоха"); a.set_ylabel("val_loss")
        a.legend(fontsize=7.5, loc="best")
    fig.suptitle("Почему стратегии останавливаются там, где останавливаются", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(f"{out}/fig5_examples.png"); plt.close(fig)


def fig_calibration(D, out):
    """fig6 — ядро варианта C: калибровка GBM-прогноза `relgain` на HELD-OUT
    кривых и как ошибка растёт с зашумлённостью кривой.

    A: предсказанное vs истинное ещё-доступное улучшение за 10 эпох (диагональ —
       идеал), точки по held-out окнам, цвет — задача.
    B: средняя |ошибка| прогноза в зависимости от числа смен знака дельты
       val_loss на последних 8 эпохах. Мало смен знака = кривая ещё гладко
       падает — именно там линейный тренд экстраполирует наклон слишком далеко
       и промахивается в разы; GBM держит ошибку ровной.
    """
    import json as _json

    import meta_forecaster as MF

    X, y_rg, _, groups = MF.build_dataset(f"data/curves_eval.jsonl")
    b = MF.load_meta()
    pred_g = np.clip(b["relgain"].predict(X), 0.0, 1.0)

    trend_g = []
    for line in open("data/curves_eval.jsonl"):
        v = np.asarray(_json.loads(line)["val_loss"], float)
        if len(v) < MF.PREFIX_MIN + 3:
            continue
        for k in range(MF.PREFIX_MIN, len(v) - 2, MF.STEP):
            trend_g.append(MF.trend_rel_gain(v[:k]))
    trend_g = np.asarray(trend_g[:len(y_rg)])

    y = y_rg.to_numpy()
    noise = X["sign_changes_8"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    tasks = sorted(set(groups))
    pal = dict(zip(tasks, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]))
    hi = float(max(y.max(), pred_g.max()) * 1.02)
    for t in tasks:
        m = groups == t
        ax[0].scatter(y[m], pred_g[m], s=8, alpha=.25, color=pal[t], label=t, lw=0)
    ax[0].plot([0, hi], [0, hi], "k--", lw=1, label="идеал")
    ax[0].set_xlim(0, hi); ax[0].set_ylim(0, hi)
    ax[0].set_xlabel("истинное улучшение за 10 эпох (relgain)")
    ax[0].set_ylabel("прогноз GBM")
    ax[0].set_title(f"A. Калибровка на held-out\nMAE {np.abs(pred_g - y).mean():.4f}  "
                    f"R² {1 - ((pred_g - y) ** 2).sum() / ((y - y.mean()) ** 2).sum():+.2f}")
    lg = ax[0].legend(fontsize=8, markerscale=2)
    for h in lg.legend_handles:
        h.set_alpha(1)

    bins = [0, 1, 2, 3, 8]
    lbl = ["0", "1", "2", "3", "4+"]
    gi = np.digitize(noise, bins) - 1
    gi = np.clip(gi, 0, len(lbl) - 1)
    xs = np.arange(len(lbl))
    g_err = [np.abs(pred_g - y)[gi == i].mean() if (gi == i).any() else np.nan for i in xs]
    t_err = [np.abs(trend_g - y)[gi == i].mean() if (gi == i).any() else np.nan for i in xs]
    ax[1].bar(xs - .19, g_err, .38, color=COL["smart_meta"], label="GBM (вариант C)")
    ax[1].bar(xs + .19, t_err, .38, color=COL["smart_trend"], label="линейный тренд")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(lbl)
    ax[1].set_xlabel("смен знака Δval_loss за 8 эпох   (0 = кривая ещё гладко падает)")
    ax[1].set_ylabel("средняя |ошибка| прогноза relgain")
    ax[1].set_title("B. Провал линейного тренда — гладко-падающие\nкривые; GBM устойчив везде")
    ax[1].legend(fontsize=9)
    fig.suptitle("Вариант C: прогнозист «сколько улучшения ещё осталось» (held-out задачи)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(f"{out}/fig6_calibration.png"); plt.close(fig)


def fig_headline(D, out):
    """fig7 — одна сводная картинка: сколько эпох экономит каждая стратегия и
    во что это обходится по качеству. Всё нормировано, чтобы MNIST/CIFAR/Wine
    можно было положить на одни оси:
      x = эпохи как доля от `early` (влево = быстрее),
      y = потеря качества до oracle как доля oracle (вверх = хуже).
    Точка — среднее по всем (задача, размер, seed), усы — ±1 s.e. по задачам.
    Левый-нижний угол — идеал.
    """
    df = pd.read_csv(f"{D}/bench_vs_size.csv")
    g = df.groupby(["task", "train_size", "run", "strat"]).agg(
        epochs=("epochs", "mean"), quality=("quality", "mean"),
        oracle=("oracle", "mean")).reset_index()
    base = g[g.strat == "early"][["task", "train_size", "run", "epochs"]].rename(
        columns={"epochs": "early_ep"})
    g = g.merge(base, on=["task", "train_size", "run"])
    g["ep_frac"] = g["epochs"] / g["early_ep"]
    g["q_loss"] = (g["oracle"] - g["quality"]) / g["oracle"].abs()

    per_task = g.groupby(["strat", "task"])[["ep_frac", "q_loss"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for strat in _present(df):
        d = per_task[per_task.strat == strat]
        mx, my = d.ep_frac.mean(), d.q_loss.mean()
        sx = d.ep_frac.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        sy = d.q_loss.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt=MRK[strat], ms=13, color=COL[strat],
                    capsize=3, lw=1.4, mec="w", label=LBL[strat], zorder=3)
        for _, r in d.iterrows():
            ax.scatter(r.ep_frac, r.q_loss, s=26, color=COL[strat], alpha=.35, lw=0, zorder=2)
        loff = {"smart_trend": (10, 7), "smart_meta": (10, -16),
                "param": (10, 9), "early": (-4, 12)}.get(strat, (9, 6))
        ax.annotate(LBL[strat], (mx, my), textcoords="offset points", xytext=loff,
                    fontsize=9, color=COL[strat], fontweight="bold")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(1, color=COL["early"], ls="--", lw=1, label="early (референс)")
    ax.set_xlabel("эпохи обучения  (доля от early — влево = быстрее)")
    ax.set_ylabel("потеря качества до oracle  (доля oracle — вверх = хуже)")
    ax.set_title("Итог: экономия эпох против цены по качеству\n"
                 "каждая стратегия при своём штрафе (PENALTY_BY_STRAT); ↙ угол — идеал",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/fig7_headline.png"); plt.close(fig)


FIGS = {"fig1": fig_vs_size, "fig2": fig_tradeoff, "fig3": fig_sweep,
        "fig4": fig_forecast, "fig5": fig_examples,
        "fig6": fig_calibration, "fig7": fig_headline}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS_DIR, help="где лежат CSV и куда класть PNG")
    ap.add_argument("--only", nargs="*", default=None, help="подмножество: fig1..fig5")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    for name, fn in FIGS.items():
        if a.only and name not in a.only:
            continue
        try:
            fn(a.dir, a.dir); print(name, "ok")
        except FileNotFoundError as e:
            print(name, "пропущен — нет данных:", e)
        except Exception as e:
            print(name, "FAIL", repr(e))
