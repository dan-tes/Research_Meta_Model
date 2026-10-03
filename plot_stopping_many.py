"""Много held-out кривых на одном графике + где на каждой остановилась каждая
стратегия ранней остановки (симуляция на уже посчитанных кривых, как в
sim_stopping_lcbench.py — без обучения).

Кривые нормированы на val_loss[0], чтобы задачи с разным масштабом лосса
легли на одну ось. Маркер = точка остановки стратегии на этой кривой,
звёздочка = argmin val_loss (oracle).

    python plot_stopping_many.py                       # 20 случайных кривых
    python plot_stopping_many.py --n 20 --task cifar10
    python plot_stopping_many.py --paths data/curves_eval_lcbench.jsonl
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_early_stopping import COL, LBL, MRK
from sim_stopping_lcbench import simulate

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# стратегия -> путь к GBM-модели (None для не-GBM)
STRATS = [("early", None), ("smart_trend", None),
          ("smart_meta", "models/meta_forecaster.pkl"), ("param", None)]


def load_curves(paths, task=None):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                rec = json.loads(line)
                if task is None or rec["task"] == task:
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["data/curves_eval.jsonl"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/fig_stopping_many.png")
    a = ap.parse_args()

    rows = load_curves(a.paths, a.task)
    if not rows:
        raise SystemExit("нет кривых (проверь --paths / --task)")
    rng = np.random.default_rng(a.seed)
    chosen = [rows[i] for i in rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 17), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.5]})
    stops = {s: [] for s, _ in STRATS}
    oracles = []
    for rec in chosen:
        vl = np.asarray(rec["val_loss"], dtype=np.float64)
        p = vl / max(float(vl[0]), 1e-8)
        ep = np.arange(1, len(p) + 1)
        ax.plot(ep, p, color="#9a9a9a", lw=1, alpha=.7, zorder=1)
        o = int(p.argmin())
        oracles.append(o + 1)
        ax.plot(o + 1, p[o], "*", color="gold", mec="#555", mew=.5, ms=11, zorder=3)
        for strat, meta_path in STRATS:
            se, _ = simulate(rec["val_loss"], rec["val_metric"], strat, meta_path)
            stops[strat].append(se)
            ax.plot(se, p[se - 1], MRK[strat], color=COL[strat], mec="w", mew=.8,
                    ms=8, alpha=.9, zorder=4)

    lo = min(float(np.min(np.asarray(r["val_loss"]) / r["val_loss"][0])) for r in chosen)
    hi = np.percentile([np.max(np.asarray(r["val_loss"]) / r["val_loss"][0]) for r in chosen], 90)
    pad = (hi - lo) * .05
    ax.set_ylim(lo - pad, hi + pad)

    handles = [Line2D([0], [0], color="#9a9a9a", lw=1.2, label="val_loss (held-out кривая)"),
               Line2D([0], [0], marker="*", color="gold", mec="#555", lw=0, ms=11,
                      label="argmin val_loss (oracle)")]
    for strat, _ in STRATS:
        mean_ep = np.mean(stops[strat])
        handles.append(Line2D([0], [0], marker=MRK[strat], color=COL[strat], lw=0, ms=8,
                              label=f"{LBL[strat]} — в ср. {mean_ep:.0f} эп."))
    ax.legend(handles=handles, fontsize=8.5, loc="upper right")
    tasks = sorted({r["task"] for r in chosen})
    ax.set_title(f"{len(chosen)} held-out кривых ({', '.join(tasks)}): где остановилась каждая стратегия",
                 fontweight="bold")
    ax.set_ylabel("val_loss / val_loss[0]")

    # Нижняя панель: строка на кривую, маркеры стратегий разнесены по вертикали —
    # иначе совпадающие остановки (часто на min_epochs) перекрывают друг друга.
    ns = len(STRATS)
    step = ns + 1.5                                    # высота «строки» одной кривой
    order = np.argsort(oracles)
    for row, i in enumerate(order):
        y0 = row * step
        ax2.axhspan(y0 - .7, y0 + ns - .3, color="#f2f2f2" if row % 2 else "w", zorder=0)
        ax2.plot(oracles[i], y0 + (ns - 1) / 2, "*", color="gold", mec="#555", mew=.5,
                 ms=12, zorder=3)
        for k, (strat, _) in enumerate(STRATS):
            ax2.plot(stops[strat][i], y0 + k, MRK[strat], color=COL[strat],
                     mec="w", mew=.6, ms=6.5, zorder=4)
    ax2.set_yticks([r * step + (ns - 1) / 2 for r in range(len(order))])
    ax2.set_yticklabels([chosen[i]["task"] for i in order], fontsize=8)
    ax2.invert_yaxis(); ax2.grid(axis="y", visible=False)
    ax2.set_title("Эпоха остановки по кривым (строки отсортированы по эпохе oracle)", fontsize=10)
    ax2.set_xlabel("эпоха")
    fig.tight_layout()
    fig.savefig(a.out)
    print("->", a.out)
    for strat, _ in STRATS:
        print(f"{strat:12s} stop epochs: {stops[strat]}")


if __name__ == "__main__":
    main()
