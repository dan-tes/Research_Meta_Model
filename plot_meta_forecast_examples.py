"""Сравнение GBM-метапрогнозиста (meta_forecaster.py) с фактом на held-out
кривых: что модель предсказала в точке p про минимум val_loss за следующие
HORIZON эпох — и чем это обернулось на самом деле.

Два режима:
  --mode lines   (по умолчанию) для каждого p (по всей длине кривой) две
                 линии — факт. минимум за [p, p+H] и прогноз GBM; сырая
                 кривая — тонким фоном. Проще всего читать как "факт vs
                 прогноз".
  --mode overlay разрозненные якоря (несколько p) поверх сырой кривой:
                 кружок = val_loss в точке p, пунктир = прогноз, ромб = факт.

    python plot_meta_forecast_examples.py                  # случайные held-out кривые
    python plot_meta_forecast_examples.py --task cifar10 --n 4
    python plot_meta_forecast_examples.py --mode overlay --task cifar10
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_forecaster as MF

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

EVAL_PATHS = ["data/curves_eval.jsonl", "data/curves_eval_lcbench.jsonl"]
N_ANCHORS = 4  # сколько точек p показать на каждой кривой


def load_curves(paths, task=None):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                rec = json.loads(line)
                if task is None or rec["task"] == task:
                    rows.append(rec)
    return rows


def anchors_for(k_max, n_anchors=N_ANCHORS):
    lo, hi = MF.PREFIX_MIN, k_max - MF.HORIZON
    if hi <= lo:
        return [lo]
    return sorted(set(np.linspace(lo, hi, n_anchors, dtype=int).tolist()))


def plot_curve(ax, rec, model_path):
    v = np.asarray(rec["val_loss"], dtype=np.float64)
    v0 = max(float(v[0]), 1e-8)
    p = v / v0
    ep = np.arange(1, len(p) + 1)
    ax.plot(ep, p, color="#333", lw=1.6, zorder=2, label="val_loss (факт, весь прогон)")

    cmap = plt.get_cmap("plasma")
    anchors = anchors_for(len(p))
    for i, k in enumerate(anchors):
        relgain = MF.predict_rel_gain(v[:k], path=model_path)
        cur = float(p[k - 1])                    # факт. значение кривой в точке p (не m0!)
        m0 = float(p[:k].min())                   # лучший-до-сих-пор — только для формулы relgain
        floor_pred = m0 * (1.0 - relgain)
        window = p[k:k + MF.HORIZON]
        if len(window):
            w_idx = int(window.argmin())
            floor_actual = float(window[w_idx])
            floor_ep = k + w_idx + 1               # истинная эпоха факт. минимума в окне
        else:
            floor_actual, floor_ep = cur, k
        color = cmap(i / max(1, len(anchors) - 1))
        ax.plot(k, cur, "o", color=color, ms=7, zorder=4)
        # тонкий вертикальный коннектор p -> прогноз (не подразумевает траекторию,
        # только связывает точку прогноза с горизонтальным "полом" ниже)
        ax.plot([k, k], [cur, floor_pred], ":", color=color, lw=1, alpha=.6, zorder=3)
        ax.plot([k, k + MF.HORIZON], [floor_pred, floor_pred], "--", color=color, lw=2, zorder=3)
        ax.plot(floor_ep, floor_actual, "D", mfc="none", mec=color, mew=2, ms=8, zorder=4)
        ax.annotate(f"p={k}\nrelgain={relgain:.2f}", (k, cur), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, color=color, ha="center")
    ax.set_xlabel("эпоха"); ax.set_ylabel("val_loss / val_loss[0]")
    ax.set_title(rec["task"], fontsize=10)


def plot_curve_lines(ax, rec, model_path):
    """Непрерывное сравнение по всем p: факт. минимум за [p, p+H] против того,
    что в этой же точке предсказал GBM — как две линии по p, а не разрозненные
    якоря поверх сырой кривой (проще читать «факт vs прогноз»)."""
    v = np.asarray(rec["val_loss"], dtype=np.float64)
    v0 = max(float(v[0]), 1e-8)
    p = v / v0
    n = len(p)
    ks = np.arange(MF.PREFIX_MIN, n - MF.HORIZON + 1)
    pred, actual = np.empty(len(ks)), np.empty(len(ks))
    for i, k in enumerate(ks):
        relgain = MF.predict_rel_gain(v[:k], path=model_path)
        m0 = float(p[:k].min())
        pred[i] = m0 * (1.0 - relgain)
        actual[i] = float(p[k:k + MF.HORIZON].min())

    ax.plot(np.arange(1, n + 1), p, color="#bbb", lw=1.2, zorder=1,
            label="val_loss (факт, весь прогон, фон)")
    ax.plot(ks, actual, color="#222", lw=2, zorder=3, label=f"факт: минимум за [p, p+{MF.HORIZON}]")
    ax.plot(ks, pred, "--", color="#d62728", lw=2, zorder=3, label="прогноз GBM: ожидаемый минимум")

    ax.set_xlabel("эпоха p"); ax.set_ylabel("val_loss / val_loss[0]")
    ax.set_title(rec["task"], fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="фильтр по задаче из curves_eval*.jsonl")
    ap.add_argument("--n", type=int, default=4, help="сколько кривых показать")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MF.MODEL_PATH)
    ap.add_argument("--mode", choices=["lines", "overlay"], default="lines",
                    help="lines: факт vs прогноз непрерывно по p; "
                         "overlay: разрозненные якоря поверх сырой кривой")
    ap.add_argument("--out", default="results/fig_meta_forecast_examples.png")
    a = ap.parse_args()

    rows = load_curves(EVAL_PATHS, a.task)
    rows = [r for r in rows if len(r["val_loss"]) >= MF.PREFIX_MIN + MF.HORIZON + 3]
    if not rows:
        raise SystemExit("нет подходящих held-out кривых (проверь --task)")
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)
    chosen = [rows[i] for i in pick]

    fig, ax = plt.subplots(len(chosen), 1, figsize=(9, 3.4 * len(chosen)), squeeze=False)
    ax = ax[:, 0]

    if a.mode == "lines":
        for a_, rec in zip(ax, chosen):
            plot_curve_lines(a_, rec, a.model)
        ax[0].legend(fontsize=7.5, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.94 if len(chosen) > 1 else 0.88])
    else:
        for a_, rec in zip(ax, chosen):
            plot_curve(a_, rec, a.model)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="#333", lw=1.6, label="val_loss (факт, весь прогон)"),
            Line2D([0], [0], marker="o", color="gray", lw=0, label="p: val_loss в точке прогноза"),
            Line2D([0], [0], color="gray", ls="--", lw=2,
                   label=f"прогноз GBM: ожидаемый минимум за [p, p+{MF.HORIZON}] (не траектория!)"),
            Line2D([0], [0], marker="D", mfc="none", mec="gray", mew=2, lw=0, ms=8,
                   label=f"факт: где и чему равен минимум за [p, p+{MF.HORIZON}]"),
        ]
        fig.suptitle("Прогноз GBM-метамодели «сколько ещё улучшения доступно» vs факт (held-out)",
                    fontweight="bold", y=0.995)
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.5,
                  bbox_to_anchor=(0.5, 0.965))
        fig.tight_layout(rect=[0, 0, 1, 0.90 if len(chosen) > 1 else 0.8])
    fig.savefig(a.out, bbox_inches="tight")
    print("->", a.out)


if __name__ == "__main__":
    main()
