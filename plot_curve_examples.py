"""Примеры СГЕНЕРИРОВАННЫХ кривых обучения (не реальные логи, а прогоны MLP
из gen_curves.py) — сетка train_size x label_noise на одной задаче (MNIST),
чтобы наглядно показать разнообразие корпуса data/curves_train.jsonl.

    python plot_curve_examples.py            # -> results/fig_curve_examples.png
    python plot_curve_examples.py --task sk_digits --metric loss
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

N_PER_CELL = 6  # сколько случайных кривых показать в каждой ячейке сетки


def load(task, path="data/curves_train.jsonl"):
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d["task"] == task:
                rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnist", help="какая задача из curves_train.jsonl")
    ap.add_argument("--metric", choices=["metric", "loss"], default="metric",
                    help="val_metric (accuracy/R^2) или val_loss")
    ap.add_argument("--out", default="results/fig_curve_examples.png")
    a = ap.parse_args()

    rows = load(a.task)
    if not rows:
        raise SystemExit(f"нет кривых для задачи {a.task!r} в data/curves_train.jsonl")
    is_clf = rows[0]["is_classifier"]
    key = "val_metric" if a.metric == "metric" else "val_loss"
    ylabel = ("accuracy" if is_clf else "R^2") if a.metric == "metric" else "val_loss"

    sizes = sorted({r["cfg"]["train_size"] for r in rows})
    noises = sorted({r["cfg"]["label_noise"] for r in rows})
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(len(noises), len(sizes),
                           figsize=(3.6 * len(sizes), 3.1 * len(noises)),
                           squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, nz in enumerate(noises):
        for j, ts in enumerate(sizes):
            cell = [r for r in rows if r["cfg"]["train_size"] == ts
                    and r["cfg"]["label_noise"] == nz]
            a_ = ax[i, j]
            if not cell:
                a_.axis("off")
                continue
            pick = rng.choice(len(cell), size=min(N_PER_CELL, len(cell)), replace=False)
            for n, idx in enumerate(pick):
                r = cell[idx]
                y = np.asarray(r[key])
                ep = np.arange(1, len(y) + 1)
                cfg = r["cfg"]
                lbl = f"width={cfg['width']} depth={cfg['depth']} lr={cfg['lr']:g}"
                a_.plot(ep, y, color=cmap(n / max(1, len(pick) - 1)), lw=1.4,
                       alpha=0.85, label=lbl)
            a_.set_title(f"train_size={ts}, label_noise={nz:g}", fontsize=9.5)
            a_.set_xlabel("эпоха"); a_.set_ylabel(ylabel)
            if i == 0 and j == 0:
                a_.legend(fontsize=6.5, loc="best")
    fig.suptitle(f"Сгенерированные кривые обучения", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
