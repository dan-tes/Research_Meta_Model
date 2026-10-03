"""Кадры для видео: по кадру на эпоху, вертикальный курсор идёт слева направо,
за ним дорисовываются кривые val_loss нескольких held-out моделей.

Фазы идут подряд:
  1. без остановки — все кривые до конца;
  2..n. стратегии ранней остановки (симуляция как в sim_stopping_lcbench.py),
     последней — GBM (smart_meta). Полная кривая остаётся бледным «призраком»,
     цветная линия обрывается в точке остановки.

Внизу кадра копится сводка по уже показанным фазам (эпохи и потеря качества).

    python plot_video_frames.py                     # 6 кривых, кадры в results/video_frames/
    python plot_video_frames.py --n 6 --stride 2 --hold 15
    # склейка: ffmpeg -framerate 30 -i results/video_frames/%05d.png -pix_fmt yuv420p out.mp4
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_early_stopping import COL, LBL
from plot_stopping_many import load_curves
from sim_stopping_lcbench import simulate

plt.rcParams.update({"figure.dpi": 100, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# (ключ, путь к GBM) — порядок фаз; "full" = без остановки
PHASES = [("full", None), ("early", None), ("smart_trend", None), ("param", None),
          ("smart_meta", "models/meta_forecaster.pkl")]
COL_FULL = "#222222"
LBL_FULL = "Без остановки"


def pick(rows, n, seed):
    """Поровну кривых с каждой задачи — чтобы на видео были разные модели."""
    rng = np.random.default_rng(seed)
    tasks = sorted({r["task"] for r in rows})
    out = []
    for k in range(n):
        pool = [r for r in rows if r["task"] == tasks[k % len(tasks)] and r not in out]
        out.append(pool[rng.integers(len(pool))])
    return out


def describe(rec):
    c = rec.get("cfg", {})
    bits = [f"{k}={c[k]}" for k in ("width", "depth", "lr", "dropout") if k in c]
    return f"{rec['task']} / {rec['model']}\n" + ", ".join(bits)


def ylim_for(vl):
    lo, hi = float(vl.min()), float(np.percentile(vl, 97))
    hi = max(hi, float(vl[:3].max()))           # стартовый участок всегда в кадре
    pad = (hi - lo) * .08 or .05
    return lo - pad, hi + pad


def draw_frame(path, curves, phase, ep, T, stops, summary):
    key, _ = phase
    color = COL_FULL if key == "full" else COL[key]
    title = LBL_FULL if key == "full" else LBL[key]
    n = len(curves)
    ncol = 3 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow + 1.4), squeeze=False)

    spent = 0
    for i, (ax, rec) in enumerate(zip(axes.flat, curves)):
        vl = np.asarray(rec["val_loss"], dtype=np.float64)
        x = np.arange(1, len(vl) + 1)
        stop = stops[i]
        upto = min(ep, stop)
        spent += upto
        if key != "full":                       # призрак полной кривой
            ax.plot(x, vl, color="#bdbdbd", lw=1, alpha=.7, zorder=1)
        ax.plot(x[:upto], vl[:upto], color=color, lw=1.8, zorder=3)
        if ep < stop:
            ax.axvline(ep, color="#888", lw=1, ls="--", zorder=2)
            ax.plot(ep, vl[ep - 1], "o", color=color, ms=5, zorder=4)
        else:
            best = float(vl[:stop].min())
            gap = (best - vl.min()) / abs(vl.min()) * 100 if vl.min() else 0.0
            if key != "full":
                ax.axvline(stop, color=color, lw=1.6, alpha=.8, zorder=2)
                ax.plot(stop, vl[stop - 1], "X", color=color, mec="w", ms=10, zorder=5)
            ax.text(.98, .95, f"стоп: {stop} эп.\nхуже лучшего: {gap:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(fc="w", ec=color, alpha=.9))
        o = int(vl.argmin())
        if key != "full" or ep > o:
            ax.plot(o + 1, vl[o], "*", color="gold", mec="#555", mew=.5, ms=12, zorder=6)
        ax.set_xlim(0, len(vl) + 1)
        ax.set_ylim(*ylim_for(vl))
        ax.set_title(describe(rec), fontsize=9)
        if i % ncol == 0:
            ax.set_ylabel("val_loss")
        if i >= n - ncol:
            ax.set_xlabel("эпоха")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    fig.suptitle(f"{title}   ·   эпоха {min(ep, T)}/{T}   ·   потрачено эпох: {spent}",
                 fontsize=14, fontweight="bold", color=color)
    lines = [f"{lbl}: {e} эп. ({e / summary[0][1] * 100:.0f}%), хуже лучшего в ср. {g:.1f}%"
             for lbl, e, g in summary]
    fig.text(.01, .01, "\n".join(lines), fontsize=9.5, family="monospace", va="bottom")
    fig.tight_layout(rect=(0, .02 + .022 * max(len(lines), 1), 1, .95))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["data/curves_eval.jsonl"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="эпох на кадр")
    ap.add_argument("--hold", type=int, default=20, help="кадров-паузы в конце фазы")
    ap.add_argument("--out", default="results/video_frames")
    a = ap.parse_args()

    rows = load_curves(a.paths, a.task)
    if not rows:
        raise SystemExit("нет кривых (проверь --paths / --task)")
    curves = pick(rows, a.n, a.seed)
    T = max(len(r["val_loss"]) for r in curves)

    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    frame, summary = 0, []
    for phase in PHASES:
        key, meta_path = phase
        if key == "full":
            stops = [len(r["val_loss"]) for r in curves]
        else:
            stops = [simulate(r["val_loss"], r["val_metric"], key, meta_path)[0] for r in curves]
        # фаза закончится, когда остановилась последняя кривая
        end = max(stops)
        eps = list(range(1, end + 1, a.stride))
        if eps[-1] != end:
            eps.append(end)
        gaps = []
        for r, s in zip(curves, stops):
            vl = np.asarray(r["val_loss"])
            gaps.append((vl[:s].min() - vl.min()) / abs(vl.min()) * 100 if vl.min() else 0.0)
        done = summary + [(LBL_FULL if key == "full" else LBL[key], sum(stops), float(np.mean(gaps)))]
        for ep in eps:
            draw_frame(f"{a.out}/{frame:05d}.png", curves, phase, ep, T, stops,
                       summary if ep < end else done)
            frame += 1
        for _ in range(a.hold - 1):            # пауза — копии последнего кадра
            shutil.copy(f"{a.out}/{frame - 1:05d}.png", f"{a.out}/{frame:05d}.png")
            frame += 1
        summary = done
        print(f"{key:12s} stops={stops}  -> кадров всего {frame}")
    print("->", a.out)


if __name__ == "__main__":
    main()
