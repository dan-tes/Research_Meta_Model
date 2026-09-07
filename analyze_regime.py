"""Как стратегии остановки ведут себя в зависимости от (а) количества обучающих
данных, (б) «резкости» кривой val_loss и (в) того, сколько улучшения на кривой
ещё оставалось к моменту решения — и правда ли, что GBM (вариант C) выигрывает
именно на мало-данных / нестабильных кривых, где тренд и exp-fit разбегаются.

Метод: НЕ обучаем сети заново. Берём held-out кривые (data/curves_eval.jsonl,
задачи, которых прогнозист не видел), проигрываем на каждой все стратегии
остановки теми же колбэками, что и в бенчмарке, и меряем:
    stop_ep — эпоха остановки;
    gap     — (max val_metric за всю кривую) − (val_metric на лучшем по val_loss
              чекпойнте до stop_ep)   [= oracle − quality из бенчмарка];
далее агрегируем ПО ЗАДАЧЕ (важно: wine — только train_size=300 и быстрая
сходимость; california — кривые падают ~90 эпох; их нельзя мешать).

    python analyze_regime.py   # results/regime.csv + results/fig8_regime.png + таблицы
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_early_stopping as E
from pytorch_version import SmartEarlyStoppingMultiStep as _SM
from plot_early_stopping import COL, LBL

EVAL_PATH = "data/curves_eval.jsonl"
STRATS = [s for s in ("early", "smart_trend", "smart_meta", "param") if s in E.STRATS]
DECIDE_EP = _SM().min_epochs                         # первая эпоха, где «умные» решают (10)


def curve_stats(v):
    v = np.asarray(v, float)
    p = v / max(v[0], 1e-8)
    d = np.diff(p)
    tail = d[3:]
    k = DECIDE_EP
    room = float(p[k - 1] - p[k - 1:].min()) if len(p) > k else 0.0   # сколько ещё упадёт после решения
    return dict(
        max_jump=float(max(0.0, tail.max())) if len(tail) else 0.0,
        roughness=float(np.mean(np.abs(np.diff(p[3:], 2)))) if len(p) > 6 else 0.0,
        room_after_decide=room,
    )


def simulate(strat, v):
    cb = E._callback(strat, E.penalty_for(strat))
    for ep, vl in enumerate(v):
        stop = cb.step(vl) if strat == "early" else cb.step(ep, vl)
        if stop:
            return ep + 1
    return len(v)


def build():
    rows = []
    for line in open(EVAL_PATH):
        r = json.loads(line)
        v = np.asarray(r["val_loss"], float)
        m = np.asarray(r["val_metric"], float)
        if len(v) < 15:
            continue
        oracle = float(m.max())
        st = curve_stats(v)
        for strat in STRATS:
            se = simulate(strat, v)
            best_ep = int(np.argmin(v[:se]))
            rows.append(dict(task=r["task"], train_size=r["cfg"]["train_size"],
                             lr=r["cfg"]["lr"], strat=strat, stop_ep=se,
                             gap=oracle - float(m[best_ep]), **st))
    df = pd.DataFrame(rows)
    df["_cid"] = df.index // len(STRATS)
    return df


def tables(df):
    n = df._cid.nunique()
    print(f"\n{n} held-out кривых\n")

    for t in sorted(df.task.unique()):
        sub = df[df.task == t]
        print(f"================  {t}  ({sub._cid.nunique()} кривых, "
              f"train_size {sorted(sub.train_size.unique())})  ================")
        if sub.train_size.nunique() > 1:
            print("  gap до oracle (медиана) по train_size:")
            print(sub.pivot_table("gap", "train_size", "strat", aggfunc="median")
                  [STRATS].round(4).to_string().replace("\n", "\n  "))
            print("  std(gap) по train_size  (ниже = стабильнее):")
            print(sub.pivot_table("gap", "train_size", "strat", aggfunc="std")
                  [STRATS].round(4).to_string().replace("\n", "\n  "))
        sub = sub.copy()
        sub["room_b"] = pd.qcut(sub.room_after_decide, min(4, sub.room_after_decide.nunique()),
                                labels=["мало", "средн", "много", "оч.много"][:min(4, sub.room_after_decide.nunique())],
                                duplicates="drop")
        print("  gap до oracle (медиана) по «сколько улучшения оставалось после эп.10»:")
        print(sub.pivot_table("gap", "room_b", "strat", aggfunc="median", observed=True)
              [STRATS].round(4).to_string().replace("\n", "\n  "))
        print()

    print("=== доля кривых, где стратегия ближе всех к oracle (min gap) ===")
    win = df.loc[df.groupby("_cid")["gap"].idxmin(), "strat"].value_counts(normalize=True)
    print(win.round(3).to_string())

    print("\n=== медиана gap среди ТОЛЬКО прогнозных стратегий, по «остатку улучшения» ===")
    fc = df[df.strat != "early"].copy()
    fc["room_b"] = pd.qcut(fc.room_after_decide, 4, labels=["мало", "средн", "много", "оч.много"])
    piv = fc.pivot_table("gap", "room_b", "strat", aggfunc="median", observed=True).round(4)
    print(piv[[s for s in STRATS if s != "early"]].to_string())
    print("  (это ядро гипотезы: на кривых, где ещё много падать, "
          "GBM должен ошибаться меньше тренда/exp-fit)")


def figure(df, out="results/fig8_regime.png"):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # A. gap vs train_size — california + cifar10 (у них есть разброс размеров)
    for t, a in zip(["california", "cifar10"], [ax[0, 0], ax[0, 1]]):
        sub = df[df.task == t]
        for s in STRATS:
            g = sub[sub.strat == s].groupby("train_size")["gap"]
            med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
            a.plot(med.index, med.values, "o-", color=COL[s], label=LBL[s])
            a.fill_between(med.index, q1.values, q3.values, color=COL[s], alpha=.12)
        a.set_xscale("log"); a.set_xticks(sorted(sub.train_size.unique()))
        a.set_xticklabels(sorted(sub.train_size.unique()))
        a.set_xlabel("train_size кривой"); a.set_ylabel("gap до oracle")
        a.set_title(f"{t}: потеря качества vs объём данных\n(медиана + IQR)")
        a.legend(fontsize=8)

    # C. gap vs «остаток улучшения после эп.10» — только прогнозные, все задачи
    fc = df[df.strat != "early"].copy()
    fc["room_b"] = pd.qcut(fc.room_after_decide, 4,
                           labels=["мало\n(Q1)", "средн\n(Q2)", "много\n(Q3)", "оч.много\n(Q4)"])
    order = list(fc.room_b.cat.categories)
    xs = np.arange(len(order)); strat_fc = [s for s in STRATS if s != "early"]
    w = .8 / len(strat_fc)
    for i, s in enumerate(strat_fc):
        med = fc[fc.strat == s].groupby("room_b", observed=True)["gap"].median().reindex(order)
        ax[1, 0].bar(xs + (i - (len(strat_fc) - 1) / 2) * w, med.values, w,
                     color=COL[s], label=LBL[s])
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(order, fontsize=8)
    ax[1, 0].set_xlabel("сколько val_loss ещё упадёт после эпохи решения (квартиль)")
    ax[1, 0].set_ylabel("медиана gap до oracle")
    ax[1, 0].set_title("C. Прогнозные стратегии: чем больше «недобор»,\nтем важнее качество прогноза")
    ax[1, 0].legend(fontsize=8)

    # D. доля кривых, где стратегия ближе всех к oracle (по task × размеру данных)
    df2 = df.copy()
    df2["sz"] = np.where(df2.train_size <= 800, "мало данных\n(≤800)", "много данных\n(≥2000)")
    winrows = []
    for (cid, ), g in df2.groupby(["_cid"]):
        w = g.loc[g.gap.idxmin()]
        winrows.append(dict(sz=w.sz, strat=w.strat))
    wr = pd.DataFrame(winrows)
    order = ["мало данных\n(≤800)", "много данных\n(≥2000)"]
    xs = np.arange(len(order)); w = .8 / len(STRATS)
    for i, s in enumerate(STRATS):
        frac = [((wr.sz == o) & (wr.strat == s)).sum() / max((wr.sz == o).sum(), 1) for o in order]
        ax[1, 1].bar(xs + (i - (len(STRATS) - 1) / 2) * w, frac, w, color=COL[s], label=LBL[s])
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels(order, fontsize=8)
    ax[1, 1].set_ylabel("доля кривых, где стратегия ближе всех к oracle")
    ax[1, 1].set_title("D. Кто выигрывает на отдельной кривой\n(early почти всегда — прогноз не нужен)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Held-out: стратегии остановки vs объём данных и «недобор» кривой",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out); plt.close(fig)
    print("\n->", out)


if __name__ == "__main__":
    df = build()
    df.to_csv("results/regime.csv", index=False)
    tables(df)
    figure(df)
