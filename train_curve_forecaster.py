"""Обучение seq2seq-прогнозиста кривой val_loss.

Источник — data/curves_train.jsonl (сгенерирован gen_curves.py на задачах, которых
НЕТ в data/curves_eval.jsonl), плюс, по желанию, MNIST-кривые из data/final.csv
(они тоже вне eval-набора).

    python train_curve_forecaster.py

Сохраняет models/curve_forecaster.pt (state_dict + config).
"""
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from curve_forecaster import (CurveForecaster, HORIZON, MIN_PREFIX, MAX_LEN,
                              N_FEATURES, make_windows, collate)

SEED = 42
OUT = "models/curve_forecaster.pt"
# Основной источник — разнородные кривые из целевого пайплайна (gen_curves.py).
# final.csv (MNIST-Keras) добавляется как augmentation с понижающим весом, чтобы
# не задавить распределение: повторяем разнородные кривые FINAL_MIX раз.
FINAL_CSV_FRACTION = 0.35     # доля примеров из final.csv в обучающем миксе (0 = не брать)


def _load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            v = json.loads(line)["val_loss"]
            if len(v) > MIN_PREFIX:
                out.append(np.asarray(v, dtype=np.float64))
    return out


def _load_final_csv():
    if not os.path.exists("data/final.csv"):
        return []
    df = pd.read_csv("data/final.csv", sep=";")
    s = df["val_loss"].map(ast.literal_eval).map(
        lambda l: [x[0] if isinstance(x, list) else x for x in l])
    df = df.assign(val_loss=s)
    df = df[(df["val_loss"].map(len) > MIN_PREFIX)
            & df["shift_type"].isin(["none", "noise"])]
    return [np.asarray(v, dtype=np.float64) for v in df["val_loss"]]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(SEED)

    diverse = _load_jsonl("data/curves_train.jsonl")
    rng.shuffle(diverse)
    n_val = int(0.15 * len(diverse))
    val_curves = diverse[:n_val]                      # валидация — только разнородные кривые
    train_curves = list(diverse[n_val:])

    final_csv = _load_final_csv()
    if final_csv and FINAL_CSV_FRACTION > 0:
        rng.shuffle(final_csv)
        # столько кривых final.csv, чтобы их доля в миксе была FINAL_CSV_FRACTION
        k = int(len(train_curves) * FINAL_CSV_FRACTION / (1 - FINAL_CSV_FRACTION))
        train_curves += final_csv[:k]
        print(f"mix: {len(diverse) - n_val} diverse + {min(k, len(final_csv))} final.csv")

    rng.shuffle(train_curves)
    train_ds = [w for c in train_curves for w in make_windows(c)]
    val_ds = [w for c in val_curves for w in make_windows(c)]
    print(f"curves: {len(train_curves)} train / {len(val_curves)} val (diverse-only)")
    print(f"windows: {len(train_ds)} train / {len(val_ds)} val")

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = CurveForecaster(hidden=64, horizon=HORIZON, n_features=N_FEATURES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    # ближние шаги важнее для решения об остановке
    step_w = torch.tensor([0.9 ** i for i in range(HORIZON)], device=device)
    step_w = step_w / step_w.mean()

    def run(dl, train):
        model.train(train)
        tot, n = 0.0, 0
        per_step = torch.zeros(HORIZON, device=device)
        per_step_n = torch.zeros(HORIZON, device=device)
        for xb, lengths, yb, mb in dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            with torch.set_grad_enabled(train):
                pred = model(xb, lengths)
                err2 = (pred - yb) ** 2 * mb
                loss = (err2 * step_w).sum() / mb.mul(step_w).sum()
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            bs = xb.size(0)
            tot += loss.item() * bs
            n += bs
            per_step += (pred - yb).abs().mul(mb).sum(0)
            per_step_n += mb.sum(0)
        mae = (per_step / per_step_n.clamp(min=1)).detach().cpu().numpy()
        return tot / n, mae

    best = np.inf
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ep in range(1, 61):
        tr_loss, _ = run(train_dl, True)
        va_loss, va_mae = run(val_dl, False)
        sched.step(va_loss)
        tag = ""
        if va_loss < best:
            best = va_loss
            torch.save({"state_dict": model.state_dict(),
                        "config": {"hidden": 64, "horizon": HORIZON,
                                   "n_features": N_FEATURES,
                                   "min_prefix": MIN_PREFIX, "max_len": MAX_LEN}},
                       OUT)
            tag = "  <- saved"
        if ep % 5 == 0 or tag:
            print(f"ep {ep:3d}  train {tr_loss:.5f}  val {va_loss:.5f}  "
                  f"val MAE@1/3/10 = {va_mae[0]:.4f}/{va_mae[2]:.4f}/{va_mae[-1]:.4f}{tag}")

    print(f"\nbest val loss {best:.5f} -> {OUT}")


if __name__ == "__main__":
    main()
