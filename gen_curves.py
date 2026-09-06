"""Генерация датасета кривых обучения для seq2seq-прогнозиста SmartStop.

Пишет два непересекающихся по задачам файла:
    data/curves_train.jsonl  — источник для обучения прогнозиста
    data/curves_eval.jsonl   — held-out задачи, только для финального бенчмарка

Каждая строка: {task, model, cfg, is_classifier, val_loss:[...], val_metric:[...]}
val_metric = accuracy (классификация) или R^2 (регрессия) по эпохам.

    python gen_curves.py
"""
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import (load_digits, load_breast_cancer, load_diabetes,
                              fetch_california_housing, make_regression)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED0 = 100
torch.manual_seed(SEED0)
np.random.seed(SEED0)


# ----------------------------------------------------------------- загрузчики
def _img(ds_cls, root="./data"):
    tr = ds_cls(root=root, train=True, download=True)
    te = ds_cls(root=root, train=False, download=True)
    Xtr = tr.data.numpy() if torch.is_tensor(tr.data) else np.asarray(tr.data)
    Xte = te.data.numpy() if torch.is_tensor(te.data) else np.asarray(te.data)
    ytr = np.asarray(tr.targets); yte = np.asarray(te.targets)
    Xtr = Xtr.reshape(len(Xtr), -1).astype("float32") / 255.0
    Xte = Xte.reshape(len(Xte), -1).astype("float32") / 255.0
    return Xtr, ytr.astype("int64"), Xte, yte.astype("int64"), True, int(ytr.max() + 1)


def _sk_clf(load):
    d = load(); X = StandardScaler().fit_transform(d.data).astype("float32")
    y = d.target.astype("int64")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    return Xtr, ytr, Xte, yte, True, int(y.max() + 1)


def _sk_reg(X, y):
    X = StandardScaler().fit_transform(X).astype("float32")
    y = ((y - y.mean()) / y.std()).astype("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    return Xtr, ytr, Xte, yte, False, 1


def load_california():
    d = fetch_california_housing(); return _sk_reg(d.data, d.target)


def load_diabetes_reg():
    d = load_diabetes(); return _sk_reg(d.data, d.target)


def load_synth_reg(n_features=20, noise=15.0):
    X, y = make_regression(n_samples=6000, n_features=n_features, n_informative=10,
                           noise=noise, random_state=0)
    return _sk_reg(X, y)


def load_student_reg():
    import pandas as pd
    df = pd.read_csv("data/student_mental_health_burnout.csv")
    num = df.select_dtypes("number").drop(columns=["student_id"], errors="ignore")
    y = num.pop("cgpa").values
    return _sk_reg(num.values, y)


def load_wine_reg():
    import pandas as pd
    df = pd.read_csv("data/WineQT.csv")
    cols = ("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free "
            "sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol").split(",")
    return _sk_reg(df[cols].values, df["quality"].values.astype("float64"))


TRAIN_TASKS = {
    "mnist":        lambda: _img(datasets.MNIST),
    "fashion":      lambda: _img(datasets.FashionMNIST),
    "sk_digits":    lambda: _sk_clf(load_digits),
    "sk_bcancer":   lambda: _sk_clf(load_breast_cancer),
    "diabetes":     load_diabetes_reg,
    "synth_reg_a":  lambda: load_synth_reg(20, 12.0),
    "synth_reg_b":  lambda: load_synth_reg(40, 30.0),
    "student_cgpa": load_student_reg,
}
# held-out задачи — только те, что не требуют новых загрузок (CIFAR-10 уже скачан,
# california кэшируется sklearn, wine — локальный CSV). KMNIST/CIFAR-100 выпали
# из-за нерабочих/медленных зеркал.
EVAL_TASKS = {
    "cifar10":      lambda: _img(datasets.CIFAR10),
    "california":   load_california,
    "wine":         load_wine_reg,
}


# ----------------------------------------------------------------- модель
def build_mlp(in_dim, out_dim, width, depth, dropout):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def run_curve(data, cfg, max_epochs, bs=64):
    Xtr, ytr, Xte, yte, is_clf, out_dim = data
    g = torch.Generator().manual_seed(cfg["seed"])
    idx = torch.randperm(len(Xtr), generator=g)[:cfg["train_size"]]
    Xt = torch.tensor(Xtr[idx.numpy()]);  yt = torch.tensor(ytr[idx.numpy()])
    Xv = torch.tensor(Xte);               yv = torch.tensor(yte)

    if is_clf and cfg.get("label_noise", 0) > 0:
        m = torch.rand(len(yt), generator=g) < cfg["label_noise"]
        yt = yt.clone()
        yt[m] = torch.randint(0, out_dim, (int(m.sum()),), generator=g)

    Xt, yt, Xv, yv = Xt.to(DEVICE), yt.to(DEVICE), Xv.to(DEVICE), yv.to(DEVICE)
    torch.manual_seed(cfg["seed"])
    model = build_mlp(Xt.shape[1], out_dim, cfg["width"], cfg["depth"],
                      cfg["dropout"]).to(DEVICE)
    crit = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    vl_curve, vm_curve = [], []
    n = len(Xt)
    for _ in range(max_epochs):
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
                ss_res = ((yv - p) ** 2).sum().item()
                ss_tot = ((yv - yv.mean()) ** 2).sum().item() + 1e-12
                vm = 1.0 - ss_res / ss_tot
        vl_curve.append(vl); vm_curve.append(vm)
    return vl_curve, vm_curve


# сколько кривых на задачу (можно переопределить: N_PER_TASK=120 python gen_curves.py).
# Вариант C (табличный GBM-прогнозист) хочет 500-2000 кривых из этого же пайплайна,
# поэтому дефолт поднят с исходных 34: 8 train-задач * 100 = 800 обучающих кривых.
N_PER_TASK = int(os.environ.get("N_PER_TASK", "100"))


# ----------------------------------------------------------------- перебор
def configs_for(is_clf, n_samples, n_curves=None):
    n_curves = n_curves or N_PER_TASK
    widths = [(16, 1), (64, 2), (128, 2), (256, 3)]
    lrs = [5e-3, 3e-3, 1e-3, 5e-4, 3e-4]
    wds = [0.0, 1e-3]
    drops = [0.0, 0.3]
    sizes = [s for s in [300, 800, 2000, 6000] if s <= int(0.9 * n_samples)] or [int(0.7 * n_samples)]
    noises = [0.0, 0.15] if is_clf else [0.0]
    combos = list(itertools.product(widths, lrs, wds, drops, sizes, noises))
    rng = np.random.default_rng(0)
    rng.shuffle(combos)
    # если комбинаций меньше, чем нужно кривых, добираем повторами с новым seed
    reps = -(-n_curves // len(combos))
    combos = (combos * reps)[:n_curves]
    out = []
    for (w, dep), lr, wd, dr, ts, nz in combos:
        out.append(dict(width=w, depth=dep, lr=lr, wd=wd, dropout=dr,
                        train_size=ts, label_noise=nz, seed=SEED0 + len(out)))
    return out


def generate(tasks, path, max_epochs=140):
    t0 = time.time()
    with open(path, "w") as fh:
        for tname, loader in tasks.items():
            try:
                data = loader()
            except Exception as exc:
                print(f"  {tname:14s} ПРОПУЩЕНА ({exc})", flush=True)
                continue
            is_clf, n = data[4], len(data[0])
            cfgs = configs_for(is_clf, n)
            for j, cfg in enumerate(cfgs):
                vl, vm = run_curve(data, cfg, max_epochs)
                fh.write(json.dumps(dict(task=tname, model="mlp", cfg=cfg,
                                         is_classifier=bool(is_clf),
                                         val_loss=vl, val_metric=vm)) + "\n")
            print(f"  {tname:14s} {len(cfgs)} кривых  ({time.time()-t0:.0f}s)", flush=True)
    print(f"-> {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    print("TRAIN split:")
    generate(TRAIN_TASKS, "data/curves_train.jsonl")
    print("EVAL split:")
    generate(EVAL_TASKS, "data/curves_eval.jsonl")
    print("DONE")
