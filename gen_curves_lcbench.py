"""Конвертация LCBench (github.com/automl/LCBench) в формат curves_*.jsonl.

LCBench — 2000 конфигураций funnel-shaped MLP (Auto-PyTorch, SGD + cosine
annealing, 52 эпохи), посчитанные на 35 датасетах OpenML. В отличие от
`gen_curves.py` (свой пайплайн: MLP + Adam, 140 эпох) это чужой пайплайн —
подмешивается ТОЛЬКО как дополнительный источник, разбитый на train/held-out
по задачам, без пересечений (см. `docs/variant_c_forecaster.md`, раздел LCBench).

Скачать данные (~340 МБ zip, ~975 МБ json после распаковки):

    https://figshare.com/articles/dataset/data_2k_lw_zip/11662422
    (или: curl -L -o data_2k_lw.zip https://ndownloader.figshare.com/files/21188598)

Распаковать `data_2k_lw.json` и положить рядом с этим скриптом (или указать
путь через SRC), затем:

    python gen_curves_lcbench.py
"""
import json
import os
import random
import sys

SRC = os.environ.get("LCBENCH_JSON", "data/lcbench_data_2k_lw.json")
N_PER_TASK = 100          # столько же, сколько в gen_curves.py по умолчанию
EVAL_FRACTION = 0.2       # доля датасетов LCBench, уходящих в held-out
SEED = 0

TRAIN_OUT = "data/curves_train_lcbench.jsonl"
EVAL_OUT = "data/curves_eval_lcbench.jsonl"


def convert():
    if not os.path.exists(SRC):
        sys.exit(f"нет {SRC} — скачай и распакуй data_2k_lw.zip (см. докстринг), "
                 f"путь можно переопределить переменной LCBENCH_JSON")

    with open(SRC) as f:
        data = json.load(f)

    tasks = sorted(data.keys())
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    n_eval = max(1, round(len(tasks) * EVAL_FRACTION))
    eval_tasks = sorted(tasks[:n_eval])
    train_tasks = sorted(tasks[n_eval:])

    print(f"{len(tasks)} датасетов LCBench -> {len(train_tasks)} train / "
          f"{len(eval_tasks)} held-out")
    print("held-out:", eval_tasks)

    def write_split(task_names, out_path):
        n_curves = 0
        with open(out_path, "w") as fh:
            for name in task_names:
                configs = data[name]
                cfg_ids = list(configs.keys())
                rng.shuffle(cfg_ids)
                for cid in cfg_ids[:N_PER_TASK]:
                    entry = configs[cid]
                    log = entry["log"]
                    rec = dict(
                        task=f"lcbench_{name}",
                        model="funnel_mlp",
                        cfg=entry["config"],
                        is_classifier=True,
                        val_loss=log["Train/val_cross_entropy"],
                        val_metric=[a / 100.0 for a in log["Train/val_accuracy"]],
                    )
                    fh.write(json.dumps(rec) + "\n")
                    n_curves += 1
        print(f"-> {out_path}  ({n_curves} кривых, {len(task_names)} задач)")

    write_split(train_tasks, TRAIN_OUT)
    write_split(eval_tasks, EVAL_OUT)


if __name__ == "__main__":
    convert()
