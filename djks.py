"""
Исправленный тест для PyTorch Early Stopping.

Модели и стратегии остановки берутся из единого модуля ``pytorch_version``,
чтобы не поддерживать три расходящиеся копии одного и того же кода.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import warnings

from pytorch_version import (
    EPOCH_PENALTY,
    MLPForMNIST,
    CNNForCIFAR,
    SimpleEarlyStopping,
    SmartEarlyStoppingMultiStep,
    ParametricEarlyStopping,
)

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ========================
# ОБУЧЕНИЕ
# ========================

def run_training_pytorch(X_train, y_train, X_test, y_test, train_size,
                         callback_name, model_builder, is_classifier=True,
                         batch_size=16, epochs=200):
    """Одно обучение с выбранным callback"""

    # Преобразование в тензоры
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long if is_classifier else torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long if is_classifier else torch.float32)

    # Преобразовать на GPU если нужно
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Dataset
    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    test_dataset = TensorDataset(X_test, y_test)

    # Split на train/val
    train_size_split = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size_split
    train_dataset, val_dataset = random_split(train_dataset, [train_size_split, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Модель
    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Выбор callback
    if callback_name == "early":
        callback = SimpleEarlyStopping(patience=5)
    elif callback_name == "smart":
        callback = SmartEarlyStoppingMultiStep(train_size=train_size, epoch_penalty=EPOCH_PENALTY)
    else:  # parametric
        callback = ParametricEarlyStopping(epoch_penalty=EPOCH_PENALTY)

    best_val_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_trained = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        epochs_trained = epoch + 1

        # Callback
        if callback_name == "early":
            should_stop = callback.step(val_loss)
        else:
            should_stop = callback.step(epoch, val_loss)

        if should_stop:
            break

    # Возвращаем лучшие веса
    model.load_state_dict(best_state)

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    test_acc = correct / total

    return {
        "epochs": epochs_trained,
        "test_acc": test_acc,
        "best_val_loss": best_val_loss
    }


def runner_pytorch(sizes, X_train, y_train, X_test, y_test,
                   model_builder, is_classifier=True, batch_size=16, epochs=200):
    """Полный бенчмарк"""
    results = []
    runs = 4
    total = len(sizes) * runs
    count = 0

    for j, size in enumerate(sizes):
        for r in range(runs):
            count += 1
            print(f"\n[{count}/{total}] Train size: {size}, Run: {r + 1}")

            # Запуск с разными callbacks
            res_early = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "early", model_builder, is_classifier, batch_size, epochs
            )

            res_smart = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "smart", model_builder, is_classifier, batch_size, epochs
            )

            res_param = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "parametric", model_builder, is_classifier, batch_size, epochs
            )

            results.append({
                "train_size": size,
                "run": r,

                "early_epochs": res_early["epochs"],
                "early_acc": res_early["test_acc"],
                "early_loss": res_early["best_val_loss"],

                "smart_epochs": res_smart["epochs"],
                "smart_acc": res_smart["test_acc"],
                "smart_loss": res_smart["best_val_loss"],

                "param_epochs": res_param["epochs"],
                "param_acc": res_param["test_acc"],
                "param_loss": res_param["best_val_loss"],
            })

    df = pd.DataFrame(results)
    df['diff_epochs'] = df['early_epochs'] - df['smart_epochs']
    df['diff_loss'] = df['early_loss'] - df['smart_loss']
    df['diff_acc'] = df['early_acc'] - df['smart_acc']

    return df


def analyze_results(df_curr, sizes, acc_per_epoch=0.002, title=""):
    """Анализ результатов.

    ``acc_per_epoch`` — ценность одной сэкономленной эпохи в единицах accuracy.
    Больше значение = сильнее штраф за лишние эпохи.
    """

    df_curr = df_curr.copy()

    # diff_epochs = early - smart  (>0, если Smart быстрее)
    # diff_acc    = early - smart  (>0, если Early точнее)
    df_curr["epochs_saved"] = df_curr["diff_epochs"]
    df_curr["acc_gain"] = -df_curr["diff_acc"]
    df_curr["rel_epochs_saved"] = df_curr["epochs_saved"] / (df_curr["early_epochs"] + 1e-8)
    df_curr["utility"] = df_curr["acc_gain"] + acc_per_epoch * df_curr["epochs_saved"]
    df_curr["smart_better"] = df_curr["utility"] > 0

    # Summary
    summary = df_curr.groupby("train_size").agg({
        "early_epochs": "mean",
        "smart_epochs": "mean",
        "param_epochs": "mean",

        "early_acc": "mean",
        "smart_acc": "mean",
        "param_acc": "mean",

        "early_loss": "mean",
        "smart_loss": "mean",
        "param_loss": "mean",

        "epochs_saved": "mean",
        "acc_gain": "mean",
        "utility": "mean",
        "smart_better": "mean"
    })

    print(f"\n{'=' * 60}")
    print(f"РЕЗУЛЬТАТЫ: {title}")
    print(f"{'=' * 60}")
    print(summary)

    # === ГРАФИК 1: Loss comparison ===
    plt.figure(figsize=(10, 6))
    plt.plot(summary.index, summary["early_loss"], marker="o", label="EarlyStopping", linewidth=2)
    plt.plot(summary.index, summary["smart_loss"], marker="s", label="SmartStop", linewidth=2)
    plt.plot(summary.index, summary["param_loss"], marker="^", label="ParamStop", linewidth=2)
    plt.xlabel("Train Size", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title(f"Loss Comparison - {title}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss_comparison_{title.replace(' ', '_')}.png", dpi=100)
    plt.show()

    # === ГРАФИК 2: Epochs comparison ===
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    width = 0.25

    ax.bar(x - width, summary["early_epochs"], width, label="EarlyStopping")
    ax.bar(x, summary["smart_epochs"], width, label="SmartStop")
    ax.bar(x + width, summary["param_epochs"], width, label="ParamStop")

    ax.set_xlabel("Train Size", fontsize=12)
    ax.set_ylabel("Mean Epochs Trained", fontsize=12)
    ax.set_title(f"Training Duration - {title}", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"epochs_comparison_{title.replace(' ', '_')}.png", dpi=100)
    plt.show()

    # === ГРАФИК 3: Trade-off ===
    plt.figure(figsize=(10, 6))

    for size in sizes:
        subset = df_curr[df_curr["train_size"] == size]
        plt.scatter(subset["diff_epochs"], subset["diff_acc"],
                    s=100, alpha=0.7, label=f"Size={size}")

    x_vals = np.linspace(df_curr["diff_epochs"].min(), df_curr["diff_epochs"].max(), 100)
    y_vals = acc_per_epoch * x_vals
    plt.plot(x_vals, y_vals, linestyle="--", color="red", linewidth=2, label="Utility=0")

    plt.axhline(0, linestyle="-", color="black", linewidth=0.8)
    plt.axvline(0, linestyle="-", color="black", linewidth=0.8)

    plt.xlabel("Epochs Saved (Early - Smart)", fontsize=12)
    plt.ylabel("Accuracy Improvement (Early - Smart)", fontsize=12)
    plt.title(f"Speed-Accuracy Trade-off - {title}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"tradeoff_{title.replace(' ', '_')}.png", dpi=100)
    plt.show()

    # === ГРАФИК 4: Win rate ===
    plt.figure(figsize=(10, 6))
    plt.plot(summary.index, summary["smart_better"], marker="o",
             linewidth=2, markersize=10, color="green")
    plt.fill_between(summary.index, summary["smart_better"], alpha=0.3, color="green")

    plt.xlabel("Train Size", fontsize=12)
    plt.ylabel("Win Rate (SmartStop better)", fontsize=12)
    plt.title(f"SmartEarlyStopping Win Rate - {title}", fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"winrate_{title.replace(' ', '_')}.png", dpi=100)
    plt.show()

    return summary, df_curr


# ========================
# MAIN
# ========================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyTorch Early Stopping Benchmark (FIXED)")
    print("=" * 60)

    # Загрузка MNIST
    print("\n[1] Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train_mnist = mnist_train.data.float() / 255.0
    y_train_mnist = mnist_train.targets
    X_test_mnist = mnist_test.data.float() / 255.0
    y_test_mnist = mnist_test.targets

    # Загрузка CIFAR-10
    print("[2] Loading CIFAR-10...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    X_train_cifar = torch.stack([x[0] for x in cifar_train])
    y_train_cifar = torch.tensor([x[1] for x in cifar_train])
    X_test_cifar = torch.stack([x[0] for x in cifar_test])
    y_test_cifar = torch.tensor([x[1] for x in cifar_test])

    sizes = [50, 500, 1000, 2000, 5000]

    # ===== MNIST BENCHMARK =====
    print("\n" + "=" * 60)
    print("MNIST Benchmark")
    print("=" * 60)

    df_mnist = runner_pytorch(
        sizes, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist,
        MLPForMNIST, is_classifier=True, batch_size=16, epochs=200
    )

    summary_mnist, df_mnist = analyze_results(df_mnist, sizes, title="MNIST")

    # ===== CIFAR-10 BENCHMARK =====
    print("\n" + "=" * 60)
    print("CIFAR-10 Benchmark")
    print("=" * 60)

    df_cifar = runner_pytorch(
        sizes, X_train_cifar, y_train_cifar, X_test_cifar, y_test_cifar,
        CNNForCIFAR, is_classifier=True, batch_size=16, epochs=200
    )

    summary_cifar, df_cifar = analyze_results(df_cifar, sizes, title="CIFAR-10")

    # ===== COMBINED ANALYSIS =====
    print("\n" + "=" * 60)
    print("Combined Analysis")
    print("=" * 60)

    df_mnist["dataset"] = "MNIST"
    df_cifar["dataset"] = "CIFAR-10"
    df_all = pd.concat([df_mnist, df_cifar], ignore_index=True)

    summary_all, df_all = analyze_results(df_all, sizes, title="All_Datasets")

    # Сохранение результатов
    print("\n[Saving results...]")
    df_all.to_csv("early_stopping_results.csv", index=False)
    summary_all.to_csv("early_stopping_summary.csv")

    print("\n✅ Benchmark complete!")
    print(f"Results saved to early_stopping_results.csv")
    print(f"Summary saved to early_stopping_summary.csv")