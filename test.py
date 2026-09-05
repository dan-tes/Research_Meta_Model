"""
PyTorch Early Stopping Benchmarks
Сравнение EarlyStopping vs SmartEarlyStopping vs ParametricEarlyStopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import warnings

from pytorch_version import EPOCH_PENALTY

warnings.filterwarnings('ignore')

# Импортируем классы из основного файла
# from pytorch_version import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ========================
# ПОЛНЫЙ RUNNER
# ========================

def runner_pytorch(sizes, X_train, y_train, X_test, y_test,
                   model_builder, is_classifier=False, batch_size=16, epochs=200,
                   model_meta=None, runs=10):
    """
    Полный бенчмарк с тремя стратегиями Early Stopping

    Args:
        sizes: список размеров выборок
        X_train, y_train: обучающие данные
        X_test, y_test: тестовые данные
        model_builder: функция для создания модели
        is_classifier: True для классификации, False для регрессии
        batch_size: размер батча
        epochs: максимальное число эпох
        runs: сколько повторов на каждый размер выборки
    """

    # не запрашиваем больше примеров, чем реально есть в обучающей выборке
    n_available = len(X_train)
    sizes = [s for s in sizes if s <= n_available] or [n_available]

    results = []
    total = len(sizes) * runs
    count = 0

    for j, size in enumerate(sizes):
        for r in range(runs):
            count += 1
            print(f"\n[{count}/{total}] Train size: {size}, Run: {r + 1}")

            # Запуск с разными callbacks
            res_early = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "early", model_builder, is_classifier, batch_size, epochs, model_meta=model_meta
            )

            res_smart = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "smart", model_builder, is_classifier, batch_size, epochs, model_meta=model_meta
            )

            res_param = run_training_pytorch(
                X_train, y_train, X_test, y_test, size,
                "parametric", model_builder, is_classifier, batch_size, epochs, model_meta=model_meta
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


def run_training_pytorch(X_train, y_train, X_test, y_test, train_size,
                         callback_name, model_builder, is_classifier=False,
                         batch_size=16, epochs=200, model_meta=None):
    """Одно обучение с выбранным callback"""

    # Преобразование в тензоры если нужно
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long if is_classifier else torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long if is_classifier else torch.float32)

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

    if is_classifier:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Callbacks
    if callback_name == "early":
        from pytorch_version import SimpleEarlyStopping
        callback = SimpleEarlyStopping(patience=5)
    elif callback_name == "smart":
        from pytorch_version import SmartEarlyStoppingMultiStep
        callback = SmartEarlyStoppingMultiStep(model_meta, train_size, epoch_penalty=EPOCH_PENALTY)
    else:  # parametric
        from pytorch_version import ParametricEarlyStopping
        callback = ParametricEarlyStopping(epoch_penalty=EPOCH_PENALTY)

    best_val_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_trained = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)

            if is_classifier:
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(-1), batch_y)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                if is_classifier:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(-1), batch_y)

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

    # Возвращаем лучшие веса: ранняя остановка не должна ухудшать итоговое качество
    model.load_state_dict(best_state)

    # Test
    model.eval()
    if is_classifier:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        test_acc = correct / total
    else:
        # Для регрессии в качестве "acc" используем R^2 на тесте
        preds, targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                out = model(batch_X).squeeze(-1).cpu().numpy()
                preds.append(np.atleast_1d(out))
                targets.append(np.atleast_1d(batch_y.numpy()))
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        ss_res = float(np.sum((targets - preds) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2)) + 1e-12
        test_acc = 1.0 - ss_res / ss_tot

    return {
        "epochs": epochs_trained,
        "test_acc": test_acc,
        "best_val_loss": best_val_loss
    }


# ========================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ========================

def analyze_results(df_curr, sizes, acc_per_epoch=0.002, title=""):
    """Анализ и визуализация результатов.

    ``acc_per_epoch`` — во сколько единиц accuracy оценивается одна сэкономленная
    эпоха. Чем больше — тем сильнее штраф за лишние эпохи в итоговой оценке
    ``utility``; SmartStop считается лучше, когда прирост accuracy плюс
    экономия эпох (в пересчёте на accuracy) положительны.
    """

    df_curr = df_curr.copy()

    # diff_epochs = early_epochs - smart_epochs  (>0, если Smart быстрее)
    # diff_acc    = early_acc - smart_acc        (>0, если Early точнее)
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

    # ===== ГРАФИК 1: Loss comparison =====
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
    plt.show()

    # ===== ГРАФИК 2: Epochs comparison =====
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
    plt.show()

    # ===== ГРАФИК 3: Trade-off (epochs saved vs accuracy) =====
    plt.figure(figsize=(10, 6))

    for size in sizes:
        subset = df_curr[df_curr["train_size"] == size]
        plt.scatter(subset["diff_epochs"], subset["diff_acc"],
                    s=100, alpha=0.7, label=f"Size={size}")

    # Decision boundary: utility=0  <=>  diff_acc = acc_per_epoch * diff_epochs
    x_vals = np.linspace(df_curr["diff_epochs"].min(), df_curr["diff_epochs"].max(), 100)
    y_vals = acc_per_epoch * x_vals
    plt.plot(x_vals, y_vals, linestyle="--", color="red", linewidth=2,
             label=f"Utility=0 (acc/epoch={acc_per_epoch})")

    plt.axhline(0, linestyle="-", color="black", linewidth=0.8)
    plt.axvline(0, linestyle="-", color="black", linewidth=0.8)

    plt.xlabel("Epochs Saved (Early - Smart)", fontsize=12)
    plt.ylabel("Accuracy Improvement (Early - Smart)", fontsize=12)
    plt.title(f"Speed-Accuracy Trade-off - {title}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===== ГРАФИК 4: Win rate =====
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
    plt.show()

    return summary, df_curr


# ========================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ========================

if __name__ == "__main__":
    from pytorch_version import MLPForMNIST, CNNForCIFAR, SimpleMLPRegression

    print("\n" + "=" * 60)
    print("PyTorch Early Stopping Benchmark")
    print("=" * 60)

    # SmartEarlyStopping по умолчанию использует линейный тренд, а не мета-модель
    # (мета-модель из final.csv не переносится на свежие кривые — см. комментарий
    # в pytorch_version.py). Поэтому meta здесь не обучаем.
    model_meta = None

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

    sizes = [500, 1000, 2000, 5000]

    # MNIST Benchmark
    print("\n" + "=" * 60)
    print("MNIST Benchmark")
    print("=" * 60)
    df_mnist = runner_pytorch(
        sizes, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist,
        MLPForMNIST, is_classifier=True, batch_size=16, epochs=200, model_meta=model_meta
    )

    summary_mnist, df_mnist = analyze_results(df_mnist, sizes, title="MNIST")

    # CIFAR-10 Benchmark
    print("\n" + "=" * 60)
    print("CIFAR-10 Benchmark")
    print("=" * 60)

    df_cifar = runner_pytorch(
        sizes, X_train_cifar, y_train_cifar, X_test_cifar, y_test_cifar,
        CNNForCIFAR, is_classifier=True, batch_size=16, epochs=200, model_meta=model_meta
    )

    summary_cifar, df_cifar = analyze_results(df_cifar, sizes, title="CIFAR-10")

    # Wine — задача регрессии (предсказываем оценку качества как непрерывную величину)
    wine_df = pd.read_csv('data/WineQT.csv')
    wine_features = (
        'fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,'
        'free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol'.split(',')
    )
    X_wine = wine_df[wine_features].values.astype('float32')
    y_wine = wine_df['quality'].values.astype('float32')

    scaler = StandardScaler()
    X_wine = scaler.fit_transform(X_wine).astype('float32')

    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )
    df_wine = runner_pytorch(
        sizes, X_train_wine, y_train_wine, X_test_wine, y_test_wine,
        SimpleMLPRegression, is_classifier=False, batch_size=16, epochs=200, model_meta=model_meta
    )
    summary_wine, df_wine = analyze_results(df_wine, sizes, title="Wine")

    # Комбинированный анализ
    print("\n" + "=" * 60)
    print("Combined Analysis")
    print("=" * 60)

    df_mnist["dataset"] = "MNIST"
    df_cifar["dataset"] = "CIFAR-10"
    df_wine["dataset"] = "Wine"
    df_all = pd.concat([df_mnist, df_cifar, df_wine], ignore_index=True)

    summary_all, df_all = analyze_results(df_all, sizes, title="All Datasets")

    # Сохранение результатов
    print("\n[Saving results...]")
    df_all.to_csv("early_stopping_results.csv", index=False)
    summary_all.to_csv("early_stopping_summary.csv")

    print("\n✅ Benchmark complete!")
